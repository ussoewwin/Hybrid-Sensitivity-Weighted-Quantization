import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Hadamard / ConvRot helpers for sibling native_convert_int8_convrot.py.
# Default convert_to_int8 below stays plain tensorwise INT8 (no ConvRot).
_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(n: int, preferred: int = _DEFAULT_GROUPSIZE) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise)."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def unrotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def rotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Offline Conv2d: rotate along in_channels. weight (O, I, kH, kW)."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_rot = rotate_weight(flat, h_matrix, group_size)
    return flat_rot.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def unrotate_weight_conv2d(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of rotate_weight_conv2d."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be 4D, got ndim={weight.ndim}")
    out_c, in_c, k_h, k_w = weight.shape
    flat = weight.permute(0, 2, 3, 1).contiguous().view(-1, in_c)
    flat_un = unrotate_weight(flat, h_matrix, group_size)
    return flat_un.view(out_c, k_h, k_w, in_c).permute(0, 3, 1, 2).contiguous()


def rotate_activation(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Linear: x_rot = x @ H (last dim = features)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    h = h_matrix.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def rotate_activation_nchw(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Conv2d: rotate channel dim of NCHW activation."""
    if x.ndim != 4:
        raise ValueError(f"NCHW activation must be 4D, got ndim={x.ndim}")
    x_perm = x.permute(0, 2, 3, 1).contiguous()
    x_rot = rotate_activation(x_perm, h_matrix, group_size)
    return x_rot.permute(0, 3, 1, 2).contiguous()


def convert_to_int8(input_path, output_path):
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)

    new_state_dict = {}
    quant_meta_layers = {}
    converted_count = 0
    skipped_count = 0

    print("Converting UNet Linear/Conv weights to INT8 (tensorwise, amax/127)...")

    for key, tensor in tqdm(state_dict.items()):
        # Target only SDXL UNet Linear/Conv weights (ndim >= 2).
        # Do NOT INT8 1D .weight (GroupNorm / LayerNorm): those modules are not
        # MixedPrecisionOps and would load int8 as float without applying
        # weight_scale, destroying SSIM (e.g. ~0.0003). HSWQ only touches
        # nn.Linear / nn.Conv2d for the same reason.
        is_unet_matmul_weight = (
            key.startswith("model.diffusion_model")
            and key.endswith(".weight")
            and tensor.ndim >= 2
        )
        if is_unet_matmul_weight and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ]:
            # Symmetric per-tensor INT8 (ComfyUI int8_tensorwise):
            #   scale = amax / 127
            #   q = round(x / scale).clamp(-127, 127).to(int8)
            w = tensor.float()
            amax = w.abs().max().item()
            amax = max(amax, 1e-6)
            scale = amax / 127.0
            q = (w / scale).round().clamp(-127, 127).to(torch.int8)

            module_key = key[: -len(".weight")]
            new_state_dict[key] = q
            new_state_dict[f"{module_key}.weight_scale"] = torch.tensor(
                scale, dtype=torch.float32
            )
            new_state_dict[f"{module_key}.comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
                dtype=torch.uint8,
            )
            quant_meta_layers[module_key] = {"format": "int8_tensorwise"}
            converted_count += 1
        else:
            # Keep norms, biases, CLIP, VAE, and non-float tensors as-is.
            new_state_dict[key] = tensor
            skipped_count += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_meta_layers}
        )
    }

    print(f"Saving to: {output_path}")
    print(f"Converted layers: {converted_count}, Kept layers: {skipped_count}")

    save_file(new_state_dict, output_path, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to input .safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output .safetensors")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        exit(1)

    convert_to_int8(args.model, args.output)
