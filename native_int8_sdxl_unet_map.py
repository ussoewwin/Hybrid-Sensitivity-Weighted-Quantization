"""SDXL ComfyUNet <-> Diffusers key map for native_convert_int8_simple Card1.

Self-contained detect + mapping helpers. Runtime does NOT import
quantize_sdxl_hswq_v3.0.py.
"""
def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        return last_transformer_depth
    return 0

def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
    filtered_keys = [k for k in state_dict_keys if k.startswith(key_prefix)]
    unet_config = {}
    if f"{key_prefix}input_blocks.0.0.weight" in state_dict_keys:
        model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
        num_res_blocks = []
        channel_mult = []
        transformer_depth = []
        transformer_depth_output = []
        input_block_count = count_blocks(state_dict_keys, f"{key_prefix}input_blocks" + '.{}.')
        last_res_blocks = 0
        last_channel_mult = 0
        for count in range(input_block_count):
            prefix = f"{key_prefix}input_blocks.{count}."
            prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
            block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
            if len(block_keys) == 0: break
            block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))
            if f"{prefix}0.op.weight" in block_keys:
                num_res_blocks.append(last_res_blocks)
                channel_mult.append(last_channel_mult)
                last_res_blocks = 0
                last_channel_mult = 0
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                transformer_depth_output.append(out)
            else:
                res_block_prefix = f"{prefix}0.in_layers.0.weight"
                if res_block_prefix in block_keys:
                    last_res_blocks += 1
                    last_channel_mult = state_dict[f"{prefix}0.out_layers.3.weight"].shape[0] // model_channels
                    out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                    transformer_depth.append(out)
                res_block_prefix = f"{prefix_output}0.in_layers.0.weight"
                if res_block_prefix in block_keys_output:
                    out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                    transformer_depth_output.append(out)
        num_res_blocks.append(last_res_blocks)
        channel_mult.append(last_channel_mult)
        if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
            transformer_depth_middle = count_blocks(state_dict_keys, f"{key_prefix}middle_block.1.transformer_blocks." + '{}')
        elif f"{key_prefix}middle_block.0.in_layers.0.weight" in state_dict_keys:
            transformer_depth_middle = -1
        else:
            transformer_depth_middle = -2
        unet_config["num_res_blocks"] = num_res_blocks
        unet_config["channel_mult"] = channel_mult
        unet_config["transformer_depth"] = transformer_depth
        unet_config["transformer_depth_output"] = transformer_depth_output
        unet_config["transformer_depth_middle"] = transformer_depth_middle
    return unet_config

def unet_to_diffusers_mapping(unet_config, state_dict=None, key_prefix="model.diffusion_model."):
    if "num_res_blocks" not in unet_config: return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    num_blocks = len(channel_mult)
    if state_dict is not None:
        import re
        state_dict_keys = list(state_dict.keys())
        filtered_keys = [k.replace(key_prefix, "") for k in state_dict_keys if k.startswith(key_prefix)]
        transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in transformer_counts: transformer_counts[block_idx] = 0
                transformer_counts[block_idx] = max(transformer_counts[block_idx], trans_idx + 1)
        output_transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in output_transformer_counts: output_transformer_counts[block_idx] = 0
                output_transformer_counts[block_idx] = max(output_transformer_counts[block_idx], trans_idx + 1)
        middle_transformer_count = 0
        for key in filtered_keys:
            match = re.match(r'middle_block\.1\.transformer_blocks\.(\d+)', key)
            if match:
                trans_idx = int(match.group(1))
                middle_transformer_count = max(middle_transformer_count, trans_idx + 1)
        transformers_mid = middle_transformer_count if middle_transformer_count > 0 else unet_config.get("transformer_depth_middle", None)
    else:
        transformer_depth = unet_config["transformer_depth"][:]
        transformer_depth_output = unet_config["transformer_depth_output"][:]
        transformers_mid = unet_config.get("transformer_depth_middle", None)
        transformer_counts = None
        output_transformer_counts = None
    UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias", "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias", "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias", "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias", "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias", "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias"}
    UNET_MAP_ATTENTIONS = {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"}
    TRANSFORMER_BLOCKS = {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias", "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias"}
    UNET_MAP_BASIC = {("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"), ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"), ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"), ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"), ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"), ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"), ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias")}
    # Map only tensors present in this checkpoint's state_dict (auto from weights).
    # No invented Diffusers names, no fixed KEEP list, no inject.
    if state_dict is None:
        raise RuntimeError(
            "unet_to_diffusers_mapping requires state_dict; refuse maps without "
            "Comfy presence checks"
        )
    _sd_keys = set(state_dict.keys())
    _comfy_bare = {
        (k[len(key_prefix):] if k.startswith(key_prefix) else k)
        for k in _sd_keys
    }

    def _comfy_present(comfy_bare: str) -> bool:
        return comfy_bare in _comfy_bare or f"{key_prefix}{comfy_bare}" in _sd_keys

    def _map_put(diff_key: str, comfy_bare: str) -> None:
        if not _comfy_present(comfy_bare):
            return
        diffusers_unet_map[diff_key] = comfy_bare

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                _map_put(
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "input_blocks.{}.0.{}".format(n, b),
                )
            if transformer_counts is not None: num_transformers = transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "input_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            n += 1
        # Last DownBlock has no downsampler in SDXL  -  register only if op exists.
        if _comfy_present("input_blocks.{}.0.op.weight".format(n)):
            for k in ["weight", "bias"]:
                _map_put(
                    "down_blocks.{}.downsamplers.0.conv.{}".format(x, k),
                    "input_blocks.{}.0.op.{}".format(n, k),
                )
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        _map_put("mid_block.attentions.{}.{}".format(i, b), "middle_block.1.{}".format(b))
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS:
                _map_put(
                    "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b),
                    "middle_block.1.transformer_blocks.{}.{}".format(t, b),
                )
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            _map_put(
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b]),
                "middle_block.{}.{}".format(n, b),
            )
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                _map_put(
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "output_blocks.{}.0.{}".format(n, b),
                )
            c += 1
            if output_transformer_counts is not None: num_transformers = output_transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "output_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            # Upsample: only if this checkpoint has that Comfy conv (presence).
            # Missing tensor → no Diffusers entry.
            if i == l - 1:
                for k in ["weight", "bias"]:
                    _map_put(
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k),
                        "output_blocks.{}.{}.conv.{}".format(n, c, k),
                    )
            n += 1
    for k, v in UNET_MAP_BASIC:
        _map_put(v, k)
    for _dk, _ck in diffusers_unet_map.items():
        if not _comfy_present(_ck):
            raise RuntimeError(
                f"Map integrity FATAL: mapped Comfy key {_ck!r} absent in checkpoint"
            )
    comfyui_to_diffusers_map = {v: k for k, v in diffusers_unet_map.items()}
    comfyui_to_diffusers_map = {f"{key_prefix}{k}": v for k, v in comfyui_to_diffusers_map.items()}

    return comfyui_to_diffusers_map
