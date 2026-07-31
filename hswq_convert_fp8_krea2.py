#!/usr/bin/env python3
"""
krea2 (moodyKrea2Mix_v40BF16) FP8 変換スクリプト
方針: SeedVR2 A2 の FP16保護 / FP8キャスト考え方を krea2 アーキに適用

FP16 保護する層:
  - IO / stem 系: first, last.linear, tmlp.*, tproj.*, txtmlp.*
  - blocks 0 と 27 (最初・最後ブロック) の全 Linear
  - 全ブロックの attn.wo (出力プロジェクション)
  - 全ブロックの mlp.down (MLP 出力側)

FP8 (e4m3fn) キャスト:
  - 上記以外の blocks.N.attn.* と blocks.N.mlp.{gate,up}

非 Linear テンソル (norm scale, mod.lin 等) は dtype 変換しないでコピー。

Do not run convert-to-disk unless the same message explicitly orders a run.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file, save_file


# --------------------------------------------------------------------------- #
# FP16 保護セット
# --------------------------------------------------------------------------- #
def _is_fp16_protected(key: str) -> bool:
    """True → FP16 のままコピー（量子化しない）"""
    k = key

    # IO / stem 系
    if (
        k.startswith("model.diffusion_model.first.")
        or k.startswith("model.diffusion_model.last.")
        or k.startswith("model.diffusion_model.tmlp.")
        or k.startswith("model.diffusion_model.tproj.")
        or k.startswith("model.diffusion_model.txtmlp.")
        or k.startswith("model.diffusion_model.txtfusion.")
    ):
        return True

    # blocks.0 と blocks.27 (最初・最後) は全保護
    if "blocks.0." in k or "blocks.27." in k:
        return True

    # 全ブロックの出力プロジェクション
    if k.endswith(".attn.wo.weight"):
        return True
    if k.endswith(".mlp.down.weight"):
        return True

    return False


# --------------------------------------------------------------------------- #
# メイン変換
# --------------------------------------------------------------------------- #
def convert(src_path: str, dst_path: str, fp8_dtype_str: str = "e4m3fn") -> None:
    fp8_dtype = torch.float8_e4m3fn if fp8_dtype_str == "e4m3fn" else torch.float8_e5m2

    print(f"Loading {src_path} ...")
    sd = load_file(src_path)

    out: dict[str, torch.Tensor] = {}
    stats = {"fp16_kept": 0, "fp8_cast": 0, "other_copy": 0}

    for key, tensor in sd.items():
        # 2D (Linear weight) 以外はコピー
        if tensor.dim() != 2:
            out[key] = tensor.clone()
            stats["other_copy"] += 1
            continue

        # 非常に小さいテンソル (e.g. txtfusion.projector (1,12)) はコピー
        if tensor.numel() < 64:
            out[key] = tensor.clone()
            stats["other_copy"] += 1
            continue

        if _is_fp16_protected(key):
            # FP16 保護: bfloat16 → float16 に変換して保存
            out[key] = tensor.to(dtype=torch.float16)
            stats["fp16_kept"] += 1
        else:
            # FP8 キャスト
            # amax を使って scale を求め、スケール付き保存
            amax = tensor.abs().max().float()
            fp8_max = 448.0  # e4m3fn の最大値
            scale = (fp8_max / amax.clamp(min=1e-12)).float()
            quantized = (tensor.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)

            stem = key  # e.g. "model.diffusion_model.blocks.1.attn.wq.weight"
            out[stem] = quantized
            # weight_scale: shape [1] の float32 (逆スケール = 1/scale)
            out[stem.replace(".weight", ".weight_scale")] = (1.0 / scale).reshape(1)
            stats["fp8_cast"] += 1

    print(f"FP16 kept : {stats['fp16_kept']}")
    print(f"FP8 cast  : {stats['fp8_cast']}")
    print(f"Other copy: {stats['other_copy']}")

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    print(f"Saving → {dst_path} ...")
    save_file(out, dst_path)
    print("Done.")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="krea2 FP8 converter")
    parser.add_argument(
        "--src",
        default=r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyKrea2Mix_v40BF16.safetensors",
    )
    parser.add_argument(
        "--dst",
        default=r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\moodyKrea2Mix_fp8_A2.safetensors",
    )
    parser.add_argument(
        "--fp8",
        choices=["e4m3fn", "e5m2"],
        default="e4m3fn",
        help="FP8 dtype (default: e4m3fn)",
    )
    args = parser.parse_args()
    convert(args.src, args.dst, args.fp8)


if __name__ == "__main__":
    main()
