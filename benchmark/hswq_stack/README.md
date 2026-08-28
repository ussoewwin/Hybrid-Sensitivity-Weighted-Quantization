# hswq_stack — HSWQ hybrid NVFP4 / INT8 Z Image stack (vendored)

This package vendors the required implementation from **ComfyUI-HSWQ-Loader-and-Tools** (AGPL-3.0, ussoewwin) to execute Z-Image / ZIT hybrid ConvRot NVFP4 (INT8 protect + NVFP4 + ConvRot) using standard ComfyUI loading pipelines.

## Upstream Source

- Repository: https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools
- Vendoring Date: 2026-08-17 (re-synced from upstream 2026-08-28: TC W4A4 opt-in + purge guards)
- Directory Mapping:

| Upstream | Vendored Path |
|---|---|
| nodes/nvfp4/ | hswq_stack/nvfp4/ |
| nodes/zimage_nvfp4/ | hswq_stack/zimage_nvfp4/ |
| patches/comfy_quant_int8.py | hswq_stack/patches/comfy_quant_int8.py |

## Modifications During Vendoring

- Rewrote relative imports from `...patches` to `..patches` (since `hswq_stack` is the parent package, 3 dots would exceed top-level package boundary):
  - zimage_nvfp4/load_unet.py
  - zimage_nvfp4/nvfp4_lora_bake.py
  - nvfp4/comfy_quant_nvfp4.py
- All other code is identical to the original implementation (mathematical operations and patch targets are unchanged).

## License

The original implementation is licensed under **GNU Affero General Public License v3 (AGPL-3.0)**.
This vendored package is distributed under the same license.

## Synchronization Policy

When upstream is updated, re-vendor into this directory.
Do not rely on external imports (this benchmark suite must operate standalone).
