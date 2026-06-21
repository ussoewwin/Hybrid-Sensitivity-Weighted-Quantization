"""Quick import check for quantize_sdxl_hswq_v2.0.py"""
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "quantize_sdxl_hswq_v2.0.py"
spec = importlib.util.spec_from_file_location("quantize_sdxl_hswq_v2", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["quantize_sdxl_hswq_v2"] = mod
spec.loader.exec_module(mod)

remap = mod._remap_profile_to_diffusers(
    {"model.diffusion_model.foo.weight": {"kurtosis": 12.0, "outlier_ratio": 5.0, "abs_max": 1.0}},
    {"model.diffusion_model.foo.weight": "down_blocks.0.bar.weight"},
)
assert "down_blocks.0.bar" in remap and remap["down_blocks.0.bar"]["kurtosis"] == 12.0

# derive_hswq_strategy must consume remapped keys (Diffusers module names, no .weight suffix)
alpha, beta, get_layer_search_low, hard_veto = mod.derive_hswq_strategy(dict(remap))
assert 0.5 <= alpha <= 0.99 and 0.5 <= beta <= 0.99
assert all(not n.endswith(".weight") for n in hard_veto)
import torch

t = torch.randn(8, 8)
low = get_layer_search_low("down_blocks.0.bar", t)
assert 0.5 <= low <= 0.99

extreme = mod._remap_profile_to_diffusers(
    {
        "model.diffusion_model.veto.weight": {
            "kurtosis": 25.0,
            "outlier_ratio": 10.0,
            "abs_max": 5.0,
        }
    },
    {"model.diffusion_model.veto.weight": "mid_block.attentions.0.veto.weight"},
)
_, _, _, hard_veto2 = mod.derive_hswq_strategy(extreme)
assert "mid_block.attentions.0.veto" in hard_veto2

print("import OK", path.stat().st_size, "bytes", len(path.read_text(encoding="utf-8").splitlines()), "lines")
print("remap smoke OK")
print("derive smoke OK")