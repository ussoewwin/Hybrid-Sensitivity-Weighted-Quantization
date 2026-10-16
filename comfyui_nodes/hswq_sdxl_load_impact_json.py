"""HSWQ node: load an existing SDXL diag impact json (reuse across runs).

The diag impact measurement is the slow stage of the SDXL reverse-hybrid
pipeline (715 pool layers x 25-step trajectories). Once measured for a
checkpoint (fixed --steps 25, fixed seed), the impact json is checkpoint-
specific and reusable as the Step 2 input. This node loads such a json from
disk and hands its path to the HSWQ Quantize node (model_type=SDXL) without
re-running the measurement.

Validation (refuses loudly, never passes through silently):
- the file must parse as JSON
- must contain a non-empty "impacts" dict (layer -> relative MSE)
- must record "steps" and "seed" (the ranking is only valid for the
  trajectory parameters it was measured with)

Default path (no explicit input): <repo>/impact/impact_<checkpoint-stem>.json
— the same convention the diag node writes.
"""
from __future__ import annotations

import json
import os
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMPACT_DIR = os.path.join(_REPO_ROOT, "impact")


def _model_base_path(model) -> str:
    """Checkpoint path of a loaded MODEL, as recorded by the loader."""
    try:
        if getattr(model, "cached_patcher_init", None):
            _func, args = model.cached_patcher_init[:2]
            if args and isinstance(args, tuple) and isinstance(args[0], str):
                return args[0]
    except Exception:
        pass
    return ""


class HSWQLoadDiagImpactJson:
    """Load a stored SDXL diag impact json for reuse by the quantize node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_path": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Path to the impact json (blank = auto: <repo>/impact/impact_<checkpoint-stem>.json)."}),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("impact_json", "report")
    FUNCTION = "load"
    CATEGORY = "HSWQ/Quantize"
    TITLE = "HSWQ Load Diag Impact Json"
    OUTPUT_NODE = False

    def load(self, json_path, model=None):
        path = (json_path or "").strip()
        if not path and model is not None:
            base = _model_base_path(model)
            if base:
                stem = os.path.splitext(os.path.basename(base))[0]
                path = os.path.join(_IMPACT_DIR, f"impact_{stem}.json")
        if not path:
            raise ValueError(
                "json_path: no path given and the MODEL input could not resolve "
                "<repo>/impact/impact_<stem>.json. Provide the impact json path "
                "explicitly."
            )
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"impact json not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise ValueError(f"impact json is not valid JSON: {path} ({e})")

        impacts = payload.get("impacts") if isinstance(payload, dict) else None
        if not isinstance(impacts, dict) or not impacts:
            raise ValueError(
                f"impact json has no usable 'impacts' dict: {path}. "
                "This file is not a diag impact json."
            )
        steps = payload.get("steps")
        seed = payload.get("seed")
        if steps is None or seed is None:
            raise ValueError(
                f"impact json does not record steps/seed: {path}. "
                "The ranking is only valid for the trajectory parameters it was "
                "measured with."
            )

        report = (
            f"impact json loaded: {path}\n"
            f"layers={len(impacts)} steps={steps} seed={seed} "
            f"cfg={payload.get('cfg')} sampler={payload.get('sampler')} "
            f"scheduler={payload.get('scheduler')}\n"
            f"base={payload.get('base')}"
        )
        print(f"[HSWQ Load Diag Impact Json] {report}", flush=True)
        return (path, report)


NODE_CLASS_MAPPINGS = {
    "HSWQLoadDiagImpactJson": HSWQLoadDiagImpactJson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQLoadDiagImpactJson": "HSWQ Load Diag Impact Json",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
