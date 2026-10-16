"""HSWQ node: pick and load an SDXL diag impact json from analyze/json/.

The diag impact measurement is the slow stage of the SDXL reverse-hybrid
pipeline (715 pool layers x 25-step trajectories). Once measured for a
checkpoint, the impact json is checkpoint-specific and reusable as the Step 2
input. This node lists every "*.json" under ``<repo>/analyze/json/`` in a
dropdown and hands the selected file's absolute path to the HSWQ Quantize node
(model_type=SDXL) without re-running the measurement.

Validation (refuses loudly, never passes through silently):
- the selected file must parse as JSON
- must contain a non-empty "impacts" dict (layer -> relative MSE)
- must record "steps" and "seed" (the ranking is only valid for the
  trajectory parameters it was measured with)
"""
from __future__ import annotations

import json
import os
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANALYZE_JSON_DIR = os.path.join(_REPO_ROOT, "analyze", "json")


def _list_impact_jsons() -> list[str]:
    try:
        entries = os.listdir(_ANALYZE_JSON_DIR)
    except Exception:
        return []
    files = [e for e in entries if e.lower().endswith(".json")
             and os.path.isfile(os.path.join(_ANALYZE_JSON_DIR, e))]
    return sorted(files)


class HSWQLoadDiagImpactJson:
    """Select and load a stored SDXL diag impact json for reuse."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_impact_jsons()
        choices = files if files else ["(no json in analyze/json yet)"]
        return {
            "required": {
                "impact_json": (choices, {"default": choices[0],
                                           "tooltip": "Impact json under <repo>/analyze/json/ (from the HSWQ SDXL Diag Impact measurement)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())  # dropdown list must refresh when files change

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("impact_json", "report")
    FUNCTION = "load"
    CATEGORY = "HSWQ/Quantize"
    TITLE = "HSWQ Load Diag Impact Json"
    OUTPUT_NODE = False

    def load(self, impact_json):
        sel = (impact_json or "").strip()
        if not sel or sel.startswith("("):
            raise ValueError(
                "impact_json: no impact json is stored under "
                f"{_ANALYZE_JSON_DIR} yet. Run the HSWQ SDXL Diag Impact node "
                "once (its json lands under analyze/json) and reload this node."
            )
        path = os.path.abspath(os.path.join(_ANALYZE_JSON_DIR, sel))
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
