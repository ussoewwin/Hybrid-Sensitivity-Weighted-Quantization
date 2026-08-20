"""HSWQ (Hybrid Sensitivity Weighted Quantization) custom node package.

This repository is cloned directly into <ComfyUI>/custom_nodes/. ComfyUI loads
the repo root __init__ via importlib spec (sanitized module name, repo dir not
on sys.path), so register the repo root first, then aggregate the node
mappings from comfyui_nodes/.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
