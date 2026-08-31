#!/usr/bin/env python3
"""Colab / cloud one-shot environment check + repair for the Krea2 diag & bench flow.

Run from the repo root on Google Colab:

    !python colab_setup.py            # check + auto-fix torch if it is a CPU build
    !python colab_setup.py --check    # check only; print fixes, change nothing

What it verifies:
  1. GPU runtime is visible (nvidia-smi)
  2. torch is a CUDA build (torch.version.cuda is set, torch.cuda.is_available())
  3. the repo-local ComfyUI-master/comfy/options.py shim exists (required by
     benchmark/krea2_convrot_nvfp4_bench.py -> import comfy.options)
  4. prints the exact diag_impact.py command for the Krea2 workflow

Why the torch check matters: a bare `pip install torch` on Colab replaces the
preinstalled CUDA build with a CPU wheel, after which any comfy.* import dies
with "No CUDA GPUs are available". If nvidia-smi sees a GPU but torch is a CPU
build, the default mode reinstalls the cu121 wheel set automatically. After a
reinstall, restart the runtime (Runtime -> Restart session) before benchmarking,
because the already-imported torch in this process is still the CPU build.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
COMFY_OPTIONS = os.path.join(REPO_ROOT, "ComfyUI-master", "comfy", "options.py")
COMFY_OPTIONS_SHIM = (
    "args_parsing = False\n"
    "\n"
    "def enable_args_parsing(enable=True):\n"
    "    global args_parsing\n"
    "    args_parsing = enable\n"
)
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def _run(cmd: list[str], echo: bool = True) -> subprocess.CompletedProcess:
    if echo:
        print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def check_gpu() -> bool:
    nv = shutil.which("nvidia-smi")
    if nv is None:
        print("[GPU] nvidia-smi NOT FOUND -> no GPU runtime.\n"
              "      Fix: Runtime -> Change runtime type -> T4 GPU, then reconnect.")
        return False
    r = _run([nv])
    if r.returncode != 0:
        print("[GPU] nvidia-smi failed -> GPU quota exhausted or runtime fell back to CPU.\n"
              "      Fix: Runtime -> Disconnect and delete runtime -> reconnect "
              "(free-tier quota), or switch accounts / Colab Pro.")
        return False
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith(("Tesla", "T4", "L4", "A100", "H100", "V100")):
            print("[GPU]", s)
            break
    print("[GPU] OK: CUDA-capable GPU visible")
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        print("[torch] torch NOT INSTALLED")
        return False
    ver = getattr(torch, "__version__", "?")
    cuda = getattr(torch.version, "cuda", None)
    avail = torch.cuda.is_available()
    print(f"[torch] version={ver} cuda_build={cuda} cuda_available={avail}")
    return cuda is not None and avail


def repair_torch() -> None:
    print("[torch] CPU build detected -> reinstalling CUDA build (cu121)...", flush=True)
    _run([sys.executable, "-m", "pip", "uninstall", "-y",
          "torch", "torchvision", "torchaudio"])
    r = _run([sys.executable, "-m", "pip", "install",
              "torch", "torchvision", "torchaudio",
              "--index-url", TORCH_INDEX])
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit("[torch] reinstall FAILED (see output above)")
    print("[torch] reinstalled. RESTART REQUIRED: Runtime -> Restart session, "
          "then re-run this script / the benchmark.", flush=True)


def check_comfyui() -> bool:
    if os.path.isfile(COMFY_OPTIONS):
        print(f"[comfy] OK: {os.path.relpath(COMFY_OPTIONS, REPO_ROOT)} present")
        return True
    print(f"[comfy] MISSING {COMFY_OPTIONS}\n"
          "      This repo's ComfyUI-master must be used (git pull / re-clone).\n"
          "      If your tree lacks it, write the shim manually:")
    print("      %%writefile " + COMFY_OPTIONS.replace("\\", "/"))
    for line in COMFY_OPTIONS_SHIM.splitlines():
        print("      " + line)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="check only; never modify the environment")
    a = ap.parse_args()

    gpu_ok = check_gpu()
    torch_ok = check_torch()

    if not torch_ok:
        if gpu_ok and not a.check:
            repair_torch()
            print("\n>>> Re-run this script after restarting the runtime to verify.")
            return 2
        print("[torch] FIX: !pip uninstall -y torch torchvision torchaudio")
        print(f"            !pip install torch torchvision torchaudio --index-url {TORCH_INDEX}")

    check_comfyui()

    print("\n[run] example:")
    print("      python Krea2/diag_impact.py <base.safetensors> <convrot_int8.safetensors> \\")
    print("          impact_krea2.json --comfy-path ComfyUI-master")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
