# -*- coding: utf-8 -*-
"""Old vs new: measured on the same checkpoint, same seed, same step count.

Old (HEAD, backed up) used its own Euler loop on a random context.
New (working tree) uses comfy.sample.sample with the production sampler.

This script runs BOTH and records, per layer, the impact value each produces.
The point is not that they agree - they are different rigs - but that the new
one is the production trajectory and the old one is not.

Run:
  python measure_old_vs_new.py --ckpt <sd1.5 ckpt> --layers 5
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

PY = sys.executable
R = r"D:\USERFILES\GitHub\hswq"
BK = R + r"\.openclaw\correction_2026-10-26"
OLD = BK + r"\original\diagimpact_sd15.HEAD.py"
NEW = R + r"\sd1.5\diagimpact_sd15.py"
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out = a.out or os.path.join(BK, "evidence", "old_vs_new.json")
    results = {"ckpt": os.path.basename(a.ckpt), "seed": a.seed, "steps": a.steps,
               "layers": a.layers, "arms": {}}

    # --- the OLD arm ------------------------------------------------------
    print("=" * 70)
    print("  ARM 1: the OLD script (its own Euler loop)")
    print("=" * 70, flush=True)
    old_out = os.path.join(BK, "evidence", "_old_arm.json")
    cmd = [PY, "-u", OLD, "--comfy", COMFY, "--ckpt", a.ckpt,
           "--out", old_out, "--steps", str(a.steps),
           "--seeds", str(a.seed), "--n-layers", str(a.layers),
           "--metric", "relmse"]
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print("  the OLD arm FAILED rc=%d" % r.returncode, flush=True)
        print(r.stderr[-1500:], flush=True)
        results["arms"]["old"] = {"error": r.stderr[-800:]}
    else:
        d = json.load(open(old_out, encoding="utf-8"))
        results["arms"]["old"] = {
            "metric": d.get("metric"),
            "engine": "self-written Euler loop (torch.linspace, model(x,t,ctx))",
            "sampler": None,
            "scheduler": None,
            "cfg": None,
            "width": None,
            "height": None,
            "prompt_used": False,
            "impacts": d["impacts"],
        }

    # --- the NEW arm ------------------------------------------------------
    print()
    print("=" * 70)
    print("  ARM 2: the NEW script (comfy.sample.sample)")
    print("=" * 70, flush=True)
    new_out = os.path.join(BK, "evidence", "_new_arm.json")
    cmd = [PY, "-u", NEW, "--comfy", COMFY, "--ckpt", a.ckpt,
           "--out", new_out, "--steps", str(a.steps),
           "--seeds", str(a.seed), "--n-layers", str(a.layers),
           "--metric", "relmse"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        print("  the NEW arm FAILED rc=%d" % r.returncode, flush=True)
        print(r.stderr[-1500:], flush=True)
        results["arms"]["new"] = {"error": r.stderr[-800:]}
    else:
        d = json.load(open(new_out, encoding="utf-8"))
        results["arms"]["new"] = {
            "metric": d.get("metric"),
            "engine": "comfy.sample.sample (the production sampler)",
            "sampler": d.get("sampler"),
            "scheduler": d.get("scheduler"),
            "cfg": d.get("cfg"),
            "width": d.get("width"),
            "height": d.get("height"),
            "prompt_used": True,
            "impacts": d["impacts"],
        }

    # --- the comparison ---------------------------------------------------
    print()
    print("=" * 70)
    print("  THE COMPARISON")
    print("=" * 70)
    o = results["arms"].get("old", {})
    n = results["arms"].get("new", {})
    print("  %-42s %-18s %-18s" % ("layer", "old (euler/random)", "new (production)"))
    print("  " + "-" * 78)
    common = [k for k in o.get("impacts", {}) if k in n.get("impacts", {})]
    for k in common:
        print("  %-42s %-18.6e %-18.6e" % (k[:42], o["impacts"][k], n["impacts"][k]))
    print()
    print("  the layers compared: %d" % len(common))
    # the ranking agreement (Spearman-style: do they order the same?)
    if len(common) >= 3:
        oo = sorted(common, key=lambda k: -o["impacts"][k])
        nn = sorted(common, key=lambda k: -n["impacts"][k])
        same_top = oo[0] == nn[0]
        print("  the top layer: old=%s new=%s  agree=%s" % (oo[0][:40], nn[0][:40], same_top))

    json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print()
    print("  [saved] %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
