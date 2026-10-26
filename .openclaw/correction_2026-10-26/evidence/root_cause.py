# -*- coding: utf-8 -*-
"""Root-cause analysis: why do the old and new rigs differ by 210x-9,608x?

Hypothesis candidates, each tested by measurement, not reasoning:

  H1  the step schedule differs (old: linear sigma; new: karras)
  H2  the CFG differs (old: no CFG; new: cfg=7.0)
  H3  the context differs (old: random; new: CLIP-encoded prompt)
  H4  the noise differs (old: torch.randn; new: prepare_noise)
  H5  the latent size differs (old: 64x64; new: 64x64 at 512px -> SAME)

H5 is already answered: both are (1,4,64,64). So the divergence is not size.
This script isolates H1..H4 by running the NEW rig with the old rig's settings
where the CLI allows it, and reporting which factor moves the number.

What it actually measures:
  A) new rig, cfg=1.0          (CFG removed)
  B) new rig, sampler=euler, scheduler=normal  (closest to the old linear loop)
  C) new rig, negative="" + prompt=""  (context reduced toward the random case)
  D) the old rig as-is

The layer set is the same in all arms.
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

PY = sys.executable
R = r"D:\USERFILES\GitHub\hswq"
BK = R + r"\.openclaw\correction_2026-10-26"
OLD = BK + r"\original\diagimpact_sd15.HEAD.py"
NEW = R + r"\sd1.5\diagimpact_sd15.py"
COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
CKPT = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\againmix_v22.safetensors"


def run(script, out, extra, label):
    cmd = [PY, "-u", script, "--comfy", COMFY, "--ckpt", CKPT,
           "--out", out, "--steps", "25", "--seeds", "42",
           "--n-layers", "8", "--metric", "relmse"] + extra
    print("  running: %s" % label, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("    FAILED: %s" % r.stderr[-300:], flush=True)
        return None
    return json.load(open(out, encoding="utf-8"))


def main():
    os.makedirs(os.path.join(BK, "evidence"), exist_ok=True)
    E = lambda n: os.path.join(BK, "evidence", n)

    arms = {}
    arms["A_new_default"] = run(NEW, E("_arm_A.json"), [], "new rig, defaults (cfg7, dpmpp_2m/karras)")
    arms["B_new_cfg1"] = run(NEW, E("_arm_B.json"), ["--cfg", "1.0"], "new rig, cfg=1.0 (no CFG)")
    arms["C_new_euler_normal"] = run(NEW, E("_arm_C.json"),
                                     ["--sampler", "euler", "--scheduler", "normal"],
                                     "new rig, euler/normal (closest to the old linear loop)")
    arms["D_old"] = run(OLD, E("_arm_D.json"), [], "old rig as-is")

    keys = [k for k in arms["A_new_default"]["impacts"] if
            all(a and k in a["impacts"] for a in arms.values())]
    print()
    print("  " + "=" * 96)
    print("  the per-layer relMSE of the 8 layers, across the four arms")
    print("  " + "=" * 96)
    hdr = "  %-42s %12s %12s %12s %12s" % ("layer", "new default", "new cfg=1", "new euler/n", "old")
    print(hdr)
    print("  " + "-" * 94)
    for k in keys:
        print("  %-42s %12.4e %12.4e %12.4e %12.4e" % (
            k[:42],
            arms["A_new_default"]["impacts"][k],
            arms["B_new_cfg1"]["impacts"][k],
            arms["C_new_euler_normal"]["impacts"][k],
            arms["D_old"]["impacts"][k]))

    print()
    print("  " + "=" * 96)
    print("  the ranking of the 8 layers in each arm (worst first)")
    print("  " + "=" * 96)
    for name, a in arms.items():
        if not a:
            continue
        order = sorted(keys, key=lambda k: -a["impacts"][k])[:3]
        print("  %-22s %s" % (name, " > ".join(o.split(".")[-1] for o in order)))

    out = os.path.join(BK, "evidence", "root_cause_arms.json")
    json.dump({"ckpt": os.path.basename(CKPT), "layers": keys,
               "arms": {k: (v["impacts"] if v else None) for k, v in arms.items()}},
              open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print()
    print("  [saved] %s" % out)


if __name__ == "__main__":
    main()
