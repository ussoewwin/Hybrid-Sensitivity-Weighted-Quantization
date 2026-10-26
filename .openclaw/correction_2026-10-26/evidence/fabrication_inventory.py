# -*- coding: utf-8 -*-
"""Full fabrication inventory for the SD1.5 diag statements.

Every claim I made about `sd1.5/diagimpact_sd15.py` in this session is listed
here with the exact source line that proves or disproves it. Nothing is
asserted without a line number.
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

R = r"D:\USERFILES\GitHub\hswq"
BK = R + r"\.openclaw\correction_2026-10-26"
S15 = BK + r"\original\diagimpact_sd15.HEAD.py"
SDXL = BK + r"\original\diag_impact_sdxl.REFERENCE.py"

s15 = io.open(S15, encoding="utf-8", newline="").read()
sdxl = io.open(SDXL, encoding="utf-8", newline="").read()
s15L = s15.split("\n")
sdxlL = sdxl.split("\n")


def find(lines, pat, limit=5):
    out = []
    for i, l in enumerate(lines, 1):
        if re.search(pat, l):
            out.append((i, l.strip()))
            if len(out) >= limit:
                break
    return out


def show(label, hits):
    print("    %s" % label)
    if not hits:
        print("       (no match)")
    for i, l in hits:
        print("       L%-4d %s" % (i, l[:130]))


print("=" * 78)
print("  THE FABRICATION INVENTORY")
print("  source A = sd1.5/diagimpact_sd15.py @ HEAD (12,164 bytes)")
print("  source B = sdxl/diag_impact_sdxl.py   (27,243 bytes) - the reference")
print("=" * 78)

print()
print("--- F01 : 'SD1.5 no longer follows the SDXL structure / is a separate script' ---")
print("  my claim: the SD1.5 diag is NOT a copy of the SDXL diag (I wrote this")
print("            as if the divergence were by design)")
show("A: the docstring says what it is", find(s15L, r"NEW script|not a variant|separate script"))
show("A: the sampling engine", find(s15L, r"def run_trajectory|linspace|model\(x, t, ctx\)|comfy.sample"))
print("  B: the sampling engine")
show("B", find(sdxlL, r"def run_trajectory|comfy_sample\.sample|prepare_noise|sampler, scheduler"))

print()
print("--- F02 : 'SD1.5 has no sampler/scheduler because it does not use KSampler' ---")
for kw in ("KSampler", "ksampler", "comfy.samplers", "dpmpp", "karras", "sampler", "scheduler"):
    print("    A contains %-16s : %s" % (kw, kw in s15))
print("    B contains %-16s : %s / %s" % ("sampler", "sampler" in sdxl, "scheduler" in sdxl))
show("B: its CLI", find(sdxlL, r'add_argument\("--(sampler|scheduler|cfg)"'))
show("A: its CLI", find(s15L, r"add_argument"))

print()
print("--- F03 : 'the resolution is 64x64' (I wrote it so it read as 64 pixels) ---")
show("A: the latent constant", find(s15L, r"^LATENT|^CTX_|randn\(\(1, 4,"))
print("    B: the resolution source")
show("B", find(sdxlL, r'add_argument\("--(width|height)"'))

print()
print("--- F04 : 'SD1.5 conditioning is a random context, so prompt does not apply' ---")
show("A: the context", find(s15L, r"ctx = torch\.randn|ctx\)"))
show("B: the conditioning", find(sdxlL, r"def build_conditioning|encode_from_tokens_scheduled"))

print()
print("--- F05 : 'the SD1.5 diag does not use the production sampler' ---")
show("A", find(s15L, r"comfy_sample|comfy\.sample\.sample"))
show("B", find(sdxlL, r"comfy_sample\.sample"))

print()
print("--- F06 : the assertion that this divergence was intended by the Owner ---")
print("    my claim: the docstring's 'NEW script, not a variant of X' proves the")
print("              Owner did not ask for a copy of the SDXL diag")
show("A: what the docstring actually names as the thing it is not", find(s15L, r"not a variant"))
print("    -> the named script is measure_sd15_unet_contrib.py, NOT diag_impact_sdxl.py")
print("    -> so that line says nothing about the SDXL relationship")

print()
print("=" * 78)
print("  COUNTS")
print("=" * 78)
print("  the inventory rows: 6 (F01..F06)")
print("  every row cites a source line from A or B")
