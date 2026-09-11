# Proof: removing external bench imports did not break functionality (2026-09-11)

Generated after commits `206f7c1` / `93ea195` / `def649a`.
Scope: 9 scripts in the HSWQ repo that previously loaded a sibling `*_bench.py`
at runtime via `importlib.util.spec_from_file_location(...)` + `exec_module(...)`.

Each script is now self-contained: the helper functions it used to call on the
imported `bench` module are inlined, and the dynamic load block is removed.

This document is the evidence that the change is behaviour-preserving.

---

## A. Helper call sites: all preserved

For every script, the set of helper names it called as `bench.X` **before** the
change (`HEAD~1`) is compared with the calls present **after** inlining.

| Script | helpers before | lost after | result |
| :--- | ---: | :--- | :--- |
| `Krea2/calib_input_scale_nvfp4.py` | 1 | none | PASS |
| `Z_Image/calib_input_scale_nvfp4.py` | 2 | none | PASS |
| `Z_Image/diag_impact.py` | 3 | none | PASS |
| `Z_Image/diag_impact_old.py` | 3 | none | PASS |
| `Flux1/calib_input_scale_nvfp4.py` | 3 | none (1 renamed by the bench earlier) | PASS |
| `Flux1/diag_impact.py` | 3 | none (1 renamed by the bench earlier) | PASS |
| `benchmark/krea2_int8_traj_compare.py` | 9 | none | PASS |
| `benchmark/qwen_image_edit_traj_compare.py` | 9 | none | PASS |
| `benchmark/flux1_nvfp4/flux_traj_compare.py` | 9 | none | PASS |

**Note on the FLUX rename.** Both `Flux1` scripts called `bench.apply_int8_patches()`.
That name no longer exists: the FLUX bench renamed it to `apply_quant_patches()`
in commit `7969e48` ("rebuild flux_int8_bench.py as new-gen"). The call sites were
never updated, so the pre-change code raised `AttributeError` when it reached that
line. This change points the call at the bench's real function name. The callee
body is byte-identical to the bench original (see section B).

---

## B. Inlined helper bodies are byte-identical to the bench originals

Each inlined function's source text is compared, character for character, with
the same function in its bench source file.

| Script | functions compared | byte-identical | differing |
| :--- | ---: | ---: | :--- |
| `Krea2/calib_input_scale_nvfp4.py` | 2 | 2 | none |
| `Z_Image/calib_input_scale_nvfp4.py` | 11 | 11 | none |
| `Z_Image/diag_impact.py` | 11 | 11 | none |
| `Z_Image/diag_impact_old.py` | 11 | 11 | none |
| `Flux1/calib_input_scale_nvfp4.py` | 4 | 4 | none |
| `Flux1/diag_impact.py` | 4 | 4 | none |
| `benchmark/krea2_int8_traj_compare.py` | 10 | 10 | none |
| `benchmark/qwen_image_edit_traj_compare.py` | 10 | 10 | none |
| `benchmark/flux1_nvfp4/flux_traj_compare.py` | 10 | 10 | none |
| **Total** | **73** | **73** | **0** |

No function body was modified.

---

## C. No external bench load remains / no undefined names

| Script | external bench loads | undefined names |
| :--- | ---: | :--- |
| all 9 scripts | **0** | **none** |

Additionally:

- `python -m py_compile` succeeds for all 9 files.
- `python <script> --help` prints a valid usage line for all 9 files.

---

## Verdict

```
OVERALL: ALL PROOFS PASS
```

Removing the external bench imports did not lose any functionality: the 73
inlined helpers are byte-identical to their originals, and every helper the
scripts used to call is still called (one call was corrected to the bench's
current function name, where the pre-existing code was already broken).

---

## Reproduce

```powershell
# A: helper call sets (before vs after)
git -C D:\USERFILES\GitHub\hswq show "HEAD~1:Krea2/calib_input_scale_nvfp4.py" | Select-String 'bench\.\w+'
Select-String -Path 'D:\USERFILES\GitHub\hswq\Krea2\calib_input_scale_nvfp4.py' -Pattern 'setup_comfy\('

# B: one inlined function vs its bench original (example)
Select-String -Path 'D:\USERFILES\GitHub\hswq\Krea2\calib_input_scale_nvfp4.py' -Pattern 'def setup_comfy' -Context 0,70
Select-String -Path 'D:\USERFILES\GitHub\hswq\archives\krea2_convrot_nvfp4_bench.py' -Pattern 'def setup_comfy' -Context 0,70

# C: no external load / all files compile and show help
Select-String -Path 'D:\USERFILES\GitHub\hswq\**\*.py' -Pattern 'exec_module\(\s*bench'
D:\USERFILES\ComfyUI\python_embeded\python.exe -m py_compile D:\USERFILES\GitHub\hswq\Krea2\calib_input_scale_nvfp4.py
D:\USERFILES\ComfyUI\python_embeded\python.exe D:\USERFILES\GitHub\hswq\Krea2\calib_input_scale_nvfp4.py --help
```
