# 証拠: 外部 bench インポート除去が機能を壊していない証明（2026-09-11）

生成: 2026-09-11 12:22:30 / commit 93ea195

```text
==============================================================================
A) helper call sites: all preserved (except 2 pre-broken FLUX names, corrected)
==============================================================================
Krea2/calib_input_scale_nvfp4.py             helpers= 1 lost=none PASS 
Z_Image/calib_input_scale_nvfp4.py           helpers= 2 lost=none PASS 
Z_Image/diag_impact.py                       helpers= 3 lost=none PASS 
Z_Image/diag_impact_old.py                   helpers= 3 lost=none PASS 
Flux1/calib_input_scale_nvfp4.py             helpers= 3 lost=none PASS (renamed by bench before: ['apply_int8_patches'])
Flux1/diag_impact.py                         helpers= 3 lost=none PASS (renamed by bench before: ['apply_int8_patches'])
benchmark/krea2_int8_traj_compare.py         helpers= 9 lost=none PASS 
benchmark/qwen_image_edit_traj_compare.py    helpers= 9 lost=none PASS 
benchmark/flux1_nvfp4/flux_traj_compare.py   helpers= 9 lost=none PASS 

==============================================================================
B) inlined helper bodies: byte-identical to bench originals
==============================================================================
Krea2/calib_input_scale_nvfp4.py             compared= 2 identical= 2 differ=none  PASS
Z_Image/calib_input_scale_nvfp4.py           compared=11 identical=11 differ=none  PASS
Z_Image/diag_impact.py                       compared=11 identical=11 differ=none  PASS
Z_Image/diag_impact_old.py                   compared=11 identical=11 differ=none  PASS
Flux1/calib_input_scale_nvfp4.py             compared= 4 identical= 4 differ=none  PASS
Flux1/diag_impact.py                         compared= 4 identical= 4 differ=none  PASS
benchmark/krea2_int8_traj_compare.py         compared=10 identical=10 differ=none  PASS
benchmark/qwen_image_edit_traj_compare.py    compared=10 identical=10 differ=none  PASS
benchmark/flux1_nvfp4/flux_traj_compare.py   compared=10 identical=10 differ=none  PASS

==============================================================================
C) no external bench load / no undefined names
==============================================================================
Krea2/calib_input_scale_nvfp4.py             external_loads=0 undefined=none PASS
Z_Image/calib_input_scale_nvfp4.py           external_loads=0 undefined=none PASS
Z_Image/diag_impact.py                       external_loads=0 undefined=none PASS
Z_Image/diag_impact_old.py                   external_loads=0 undefined=none PASS
Flux1/calib_input_scale_nvfp4.py             external_loads=0 undefined=none PASS
Flux1/diag_impact.py                         external_loads=0 undefined=none PASS
benchmark/krea2_int8_traj_compare.py         external_loads=0 undefined=none PASS
benchmark/qwen_image_edit_traj_compare.py    external_loads=0 undefined=none PASS
benchmark/flux1_nvfp4/flux_traj_compare.py   external_loads=0 undefined=none PASS

OVERALL: ALL PROOFS PASS

```n