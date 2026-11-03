# -*- coding: utf-8 -*-
"""Conv2d (4-D) ConvRot INT8 tamper test — 検出器が 4-D 経路を見ていることを表で証明する。

変換は converter と同じ手順で行う: permute (O, kH, kW, I) -> in_channels 方向に rotate ->
per-out-channel amax/127 -> round。監査は backdoor_audit.provenance() が pack を source と突き合わせる。

出力: md/backdoor/verdicts/records/conv2d_tamper_test.json
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import torch
import backdoor_audit as ba

GS = 16


def quantize_conv2d_pack(w4, convrot=True, gs=GS):
    w = w4.float()
    o_c, i_c, k_h, k_w = w.shape
    if convrot:
        h = ba.hadamard(gs)
        flat = w.permute(0, 2, 3, 1).contiguous().view(-1, i_c)
        flat_rot = torch.matmul(flat.view(-1, i_c // gs, gs), h.T).reshape(flat.shape)
        w = flat_rot.view(o_c, k_h, k_w, i_c).permute(0, 3, 1, 2).contiguous()
    s = w.abs().amax(dim=tuple(range(1, w.dim()))).reshape(-1) / 127.0
    q = (w / s.view(-1, 1, 1, 1)).round().clamp(-127, 127).to(torch.int8)
    cfg = {"format": "int8_tensorwise"}
    if convrot:
        cfg.update({"convrot": True, "convrot_groupsize": int(gs)})
    return ({"m.conv0.weight": q,
             "m.conv0.weight_scale": s.float().contiguous(),
             "m.conv0.comfy_quant": torch.tensor(list(json.dumps(cfg).encode()), dtype=torch.uint8)},
            {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": {"m.conv0": dict(cfg)}})})


DT = {'float16': 'F16', 'float32': 'F32', 'bfloat16': 'BF16', 'int8': 'I8',
      'uint8': 'U8', 'int16': 'I16', 'int32': 'I32', 'int64': 'I64',
      'float64': 'F64', 'bool': 'BOOL'}


def save(path, tensors, meta=None):
    hdr = {}
    data = b''
    for k, v in tensors.items():
        v = v.contiguous()
        b = v.numpy().tobytes()
        hdr[k] = {"dtype": DT.get(str(v.dtype).split('.')[-1], str(v.dtype).split('.')[-1].upper()),
                  "shape": list(v.shape),
                  "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    if meta:
        hdr["__metadata__"] = {kk: str(vv) for kk, vv in meta.items()}
    h = json.dumps(hdr).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(len(h).to_bytes(8, 'little'))
        f.write(h)
        f.write(data)


def tamper_3lsb(p):
    w = p["m.conv0.weight"].to(torch.int16).clone()
    v = w.view(-1)
    v[100] = v[100] + 3
    p["m.conv0.weight"] = v.clamp(-127, 127).to(torch.int8).reshape(w.shape)


def tamper_scale_6pct(p):
    s = p["m.conv0.weight_scale"].clone()
    s[3] = s[3] * 1.06
    p["m.conv0.weight_scale"] = s


def tamper_one_code(p):
    w = p["m.conv0.weight"].to(torch.int16).clone()
    v = w.view(-1)
    v[77] = v[77] + 1
    p["m.conv0.weight"] = v.clamp(-127, 127).to(torch.int8).reshape(w.shape)


CASES = [
    ("untouched Conv2d pack", None, "FAITHFUL"),
    ("3-LSB edit of one quantized Conv2d weight", tamper_3lsb, "EDITED"),
    ("6 % change of one Conv2d weight_scale", tamper_scale_6pct, "EDITED"),
    ("single int8 code +1 LSB in a Conv2d weight", tamper_one_code, "EDITED"),
]


def main():
    d = os.path.join(ROOT, 'audit', '_conv2d_tamper_tmp')
    os.makedirs(d, exist_ok=True)
    src_w = (torch.randn(8, 64, 3, 3, generator=torch.Generator().manual_seed(7)) * 0.02).to(torch.float16)
    results = []
    for label, tamper, expect in CASES:
        pc, mc = quantize_conv2d_pack(src_w.float(), gs=GS)
        if tamper:
            tamper(pc)
        sp = os.path.join(d, "c2src.safetensors")
        pp = os.path.join(d, "c2pack.safetensors")
        save(sp, {"m.conv0.weight": src_w})
        save(pp, pc, mc)
        r = ba.provenance(sp, pp)
        results.append({
            "case": label,
            "expected_verdict": expect,
            "verdict": r.get("verdict"),
            "q_mismatch_total": r.get("q_mismatch_total"),
            "max_q_delta": r.get("max_q_delta"),
            "scale_exact_rel_max": r.get("scale_exact_rel_max"),
            "rel_l2_max": r.get("rel_l2_max"),
            "q_mismatch_ratio_max": r.get("q_mismatch_ratio_max"),
            "quantized_layers": r.get("quantized_layers"),
            "float_mismatch": r.get("float_mismatch"),
            "extra": r.get("extra"),
            "conv2d_shape": list(src_w.shape),
            "convrot_groupsize": GS,
            "matches_expectation": r.get("verdict") == expect,
        })
        print('%-46s expect=%-9s got=%-9s q_mismatch=%s max_q_delta=%s scale_rel=%s' % (
            label, expect, r.get('verdict'), r.get('q_mismatch_total'),
            r.get('max_q_delta'), r.get('scale_exact_rel_max')))

    out = {
        "what": "Conv2d (4-D) ConvRot INT8 tamper test for audit/backdoor_audit.py",
        "how": "the pack is built with the converter's own steps (permute (O,kH,kW,I), rotate along "
               "in_channels, per-out-channel amax/127, round); the audit re-derives the codes from the "
               "float source and compares them",
        "source_tensor": "m.conv0.weight",
        "source_shape": [8, 64, 3, 3],
        "cases": results,
        "all_match_expectation": all(c["matches_expectation"] for c in results),
    }
    path = os.path.join(ROOT, 'md', 'backdoor', 'verdicts', 'records', 'conv2d_tamper_test.json')
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print('written:', path, '| all match:', out['all_match_expectation'])

    for fn in ('c2src.safetensors', 'c2pack.safetensors'):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)
    try:
        os.rmdir(d)
    except OSError:
        pass


if __name__ == '__main__':
    main()
