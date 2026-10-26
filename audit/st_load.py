# -*- coding: utf-8 -*-
"""Load a candidate safetensors file into an already-built reference model, on the record.

The audit tools used to call `load_state_dict(sd, strict=False)` with the raw candidate
keys. On a file that stores a parameter under another name - or transposed - the candidate
tensor is dropped as "unexpected" and the reference's own tensor stays in the model, so
every downstream number is computed from the reference while claiming to be the candidate.
That failure is silent, which is why it is unacceptable in an audit.

This helper makes the mapping explicit and loud:

  * `remap` renames a candidate key onto a model key (`{"text_projection":
    "text_projection.weight"}`);
  * `transpose` marks a renamed key that the file stores transposed (open_clip stores
    `x @ W` where nn.Linear applies `x @ W.T`);
  * a candidate key that resolves to no model parameter is an error unless the caller
    passes `allow_unexpected=True`, in which case the keys are returned for recording;
  * missing model keys are returned, not swallowed: an absent head is a fact about the
    artifact and belongs in the record.

    sd, info = build_state_dict(path, model, remap={"a": "b"}, transpose={"a"})
    missing, unexpected = apply_state_dict(model, sd)
"""
from __future__ import annotations

from safetensors import safe_open


def parse_pairs(items):
    """['src=dst', ...] -> {'src': 'dst'}"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit("expected src=dst, got %r" % it)
        a, b = it.split("=", 1)
        out[a] = b
    return out


def build_state_dict(path, model, remap=None, transpose=None):
    remap = remap or {}
    transpose = transpose or set()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    sd, renamed, unmapped = {}, [], []
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tgt = remap.get(k, k)
            if tgt in params or tgt in buffers:
                t = f.get_tensor(k).float()
                if k in transpose or tgt in transpose:
                    if t.dim() != 2:
                        raise SystemExit("transpose requested for a %d-D tensor: %s"
                                         % (t.dim(), k))
                    t = t.T
                sd[tgt] = t
                if tgt != k:
                    renamed.append([k, tgt])
            else:
                unmapped.append(k)
    info = {"renamed": renamed, "unmapped": unmapped}
    if unmapped:
        raise SystemExit(
            "candidate tensors with no model key: %s\n"
            "these would be dropped while the reference's parameter stays in the model. "
            "Pass remap={'src': 'dst'} (and transpose={'src'} if the file stores it "
            "transposed), or allow_unexpected=True only if the drop is understood."
            % unmapped[:8])
    return sd, info


def apply_state_dict(model, sd):
    res = model.load_state_dict(sd, strict=False)
    return list(res.missing_keys), list(res.unexpected_keys)
