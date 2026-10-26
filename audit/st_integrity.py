# -*- coding: utf-8 -*-
"""Structural-integrity checks for safetensors files, shared by the audit tools.

A safetensors file is a u64 header length, a JSON header, and one flat data
region that the header's ``data_offsets`` carve into tensors. A weight edit is
not the only way to carry something extra in such a file; the container itself
has room to hide it:

- bytes appended after the last tensor, or left in a gap between tensors
  (an unreferenced region no loader ever reads, but a second program can);
- overlapping tensor ranges, or ranges whose size disagrees with dtype x shape;
- duplicate JSON keys in the header (parsers disagree on which one wins, so two
  readers can see two different models in one file);
- free-form ``__metadata__`` strings (URLs, shell fragments, encoded blobs);
- tensor names that are not weight names (paths, scripts, control characters).

``check`` reports each of these. It reads only the header and file size - no
tensor data - so it is cheap on multi-gigabyte files.

Nothing here proves a file is clean. It proves the file is *exactly* the
declared header plus the declared tensors, with nothing else in it.
"""
from __future__ import annotations

import json
import os
import struct

MAX_HEADER = 200 * 1024 * 1024

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

SUSPICIOUS_WORDS = ("backdoor", "trigger", "payload", "exploit", "shell",
                    "reverse", "cmd", "exec", "hook")
SCRIPT_EXT = (".py", ".sh", ".exe", ".dll", ".bat", ".ps1", ".js", ".so",
              ".pkl", ".pt", ".bin", ".ckpt")
META_MARKERS = ("http://", "https://", "curl ", "wget ", "| sh", "| bash",
                "base64", "import os", "import sys", "subprocess", "exec(",
                "eval(", "powershell", "cmd.exe", "/etc/passwd", "rm -rf")
META_MAX_VALUE = 64 * 1024
# the one metadata value that is legitimately large and JSON; it is validated
# structurally by the pack audit instead of being scanned for markers
QUANT_META_KEY = "_quantization_metadata"


def read_header(path):
    """Return (header, header_len, duplicate_keys, file_size) or raise ValueError."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError("file shorter than 8 bytes")
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > MAX_HEADER or 8 + n > size:
            raise ValueError("implausible header length %d" % n)
        raw = f.read(n)
    dups = []

    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                dups.append(k)
            d[k] = v
        return d

    try:
        hdr = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except Exception as e:
        raise ValueError("header is not JSON: %s" % e)
    if not isinstance(hdr, dict):
        raise ValueError("header is not an object")
    return hdr, n, dups, size


def key_findings(keys):
    """Findings about tensor *names*: suspicious words, odd characters, paths."""
    keys = list(keys)
    out = []
    sus = [k for k in keys if any(w in k.lower() for w in SUSPICIOUS_WORDS)]
    if sus:
        out.append("suspicious key names: %s" % sus[:8])
    odd = [k for k in keys
           if (not k.isascii()) or len(k) > 512
           or any(ord(c) < 32 or ord(c) == 127 for c in k)]
    if odd:
        out.append("non-ASCII / control / oversized key names: %s"
                   % [repr(k)[:80] for k in odd[:6]])
    pathy = [k for k in keys
             if "/" in k or "\\" in k or k.lower().endswith(SCRIPT_EXT)]
    if pathy:
        out.append("path- or script-like key names: %s" % pathy[:8])
    return out


def metadata_findings(meta):
    out = []
    if not isinstance(meta, dict):
        return ["__metadata__ is not an object"]
    for k, v in meta.items():
        if not isinstance(v, str):
            out.append("metadata %r: value is not a string" % k)
            continue
        if k == QUANT_META_KEY:
            # this key was exempt from the marker scan, so a command string inside the
            # quantisation metadata was invisible. Parse the blob and scan its strings.
            import json as _json
            try:
                blob = _json.loads(v)
            except Exception:
                out.append("metadata %r is not valid JSON" % k)
                continue
            def _walk(x):
                if isinstance(x, str):
                    yield x
                elif isinstance(x, dict):
                    for vv in x.values():
                        yield from _walk(vv)
                elif isinstance(x, list):
                    for vv in x:
                        yield from _walk(vv)
            hits = []
            for sval in _walk(blob):
                low = sval.lower()
                hits += [m for m in META_MARKERS if m in low]
            if hits:
                out.append("metadata %r contains %s" % (k, sorted(set(hits))))
            continue
        if len(v) > META_MAX_VALUE:
            out.append("metadata %r: %d characters (oversized)" % (k, len(v)))
        low = v.lower()
        hit = [m for m in META_MARKERS if m in low]
        if hit:
            out.append("metadata %r contains %s" % (k, hit))
        if any(ord(c) < 32 and c not in "\n\r\t" for c in v):
            out.append("metadata %r contains control characters" % k)
    return out


def check(path):
    """Return (header or None, findings). ``None`` means not a safetensors stream."""
    try:
        hdr, n, dups, size = read_header(path)
    except ValueError as e:
        return None, ["NOT a safetensors stream: %s" % e]
    findings = []
    if dups:
        findings.append("duplicate header keys (two readers can disagree on the "
                        "model): %s" % sorted(set(dups))[:8])
    findings += metadata_findings(hdr.get("__metadata__", {}))

    data_len = size - 8 - n
    spans = []
    names = []
    bad_entries = []
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        names.append(k)
        if not isinstance(v, dict) or not {"dtype", "shape", "data_offsets"} <= set(v):
            bad_entries.append("%s: malformed entry" % k)
            continue
        unknown = set(v) - {"dtype", "shape", "data_offsets"}
        if unknown:
            bad_entries.append("%s: unknown fields %s" % (k, sorted(unknown)))
        dt, shape, off = v["dtype"], v["shape"], v["data_offsets"]
        if not (isinstance(shape, list)
                and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                        for d in shape)):
            bad_entries.append("%s: invalid shape %r" % (k, shape))
            continue
        if not (isinstance(off, list) and len(off) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in off)
                and 0 <= off[0] <= off[1]):
            bad_entries.append("%s: invalid data_offsets %r" % (k, off))
            continue
        if dt in DTYPE_SIZE:
            expected = DTYPE_SIZE[dt]
            for d in shape:
                expected *= d
            if off[1] - off[0] != expected:
                bad_entries.append("%s: %d bytes for %s%s (expected %d)"
                                   % (k, off[1] - off[0], dt, shape, expected))
        else:
            bad_entries.append("%s: unknown dtype %r" % (k, dt))
        spans.append((off[0], off[1], k))
    if bad_entries:
        findings.append("%d malformed tensor entr%s: %s" % (
            len(bad_entries), "y" if len(bad_entries) == 1 else "ies", bad_entries[:6]))

    spans.sort()
    cursor = 0
    gaps, overlaps = [], []
    for b, e, k in spans:
        if b < cursor:
            overlaps.append(k)
        elif b > cursor:
            gaps.append((cursor, b - cursor, k))
        cursor = max(cursor, e)
    if overlaps:
        findings.append("overlapping tensor ranges: %s" % overlaps[:8])
    if gaps:
        findings.append("%d unreferenced gap(s) between tensors, %d bytes in total "
                        "(first before %r)" % (len(gaps), sum(g[1] for g in gaps), gaps[0][2]))
    if cursor < data_len:
        findings.append("%d unreferenced byte(s) after the last tensor "
                        "(possible smuggled payload)" % (data_len - cursor))
    elif cursor > data_len:
        findings.append("tensor ranges run %d byte(s) past the end of the file"
                        % (cursor - data_len))
    if not names:
        findings.append("no tensors")
    findings += key_findings(names)
    return hdr, findings
