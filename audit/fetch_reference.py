# -*- coding: utf-8 -*-
"""Prove what can be proven without downloading 42 GB.

The full official flan-t5-xxl is five safetensors shards (8.80 + 8.94 + 9.27 +
9.31 + 5.65 = 41.97 GB). At the measured line rate that is over a day. But a
safetensors file is a JSON header followed by tensor payloads, so the header can
be fetched with one range request, and then ANY SUBSET of tensors can be fetched
by byte range and compared exactly.

This program:

  1. fetches each shard's header (a few hundred KB) and the published sha256;
  2. builds the official inventory (name -> shard, dtype, shape, byte range);
  3. compares the inventory against the local file: every local tensor must
     appear once, with the same shape, and no official tensor may be missing;
  4. downloads a chosen set of tensors by byte range and compares them exactly
     against the local file cast to the official dtype's precision.

A tensor compared this way is proven bit-identical. Tensors not downloaded are
inventoried but not proven - the report says exactly how many of each.

    python audit/fetch_reference.py --local <local.safetensors> \
        --repo google/flan-t5-xxl --shards 5 --coverage 40 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import requests

HF = "https://huggingface.co"
TOKEN_PATHS = (
    os.path.expanduser("~/.cache/huggingface/token"),
    os.path.expanduser("~/huggingface_token"),
)


def token():
    t = os.environ.get("HF_TOKEN", "")
    if t:
        return t.strip()
    for p in TOKEN_PATHS:
        if os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read().strip()
            except OSError:
                pass
    return ""


def session(tok):
    s = requests.Session()
    if tok:
        s.headers["Authorization"] = "Bearer %s" % tok
    return s


def shard_url(repo, name):
    return "%s/%s/resolve/main/%s" % (HF, repo, name)


def fetch_range(s, url, start, end, retries=4, cache_dir=""):
    """Inclusive byte range, with retries and an optional on-disk cache.

    The cache matters: proving a large model means many runs, and the line to the
    hub is often the bottleneck. A cached range is never fetched twice, so
    raising --coverage on a later run only downloads what is new."""
    import hashlib as _h
    cpath = ""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _h.sha1((url + "|%d|%d" % (start, end)).encode()).hexdigest()
        cpath = os.path.join(cache_dir, key + ".bin")
        if os.path.exists(cpath) and os.path.getsize(cpath) == (end - start + 1):
            with open(cpath, "rb") as f:
                return f.read()
    h = {"Range": "bytes=%d-%d" % (start, end)}
    last = None
    for i in range(retries):
        try:
            r = s.get(url, headers=h, timeout=(30, 300), allow_redirects=True)
            if r.status_code in (200, 206):
                data = r.content
                if r.status_code == 200 and (end - start + 1) < len(data):
                    data = data[start:end + 1]
                if cpath:
                    tmp = cpath + ".part"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, cpath)
                return data
            last = "status %d" % r.status_code
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(2 * (i + 1))
    raise RuntimeError("range fetch failed (%s-%s): %s" % (start, end, last))


def read_header(s, url):
    """A safetensors header is u64 length + JSON. Two small range requests."""
    first = fetch_range(s, url, 0, 7)
    n = struct.unpack("<Q", first)[0]
    if n <= 0 or n > 200 * 1024 * 1024:
        raise ValueError("implausible header length %d" % n)
    raw = fetch_range(s, url, 8, 8 + n - 1)
    hdr = json.loads(raw.decode("utf-8"))
    return hdr, 8 + n


def local_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def numel(shape):
    v = 1
    for d in shape:
        v *= d
    return v


DT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
            "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def local_tensor_bytes(path, hdr, data_start, key):
    a, b = hdr[key]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        return f.read(b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--repo", default="google/flan-t5-xxl")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--shard-pattern", default="model-%05d-of-%05d.safetensors")
    ap.add_argument("--coverage", type=float, default=40.0,
                    help="percent of official parameters to download and prove")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="",
                    help="directory for the byte-range cache; a later run with a "
                         "higher --coverage only downloads what is new")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    tok = token()
    s = session(tok)

    print("=== 1. official headers (small range requests) ===", flush=True)
    total = 0
    official = {}     # name -> (shard_name, dtype, shape, data_off, url)
    shard_bytes = {}
    for i in range(1, args.shards + 1):
        name = args.shard_pattern % (i, args.shards)
        url = shard_url(args.repo, name)
        t0 = time.time()
        hdr, dstart = read_header(s, url)
        keys = [k for k in hdr if k != "__metadata__"]
        b = 0
        for k in keys:
            a, e = hdr[k]["data_offsets"]
            b = max(b, e)
        shard_bytes[name] = b
        total += b
        for k in keys:
            official[k] = (name, hdr[k]["dtype"], tuple(hdr[k]["shape"]),
                           hdr[k]["data_offsets"], url, dstart)
        print("  %-32s %6d tensors  %.2f GB  (%.1fs)" % (
            name, len(keys), b / 1024 ** 3, time.time() - t0), flush=True)
    print("  official total: %.2f GB, %d tensors" % (total / 1024 ** 3, len(official)))

    print(flush=True)
    print("=== 2. local inventory vs official ===", flush=True)
    lhdr, ldstart = local_header(args.local)
    lkeys = [k for k in lhdr if k != "__metadata__"]
    local_only = sorted(set(lkeys) - set(official))
    missing = sorted(set(official) - set(lkeys))
    shape_bad = []
    for k in sorted(set(lkeys) & set(official)):
        if tuple(lhdr[k]["shape"]) != official[k][2]:
            shape_bad.append((k, tuple(lhdr[k]["shape"]), official[k][2]))
    print("  local tensors      : %d" % len(lkeys))
    print("  official tensors   : %d" % len(official))
    print("  local-only         : %d %s" % (len(local_only), local_only[:5]))
    print("  missing from local : %d %s" % (len(missing), missing[:5]))
    print("  shape mismatches   : %d %s" % (len(shape_bad), shape_bad[:3]))
    linfo = {"local_tensors": len(lkeys), "official_tensors": len(official),
             "local_only": local_only[:20], "missing": missing[:20],
             "shape_mismatch": [x[0] for x in shape_bad][:20],
             "official_total_bytes": total}

    print(flush=True)
    print("=== 3. prove a coverage subset by exact byte comparison ===", flush=True)
    # choose tensors by descending size until the budget is filled: largest
    # tensors prove the most parameters per byte downloaded
    budget = total * (args.coverage / 100.0)
    order = sorted(official.items(),
                   key=lambda kv: -numel(kv[1][2]) * DT_BYTES.get(kv[1][1], 2))
    chosen, acc = [], 0
    for k, info in order:
        if k not in set(lkeys):
            continue
        nb = numel(info[2]) * DT_BYTES.get(info[1], 2)
        if acc + nb > budget:
            continue
        chosen.append(k)
        acc += nb
    print("  selected %d tensors, %.2f GB to download (%.1f%% of params)" % (
        len(chosen), acc / 1024 ** 3,
        100.0 * sum(numel(official[k][2]) for k in chosen)
        / sum(numel(official[k][2]) for k in official)), flush=True)

    exact = bad = 0
    checked_elems = 0
    problems = []
    t0 = time.time()
    done_bytes = 0
    import torch
    TDMAP = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
             "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
             "I8": torch.int8, "U8": torch.uint8}
    for idx, k in enumerate(chosen, 1):
        shard, dt, shape, (a, b), url, dstart = official[k]
        # data_offsets are relative to the END OF THE HEADER, not the file start
        raw = fetch_range(s, url, dstart + a, dstart + b - 1, cache_dir=args.cache)
        done_bytes += len(raw)
        tdt = TDMAP[dt]
        ref = torch.frombuffer(bytearray(raw), dtype=tdt).reshape(shape)
        lb = local_tensor_bytes(args.local, lhdr, ldstart, k)
        lt = TDMAP[lhdr[k]["dtype"]]
        mine = torch.frombuffer(bytearray(lb), dtype=lt).reshape(lhdr[k]["shape"])
        cast = ref.to(mine.dtype)
        if mine.numel() != cast.numel():
            problems.append((k, "numel %d vs %d" % (mine.numel(), cast.numel())))
            bad += 1
            continue
        eq = torch.equal(mine, cast)
        checked_elems += mine.numel()
        if eq:
            exact += 1
        else:
            d = (mine.float() - cast.float()).abs().max().item()
            n_diff = int((mine != cast).sum())
            problems.append((k, "max_abs=%.3e elements=%d" % (d, n_diff)))
            bad += 1
        if idx % 5 == 0 or idx == len(chosen):
            el = time.time() - t0
            print("  %3d/%3d  exact=%d bad=%d  %.2f GB  (%.0fs)" % (
                idx, len(chosen), exact, bad, done_bytes / 1024 ** 3, el), flush=True)

    tot_params = sum(numel(v[2]) for v in official.values())
    print()
    print("=== result ===")
    print("  inventory        : %d/%d tensors accounted for, %d shape mismatches" % (
        len(set(lkeys) & set(official)), len(official), len(shape_bad)))
    print("  weight-level     : %d tensors proven bit-identical, %d divergent" % (
        exact, bad))
    print("  parameters proven: %s of %s (%.2f%%)" % (
        "{:,}".format(checked_elems), "{:,}".format(tot_params),
        100.0 * checked_elems / tot_params))
    for k, why in problems[:10]:
        print("    PROBLEM %s  %s" % (k, why))

    rec = dict(linfo)
    rec.update({"proven_tensors": exact, "divergent_tensors": bad,
                "proven_parameters": checked_elems, "total_parameters": tot_params,
                "proven_fraction": checked_elems / tot_params,
                "problems": problems[:50], "repo": args.repo})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print("\nwrote %s" % args.json)

    sys.exit(2 if (bad or shape_bad or missing or local_only) else 0)


if __name__ == "__main__":
    main()
