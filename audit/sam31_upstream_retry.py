# -*- coding: utf-8 -*-
"""S6. Upstream comparison retry for SAM3.1 (Appendix A.8).

What this does, in order:
 1. Re-attempts the official repository (facebook/sam3.1) with the supplied token
    and records every response verbatim, with UTC timestamps.
 2. Enumerates public mirrors via the Hub search API, records each candidate's
    gated flag, declared base_model, weight files, sizes and LFS oids, and
    identifies which are fp32-scale (matching the official .pt byte size) versus
    fp16 repacks (the audited file's byte size).
 3. If the official weights stay blocked, downloads the selected mirror's fp32
    checkpoint, verifies its size against the official entry, hashes it, loads it,
    and compares ALL tensors of the audited fp16 file against fp16(fp32 upstream):
    element counts, bit equality, and the maximum absolute difference.
 4. Writes the dated result into the record (obtained, or refusal with the
    verbatim body) so the gate can check it.

The record must not contain local absolute paths; only repository-relative
record paths and public URLs are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

HF = "https://huggingface.co"
UA = "hswq-audit-s6/1.0"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_token(path: str) -> str:
    if path:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    return os.environ.get("HF_TOKEN", "").strip()


def http(url: str, token: str, method: str = "GET", byte_range: str = "",
         limit: int = 0, stream_to: str = ""):
    """Return (status, headers, body_or_bytes_written)."""
    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = "Bearer " + token
    if byte_range:
        headers["Range"] = byte_range
    req = urllib.request.Request(url, headers=headers, method=method)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=180, context=ctx) as resp:
            if stream_to:
                h = hashlib.sha256()
                n = 0
                t0 = time.time()
                with open(stream_to, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
                        if n % (256 << 20) < (1 << 22):
                            print("    ... %d MiB (%.1f MiB/s)" % (
                                n >> 20, (n / (1 << 20)) / max(1e-6, time.time() - t0)), flush=True)
                return resp.status, dict(resp.headers), {"bytes": n, "sha256": h.hexdigest()}
            body = resp.read() if limit == 0 else resp.read(limit)
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read(2000)
        return e.code, dict(e.headers), body
    except Exception as e:  # network failure: record, do not hide
        return None, {}, ("%s: %s" % (type(e).__name__, e)).encode()


def jget(url: str, token: str):
    s, h, b = http(url, token)
    if s == 200:
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return None
    return None


def flatten_tensors(obj, prefix=""):
    import torch
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_tensors(v, "%s%s." % (prefix, k) if prefix else "%s." % k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_tensors(v, "%s%d." % (prefix, i)))
    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        out[prefix[:-1] if prefix.endswith(".") else prefix] = obj
    return out


def load_upstream(path: str):
    """Load a checkpoint and return (tensor_dict, loader_note)."""
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        note = "torch.load(weights_only=True)"
    except Exception as e:
        return None, "torch.load failed: %s" % e
    tensors = flatten_tensors(obj)
    return tensors, note


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM3.1 S6 upstream comparison retry")
    ap.add_argument("--local", required=True, help="audited fp16 safetensors")
    ap.add_argument("--work-dir", required=True, help="download directory (explicit)")
    ap.add_argument("--token-file", default="", help="optional file holding an HF token")
    ap.add_argument("--mirror-repo", default="AEmotionStudio/sam3.1")
    ap.add_argument("--mirror-file", default="sam3.1_multiplex.pt")
    ap.add_argument("--out", default="md/backdoor/verdicts/records/sam_gated.json")
    args = ap.parse_args()

    token = read_token(args.token_file)
    print("token present:", bool(token), flush=True)
    t_start = time.time()

    record = {}
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            record = json.load(f)
    record.setdefault("s6", {})

    # --- 1. official attempts ---
    print("[1] official repository attempts", flush=True)
    official = []
    s, h, b = http(HF + "/api/models/facebook/sam3.1", token)
    meta = None
    if s == 200:
        meta = json.loads(b.decode("utf-8"))
    official.append({
        "ts_utc": now_utc(), "request": "GET /api/models/facebook/sam3.1",
        "status": s,
        "body": (b.decode("utf-8", "replace")[:400] if s != 200 else
                 json.dumps({"gated": meta.get("gated"), "private": meta.get("private"),
                             "disabled": meta.get("disabled"), "downloads": meta.get("downloads")})),
    })
    s, h, b = http(HF + "/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt", token,
                   byte_range="bytes=0-0")
    official.append({"ts_utc": now_utc(),
                     "request": "GET /facebook/sam3.1/resolve/main/sam3.1_multiplex.pt (Range 0-0)",
                     "status": s, "body": b.decode("utf-8", "replace")[:400]})
    s, h, b = http(HF + "/facebook/sam3.1/raw/main/README.md", token, limit=400)
    official.append({"ts_utc": now_utc(), "request": "GET /facebook/sam3.1/raw/main/README.md",
                     "status": s, "body": b.decode("utf-8", "replace")[:400]})
    for a in official:
        print("   %-70s %s" % (a["request"][:70], a["status"]), flush=True)

    # official file sizes (metadata is readable even while the gate blocks content)
    official_size = None
    tree = jget(HF + "/api/models/facebook/sam3.1/tree/main?recursive=true", token)
    if tree:
        for x in tree:
            if x.get("path", "").endswith("sam3.1_multiplex.pt"):
                official_size = (x.get("lfs") or {}).get("size")
    print("   official .pt size (metadata):", official_size, flush=True)

    # --- 2. mirror enumeration ---
    print("[2] mirror enumeration", flush=True)
    candidates = {}
    for q in ("sam3.1", "sam3", "SAM3.1"):
        arr = jget(HF + "/api/models?search=%s&limit=100" % urllib.parse.quote(q), token)
        for m in arr or []:
            candidates[m["id"]] = m
    mirror_rows = []
    for rid in sorted(candidates):
        j = jget(HF + "/api/models/%s?blobs=true" % rid, token)
        if not j:
            continue
        files = []
        for x in j.get("siblings", []):
            n = x.get("rfilename", "")
            if n.endswith((".pt", ".safetensors", ".pth", ".bin")):
                lfs = x.get("lfs") or {}
                files.append({"path": n, "size": lfs.get("size") or x.get("size"),
                              "oid": (lfs.get("oid") or "")[:64]})
        if not files:
            continue
        cd = j.get("cardData", {}) or {}
        mirror_rows.append({
            "repo": rid, "gated": j.get("gated"), "private": j.get("private"),
            "base_model": cd.get("base_model"), "downloads": j.get("downloads"),
            "weights": files,
        })
    fp32_like = [m for m in mirror_rows
                 if any(f["size"] == official_size for f in m["weights"])]
    fp16_like = [m for m in mirror_rows
                 if any(f["size"] == 1745546848 for f in m["weights"])]
    print("   mirrors with a weight file of the official .pt size: %d" % len(fp32_like), flush=True)
    print("   mirrors hosting the audited fp16 file size: %d" % len(fp16_like), flush=True)

    selected = None
    for m in fp32_like:
        if m["repo"] == args.mirror_repo:
            selected = m
            break
    if selected is None and fp32_like:
        selected = fp32_like[0]

    record["s6"].update({
        "date_utc": now_utc(),
        "official_attempts": official,
        "official_weight_request_status": official[1]["status"],
        "official_pt_size_metadata": official_size,
        "mirror_enumeration": {
            "candidates_with_weights": len(mirror_rows),
            "fp32_scale_candidates": [{"repo": m["repo"], "base_model": m["base_model"],
                                       "files": m["weights"]} for m in fp32_like][:20],
            "fp16_repack_candidates": [{"repo": m["repo"], "base_model": m["base_model"],
                                        "files": m["weights"]} for m in fp16_like][:20],
        },
    })

    # --- 3. download selected mirror and compare ---
    gate = False
    comparison = None
    download = None
    if selected is None:
        print("[3] no fp32-scale mirror found; recording refusal only", flush=True)
        record["s6"]["result"] = "no_upstream_obtained"
    else:
        fmeta = next(f for f in selected["weights"] if f["size"] == official_size)
        url = "%s/%s/resolve/main/%s" % (HF, selected["repo"], urllib.parse.quote(fmeta["path"]))
        dest = os.path.join(args.work_dir, os.path.basename(fmeta["path"]))
        os.makedirs(args.work_dir, exist_ok=True)
        declared = int(fmeta["size"])
        # A.10: declare the download size before starting
        record["s6"]["declared_download"] = {
            "ts_utc": now_utc(), "repo": selected["repo"], "file": fmeta["path"],
            "declared_bytes": declared,
            "note": "size equals the official facebook/sam3.1 .pt entry",
        }
        print("[3] declared download: %d bytes (%.2f GiB) from %s"
              % (declared, declared / (1024 ** 3), selected["repo"]), flush=True)
        if os.path.exists(dest) and os.path.getsize(dest) == declared:
            print("   reusing existing file", flush=True)
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    h.update(c)
            download = {"bytes": declared, "sha256": h.hexdigest()}
        else:
            s, hd, res = http(url, token, stream_to=dest)
            print("   download status:", s, flush=True)
            if s != 200:
                record["s6"]["result"] = "mirror_download_failed"
                record["s6"]["download_error"] = {"status": s, "body": res[:400].decode("utf-8", "replace")}
            else:
                download = res
        if download:
            record["s6"]["mirror_download"] = {
                "repo": selected["repo"], "file": fmeta["path"],
                "bytes": download["bytes"], "sha256": download["sha256"],
                "size_matches_official": download["bytes"] == declared,
                "oid_from_hub": fmeta.get("oid", ""),
            }
            print("[4] loading upstream %s" % fmeta["path"], flush=True)
            up, note = load_upstream(dest)
            record["s6"]["upstream_loader"] = note
            if up is None:
                record["s6"]["result"] = "upstream_load_failed"
                record["s6"]["load_error"] = note
            else:
                import torch
                import safetensors.torch as st
                local = st.load_file(args.local)
                # choose the prefix with maximum key overlap
                best = None
                for pref in ("", "model.", "state_dict.", "module.", "model.model."):
                    keys = [k[len(pref):] for k in up if k.startswith(pref)]
                    overlap = len(set(keys) & set(local))
                    if best is None or overlap > best[1]:
                        best = (pref, overlap, keys)
                pref, overlap, keys = best
                up_map = {}
                for k in up:
                    if k.startswith(pref):
                        up_map[k[len(pref):]] = up[k]
                print("   upstream tensors: %d | prefix %r | overlap with local: %d"
                      % (len(up), pref, overlap), flush=True)

                missing = sorted(set(local) - set(up_map))
                extra = sorted(set(up_map) - set(local))
                total_elems = 0
                diff_elems = 0
                bit_equal_tensors = 0
                max_abs = 0.0
                diff_tensors = []
                up_dtypes = {}
                for k, lt in local.items():
                    if k not in up_map:
                        continue
                    ut = up_map[k]
                    up_dtypes[str(ut.dtype)] = up_dtypes.get(str(ut.dtype), 0) + 1
                    if tuple(ut.shape) != tuple(lt.shape):
                        diff_tensors.append({"key": k, "issue": "shape", "local": list(lt.shape),
                                             "upstream": list(ut.shape)})
                        continue
                    u16 = ut.to(torch.float32).half()
                    total_elems += int(lt.numel())
                    if torch.equal(u16, lt):
                        bit_equal_tensors += 1
                        continue
                    d = (u16.to(torch.float32) - lt.to(torch.float32)).abs()
                    n_diff = int((d > 0).sum().item())
                    diff_elems += n_diff
                    m = float(d.max().item())
                    max_abs = max(max_abs, m)
                    if len(diff_tensors) < 40:
                        diff_tensors.append({"key": k, "diff_elements": n_diff,
                                             "max_abs_diff": m, "numel": int(lt.numel())})
                comparison = {
                    "local_tensors": len(local),
                    "upstream_tensors": len(up_map),
                    "keys_matched": len(set(local) & set(up_map)),
                    "missing_in_upstream": missing[:40],
                    "missing_in_upstream_count": len(missing),
                    "extra_in_upstream_count": len(extra),
                    "prefix_used": pref,
                    "upstream_dtypes": up_dtypes,
                    "total_elements_compared": total_elems,
                    "bit_identical_tensors": bit_equal_tensors,
                    "tensors_with_differences": len(local) - bit_equal_tensors - len(missing),
                    "differing_elements": diff_elems,
                    "max_abs_diff_fp16": max_abs,
                    "diff_sample": diff_tensors,
                }
                full_match = (len(missing) == 0 and bit_equal_tensors == len(local))
                comparison["verdict"] = (
                    "all %d tensors bit-identical to fp16(fp32 upstream)" % len(local)
                    if full_match else
                    "differences measured: %d tensors differ, %d elements, max |diff| %.6g"
                    % (comparison["tensors_with_differences"], diff_elems, max_abs))
                record["s6"]["comparison"] = comparison
                record["s6"]["result"] = "obtained_and_compared"
                gate = True
                print("[5] comparison: %s" % comparison["verdict"], flush=True)

    # gate: a dated result exists in the record (obtained or refused)
    if not gate:
        gate = bool(record["s6"].get("official_attempts"))
    record["s6"]["gate_s6_pass"] = gate
    record["s6"]["elapsed_seconds"] = round(time.time() - t_start, 1)
    record["s6"]["environment"] = {
        "device": "cpu (HTTPS metadata/transfer + fp32 checkpoint comparison in RAM)",
        "gpu_peak_memory_mib": 0,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print("=== S6 (%.1f s) ===" % record["s6"]["elapsed_seconds"], flush=True)
    print("official status:", record["s6"]["official_weight_request_status"], flush=True)
    print("result:", record["s6"].get("result"), flush=True)
    print("Gate S6:", "PASS" if gate else "FAIL", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
