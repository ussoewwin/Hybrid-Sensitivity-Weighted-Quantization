"""Does the pack actually run on A1111 and Forge?

The ComfyUI verdict said "no route on this machine". That was wrong: it was written
after looking at ComfyUI only. This measures the two routes that were not looked at.

A1111 runs `IPAdapterModel.load` -> `modules/ipadapter_int8_convrot.prepare_state_dict`
-> `install_runtime_int8`, and the INT8 payload is installed in place. The question
here is whether that path accepts *this* pack and whether the kernel fires.

Forge carries `backend.operations_mixed_precision` which pops `comfy_quant` from the
state dict, and `extensions-builtin/sd_forge_ipadapter`. That route is measured the
same way: build the parser that reads the marker, and see whether it reads this park's
stamps.

The report of this run goes into the CN-2 record. It is a route census, not an
acceptance: it says whether the code path takes the file, not whether a render is
correct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def a1111_route(pack_path, device):
    out = {"route": "a1111", "ok": False, "steps": []}

    def step(name, **kw):
        rec = {"step": name}
        rec.update(kw)
        out["steps"].append(rec)
        print("  [a1111] %-28s %s" % (name, kw), flush=True)

    try:
        from modules import ipadapter_int8_convrot as ip8
        step("import modules.ipadapter_int8_convrot", ok=True, file=ip8.__file__)
    except Exception as e:
        step("import modules.ipadapter_int8_convrot", ok=False, error="%s: %s" % (type(e).__name__, e))
        return out

    import torch
    try:
        import comfy_kitchen.backends.cuda as kc
        step("comfy_kitchen.backends.cuda", ok=True, has_int8_linear=hasattr(kc, "int8_linear"))
    except Exception as e:
        step("comfy_kitchen.backends.cuda", ok=False, error="%s: %s" % (type(e).__name__, e))

    try:
        sd_raw = torch.load(pack_path, map_location="cpu", weights_only=True)
        step("torch.load", ok=True, keys=sorted(sd_raw.keys()))
    except Exception as e:
        step("torch.load", ok=False, error="%s: %s" % (type(e).__name__, e))
        return out

    # The pack stores the two sub graphs flat ("image_proj.proj.0.weight"); the A1111
    # entry point indexes state_dict["image_proj"] / ["ip_adapter"], so rebuild them.
    nested = {"image_proj": {}, "ip_adapter": {}}
    for key, value in sd_raw.items():
        for head in ("image_proj.", "ip_adapter."):
            if key.startswith(head):
                nested[head[:-1]][key[len(head):]] = value
                break
    if not nested["image_proj"] and "image_proj" in sd_raw:
        nested = {"image_proj": dict(sd_raw["image_proj"]),
                  "ip_adapter": dict(sd_raw["ip_adapter"])}
    image_proj = nested["image_proj"]
    ip_adapter = nested["ip_adapter"]
    step("sub graphs", image_proj=len(image_proj), ip_adapter=len(ip_adapter),
         nested_or_flat="nested" if "image_proj" in sd_raw else "flat rebuilt")

    has_marker_proj = ip8.state_dict_has_convrot_int8(image_proj)
    has_marker_ip = ip8.state_dict_has_convrot_int8(ip_adapter)
    step("state_dict_has_convrot_int8", image_proj=has_marker_proj, ip_adapter=has_marker_ip)

    try:
        from extensions_builtin_shim import _noop  # noqa: F401
    except Exception:
        pass

    # The real entry point: scripts/ipadapter/ipadapter_model.py builds the graph and
    # calls prepare_state_dict itself. Import it by path so A1111 startup is not run.
    ipm_path = os.path.join("extensions-builtin", "sd-webui-controlnet", "scripts",
                            "ipadapter", "ipadapter_model.py")
    model = None
    if os.path.exists(ipm_path):
        import importlib.util, types
        try:
            # ipadapter_model.py uses relative imports, so it needs its real package
            # identity: register scripts.ipadapter as a package pointing at the dir.
            pkg_dir = os.path.dirname(os.path.abspath(ipm_path))
            scripts_dir = os.path.dirname(os.path.dirname(pkg_dir))
            # Plus adapters read CLIP-vision hidden states and import the extension's
            # annotator package, which sits next to scripts/ in the extension root.
            ext_root = os.path.dirname(scripts_dir)
            if ext_root not in sys.path:
                sys.path.insert(0, ext_root)
            for name, path in (("scripts", scripts_dir),
                               ("scripts.ipadapter", pkg_dir)):
                if name not in sys.modules:
                    mod_pkg = types.ModuleType(name)
                    mod_pkg.__path__ = [path]
                    sys.modules[name] = mod_pkg
            spec = importlib.util.spec_from_file_location(
                "scripts.ipadapter.ipadapter_model", ipm_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["scripts.ipadapter.ipadapter_model"] = mod
            spec.loader.exec_module(mod)
            step("import ipadapter_model.py", ok=True, file=ipm_path)
            model = mod.IPAdapterModel.load(nested, os.path.basename(pack_path))
            step("IPAdapterModel.load", ok=True, type=type(model).__name__)
        except Exception as e:
            step("IPAdapterModel.load", ok=False,
                 error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc()[-600:])
    else:
        step("import ipadapter_model.py", ok=False, error="not found at " + ipm_path)

    if model is None:
        return out

    int8_mods = []
    for name, m in model.named_modules():
        if getattr(m, "_hswq_kind", None) == "linear" and m.weight.dtype == torch.int8:
            int8_mods.append({"name": name, "shape": list(m.weight.shape),
                              "groupsize": int(getattr(m, "_hswq_gs", -1))})
    step("installed INT8 Linears", count=len(int8_mods),
         image_proj=sum(1 for m in int8_mods if "image_proj" in m["name"]),
         ip_layers=sum(1 for m in int8_mods if "to_kvs" in m["name"]),
         sample=int8_mods[:3])

    float_params = [(n, tuple(p.shape), str(p.dtype)) for n, p in model.named_parameters()
                    if p.dtype == torch.float32][:4]
    step("float parameters", count=sum(1 for _ in model.named_parameters()),
         sample=float_params)

    # forward through the node's own entry, on the GPU, with the INT8 kernel counted
    counted = {"n": 0}
    if hasattr(kc, "int8_linear"):
        orig = kc.int8_linear
        def wrapper(*a, **kw):
            counted["n"] += 1
            return orig(*a, **kw)
        kc.int8_linear = wrapper

    try:
        model.device = device
        model.to(device)
        if "faceid" not in os.path.basename(pack_path):
            # Resampler (Plus) and basic XL both go through _get_image_embeds, which reads
            # clip_vision_output["image_embeds"]; only FaceID takes a bare 512-d embedding.
            emb = torch.randn(1, 257, 1280, device=device, dtype=torch.float32)
        else:
            emb = torch.randn(1, 512, device=device, dtype=torch.float32)
        with torch.inference_mode():
            if "faceid" not in os.path.basename(pack_path):
                from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
                cvo = CLIPVisionModelOutput(image_embeds=torch.randn(1, 1280, device=device, dtype=torch.float32),
                                            last_hidden_state=emb,
                                            hidden_states=(emb, emb))
                emb_out = model.get_image_emb(cvo)
            else:
                emb_out = model.get_image_emb(emb)
        step("get_image_emb (forward)", ok=True,
             cond_shape=list(emb_out.cond_emb.shape),
             finite=bool(torch.isfinite(emb_out.cond_emb).all()),
             int8_linear_calls=counted["n"])
        out["cond_mean"] = float(emb_out.cond_emb.mean())
        out["cond_std"] = float(emb_out.cond_emb.std())
    except Exception as e:
        step("get_image_emb (forward)", ok=False,
             error="%s: %s" % (type(e).__name__, e), tb=traceback.format_exc()[-800:])
        if hasattr(kc, "int8_linear"):
            kc.int8_linear = orig

    out["int8_linears"] = int8_mods
    out["int8_linear_calls"] = counted["n"]
    out["ok"] = bool(out["int8_linear_calls"] > 0)
    return out


def forge_route(pack_path):
    out = {"route": "forge", "ok": False, "steps": []}

    def step(name, **kw):
        rec = {"step": name}
        rec.update(kw)
        out["steps"].append(rec)
        print("  [forge] %-28s %s" % (name, kw), flush=True)

    import torch
    try:
        import comfy.utils as cu
        sd = cu.load_torch_file(pack_path)
        step("comfy.utils.load_torch_file", ok=True, keys=len(sd))
    except Exception as e:
        step("comfy.utils.load_torch_file", ok=False, error="%s: %s" % (type(e).__name__, e))
        try:
            sd = torch.load(pack_path, map_location="cpu", weights_only=True)
            flat = {}
            for k, v in sd.get("image_proj", {}).items():
                flat["image_proj." + k] = v
            for k, v in sd.get("ip_adapter", {}).items():
                flat["ip_adapter." + k] = v
            sd = flat
            step("torch.load fallback", ok=True, keys=len(sd))
        except Exception as e2:
            step("torch.load fallback", ok=False, error=str(e2))
            return out

    markers = [k for k in sd if k.endswith(".comfy_quant")]
    step("comfy_quant markers", count=len(markers), sample=sorted(markers)[:2])

    # What backend/operations_mixed_precision.py does at L75: pop the marker per prefix.
    popped = {}
    for k in markers:
        prefix = k[: -len(".comfy_quant")]
        conf = sd.pop(k)
        popped[prefix] = conf
    step("markers popped (L75 behaviour)", count=len(popped))

    # L118-119: read the convrot flag out of the marker.
    convrot = 0
    formats = {}
    for prefix, conf in popped.items():
        if isinstance(conf, dict):
            if conf.get("convrot"):
                convrot += 1
            fmt = conf.get("format", "")
            formats[fmt] = formats.get(fmt, 0) + 1
    step("markers with convrot flag", count=convrot, formats=formats)

    # The layout registry the backend would dispatch on.
    try:
        import importlib.util
        qp = os.path.join("backend", "quant_ops.py")
        spec = importlib.util.spec_from_file_location("_forge_quant_ops", qp)
        qmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qmod)
        step("backend/quant_ops.py", ok=True,
             has_convrot_entry=("convrot_w4a4" in getattr(qmod, "QUANT_ALGOS", {})
                                if hasattr(qmod, "QUANT_ALGOS") else "unresolved"))
    except Exception as e:
        step("backend/quant_ops.py", ok=False, error="%s: %s" % (type(e).__name__, e))

    out["convrot_markers"] = convrot
    out["ok"] = convrot > 0
    return out


def forge_linear_route(pack_path, device):
    """Load the pack's declared layers into Forge's own quantised Linear and run them.

    `backend/operations_mixed_precision.py` reads the marker at L75-76
    (`json.loads(layer_conf.numpy().tobytes())`) and builds a QuantizedTensor with
    `convrot` / `convrot_groupsize`, then dispatches to `ck.int8_linear(convrot=True)`
    at the bottom of the file. This runs that path on this pack.
    """
    out = {"route": "forge", "ok": False, "steps": []}

    def step(name, **kw):
        rec = {"step": name}
        rec.update(kw)
        out["steps"].append(rec)
        print("  [forge] %-30s %s" % (name, kw), flush=True)

    import torch
    # backend/operations.py L539 imports mixed_precision itself, so it is the entry
    # point; importing the submodule first hits a circular initialisation.
    try:
        import backend.operations as _bo  # noqa: F401
        from backend.operations_mixed_precision import mixed_precision_ops
        step("import backend.operations -> mixed_precision", ok=True)
    except Exception as e:
        step("import backend.operations -> mixed_precision", ok=False,
             error="%s: %s" % (type(e).__name__, e))
        return out

    import comfy.utils as cu
    sd = cu.load_torch_file(pack_path)
    markers = sorted(k[: -len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant"))
    step("comfy_quant markers", count=len(markers))

    import json as _json
    conf0 = _json.loads(sd[markers[0] + ".comfy_quant"].numpy().tobytes())
    step("marker decoded (L76)", layer=markers[0], conf=conf0)

    Ops = mixed_precision_ops()
    loaded, failed = 0, []
    mods = {}
    for prefix in markers:
        w = sd[prefix + ".weight"]
        sc = sd[prefix + ".weight_scale"]
        mk = sd[prefix + ".comfy_quant"]
        # _load_quantized_module pops f"{prefix}weight", so the dict keys keep their
        # full dotted names and the prefix passed in must match them.
        sub = {prefix + ".weight": w, prefix + ".weight_scale": sc,
               prefix + ".comfy_quant": mk}
        try:
            m = Ops.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
            m._load_from_state_dict(sub, prefix + ".", {}, False, [], [], [])
            mods[prefix] = m
            loaded += 1
        except Exception as e:
            failed.append({"layer": prefix, "error": "%s: %s" % (type(e).__name__, e)})
    step("quantised Linears loaded", loaded=loaded, failed=len(failed),
         first_failure=failed[0] if failed else None)

    if not loaded:
        return out

    import comfy_kitchen.backends.cuda as kc
    counted = {"n": 0}
    orig = getattr(kc, "int8_linear", None)
    if orig is not None:
        def wrapper(*a, **kw):
            counted["n"] += 1
            return orig(*a, **kw)
        kc.int8_linear = wrapper

    # Forge's Linear dispatches through the layout class, so count on the layout too.
    try:
        from backend.quant_ops import TensorWiseINT8Layout
        orig_deq = getattr(TensorWiseINT8Layout, "dequantize", None)
    except Exception:
        TensorWiseINT8Layout = None

    results = []
    try:
        for prefix, m in list(mods.items())[:3]:
            m = m.to(device)
            x = torch.randn(2, m.in_features, device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                y = m(x)
            results.append({"layer": prefix, "out_shape": list(y.shape),
                            "finite": bool(torch.isfinite(y).all())})
        step("forward 3 layers", results=results, int8_linear_calls=counted["n"])
    except Exception as e:
        step("forward 3 layers", ok=False, error="%s: %s" % (type(e).__name__, e))

    if orig is not None:
        kc.int8_linear = orig

    out["loaded"] = loaded
    out["failed"] = failed[:5]
    out["int8_linear_calls"] = counted["n"]
    out["ok"] = bool(loaded == len(markers) and counted["n"] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--root", required=True, help="the A1111 or Forge install root")
    ap.add_argument("--which", required=True, choices=["a1111", "forge", "both"])
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.root))
    result = {"pack": os.path.basename(a.pack), "root": os.path.abspath(a.root)}

    if a.which in ("a1111", "both"):
        print("[A1111]", flush=True)
        result["a1111"] = a1111_route(a.pack, a.device)
    if a.which in ("forge", "both"):
        print("[Forge]", flush=True)
        result["forge"] = forge_linear_route(a.pack, a.device)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()
