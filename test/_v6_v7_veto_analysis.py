"""V6 vs V7 profile veto analysis (no torch)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def short(k: str) -> str:
    return k.replace(".weight", "").split("model.diffusion_model.")[-1]


def hard_veto_set(prof: dict) -> set[str]:
    s = set()
    for k, v in prof["layers"].items():
        if not isinstance(v, dict):
            continue
        kurt = v.get("kurtosis", 0)
        o = v.get("outlier_ratio", 0)
        m = v.get("abs_max", 0)
        if kurt > 20 or o > 40 or m > 20:
            s.add(short(k))
    return s


def outlier_only_set(prof: dict) -> set[str]:
    s = set()
    for k, v in prof["layers"].items():
        if not isinstance(v, dict):
            continue
        kurt = v.get("kurtosis", 0)
        o = v.get("outlier_ratio", 0)
        m = v.get("abs_max", 0)
        if o > 40 and kurt <= 20 and m <= 20:
            s.add(short(k))
    return s


def keypattern_v7_veto(layer_names: list[str]) -> set[str]:
    """Mirror _moody_v7_keypattern_veto in v1.93."""
    _boundary = (".cap_embedder.1", ".final_layer.linear", ".x_embedder")
    _suffixes = (
        ".attention.qkv",
        ".feed_forward.w2",
        ".adaLN_modulation",
        ".attention.out",
    )
    out = set()
    for name in layer_names:
        if name.startswith("t_embedder."):
            out.add(name)
            continue
        if name.endswith(_boundary):
            out.add(name)
            continue
        if any(name.endswith(s) for s in _suffixes):
            out.add(name)
    return {short(x) for x in out}


def main():
    v6 = json.loads((ROOT / "moodyRealMix_zitV6_distribution_profile.json").read_text())
    v7 = json.loads((ROOT / "moodyRealMix_zitV7_distribution_profile.json").read_text())
    layers = sorted(short(k) for k in v7["layers"] if isinstance(v7["layers"][k], dict))

    h6, h7 = hard_veto_set(v6), hard_veto_set(v7)
    oo6, oo7 = outlier_only_set(v6), outlier_only_set(v7)
    kp7 = keypattern_v7_veto(layers)

    print("=== SUMMARY ===")
    print("V6", v6["summary"])
    print("V7", v7["summary"])
    print("hard_veto V6", len(h6), "V7", len(h7))
    print("outlier_only (MSE release risk) V6", len(oo6), "V7", len(oo7))
    print("moodyV7 key-pattern VETO", len(kp7))

    only7 = sorted(h7 - h6)
    only6 = sorted(h6 - h7)
    print("hard_veto only V7:", len(only7))
    for x in only7[:8]:
        print(" ", x)
    print("hard_veto only V6:", len(only6))
    for x in only6[:8]:
        print(" ", x)

    release_overlap = oo7 & h7
    print("V7 outlier_only that are ALSO hard_veto:", len(release_overlap))
    for x in sorted(release_overlap)[:15]:
        print(" ", x)

    # layers hard_veto in V7 but NOT in keypattern and NOT outlier_only
    profile_only = h7 - kp7
    print("profile hard_veto not in keypattern:", len(profile_only))
    for x in sorted(profile_only)[:12]:
        print(" ", x)

    # kurtosis top V7
    items = []
    for k, v in v7["layers"].items():
        if isinstance(v, dict):
            items.append((short(k), v["kurtosis"], v["outlier_ratio"], v["abs_max"]))
    items.sort(key=lambda t: -t[1])
    print("=== TOP kurtosis V7 ===")
    for row in items[:10]:
        print(f"  {row[0]}: k={row[1]:.2f} o={row[2]:.1f} m={row[3]}")

    # w2 with o>40 but not hard veto (k<=20, m<=20)
    w2_outlier = []
    for k, v in v7["layers"].items():
        if not isinstance(v, dict) or not k.endswith("feed_forward.w2.weight"):
            continue
        if v["outlier_ratio"] > 40 and v["kurtosis"] <= 20 and v["abs_max"] <= 20:
            w2_outlier.append((short(k), v["outlier_ratio"], v["abs_max"]))
    w2_outlier.sort(key=lambda t: -t[1])
    print("=== w2 outlier-only (MSE RELEASE candidates) ===", len(w2_outlier))
    for row in w2_outlier[:15]:
        print(f"  {row[0]}: o={row[1]:.1f} m={row[2]}")


if __name__ == "__main__":
    main()
