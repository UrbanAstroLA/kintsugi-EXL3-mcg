#!/usr/bin/env python3
"""
Analyze Phase 3 Stage 3A results:
  - Untouched P-A teacher panel KL (51,175 rows, 25 clusters) vs k325-rebuilt, k325-native, k4-bf16core
  - Untouched D_margin_heldout (n = 36) B_close paired bootstrap CIs
  - Public Neutral-Author D_margin_neutral (n = 24) B_close paired bootstrap CIs (self-authorship control)
  - Gate L3.1 & L3.2 summary.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

ARMS = [
    ("k325-native", 0.0, 3.2442),
    ("k325-rebuilt", 0.0, 3.2442),
    ("k325-iso-hot", 0.0, 3.2442),
    ("k335-b2", 3.7969, 3.3488),
    ("k345-b1b2", 7.5938, 3.4535),
    ("k4-bf16core", 27.4219, 4.0000),
]

P1_K4_PA_HEADROOM = 0.016580565282372197


def paired_boot_ci(diffs: List[float], n_boot: int = 10000, seed: int = 20261123) -> dict:
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {
        "n": n,
        "mean": round(mean, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def load_margins(path: Path) -> Dict[str, dict]:
    d = json.loads(path.read_text())
    return {it["item_id"]: it for it in d["items"]}


def compare_margin_suite(arms_data: Dict[str, Dict[str, dict]], base_arm: str) -> Dict[str, dict]:
    base = arms_data[base_arm]
    ids = sorted(base.keys())
    out = {}
    for arm, m in arms_data.items():
        if arm == base_arm:
            continue
        db = [m[i]["comp_b"]["close_margin"] - base[i]["comp_b"]["close_margin"] for i in ids]
        st = paired_boot_ci(db)
        st["items_higher"] = sum(1 for x in db if x > 0.0)
        st["arm_mean"] = round(sum(m[i]["comp_b"]["close_margin"] for i in ids) / len(ids), 4)
        st["base_mean"] = round(sum(base[i]["comp_b"]["close_margin"] for i in ids) / len(ids), 4)
        out[f"{arm}_vs_{base_arm}"] = st
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p3 = Path(args.phase3_dir)

    # 1. P-A arm levels & paired contrasts
    pa_levels = {}
    for arm, dgib, bpw in ARMS:
        fp = p3 / "bf16ref" / "scores" / "pa" / f"pa-{arm}.json"
        if fp.exists():
            ov = json.loads(fp.read_text())["overall"]
            pa_levels[arm] = {
                "delta_gib": dgib,
                "bpw": bpw,
                "kl": round(ov["kl"]["value"], 6),
                "kl_ci95": [round(ov["kl"]["ci95"][0], 6), round(ov["kl"]["ci95"][1], 6)],
                "nll": round(ov["nll"]["value"], 6),
                "top1": round(ov["top1"]["value"], 6) if "top1" in ov else None,
            }

    pa_pairs = {}
    for pp in sorted((p3 / "bf16ref" / "scores" / "pa").glob("pair-*.json")):
        ov = json.loads(pp.read_text())["overall"]["kl"]
        pa_pairs[pp.stem] = {
            "A": round(ov["A"], 6),
            "B": round(ov["B"], 6),
            "B_minus_A": round(ov["B_minus_A"], 6),
            "ci95": [round(ov["ci95"][0], 6), round(ov["ci95"][1], 6)],
            "excludes_zero": bool(ov["ci95"][0] > 0.0 or ov["ci95"][1] < 0.0),
            "headroom_pct_vs_k4": round(100.0 * (-ov["B_minus_A"]) / P1_K4_PA_HEADROOM, 2),
        }

    # 2. Held-out (n=36) & Neutral-author (n=24) margins
    held = {}
    neut = {}
    for arm, _, _ in ARMS:
        hp = p3 / "margins_heldout" / f"{arm}.margins.json"
        np_ = p3 / "margins_neutral" / f"{arm}.margins.json"
        if hp.exists():
            held[arm] = load_margins(hp)
        if np_.exists():
            neut[arm] = load_margins(np_)

    summary = {
        "schema": "kintsugi-phase3a-summary-v1",
        "pa_levels": pa_levels,
        "pa_pairs": pa_pairs,
        "heldout_n36_vs_rebuilt": compare_margin_suite(held, "k325-rebuilt") if "k325-rebuilt" in held else {},
        "heldout_n36_vs_native": compare_margin_suite(held, "k325-native") if "k325-native" in held else {},
        "neutral_n24_vs_rebuilt": compare_margin_suite(neut, "k325-rebuilt") if "k325-rebuilt" in neut else {},
        "neutral_n24_vs_native": compare_margin_suite(neut, "k325-native") if "k325-native" in neut else {},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== STAGE 3A: UNTOUCHED P-A TEACHER PANEL KL (51,175 rows, 25 clusters) ===")
    for arm, v in pa_levels.items():
        print(f"  {arm:14s} ({v['delta_gib']:+5.2f}G): KL={v['kl']:.6f} CI={v['kl_ci95']} NLL={v['nll']:.6f}")
    print("\n=== STAGE 3A: PAIRED P-A CONTRASTS (B - A) ===")
    for k, v in pa_pairs.items():
        print(f"  {k:38s}: dKL={v['B_minus_A']:+.6f} CI={v['ci95']} ({v['headroom_pct_vs_k4']:+5.1f}% K4 hdr) sig={v['excludes_zero']}")
    print("\n=== STAGE 3A: UNTOUCHED D_margin_heldout (n=36) B_close vs k325-rebuilt ===")
    for k, v in summary["heldout_n36_vs_rebuilt"].items():
        print(f"  {k:34s}: dB={v['mean']:+.4f} CI={v['ci95']} (+{v['items_higher']}/{v['n']}) arm={v['arm_mean']:+.4f} sig={v['excludes_zero']}")
    print("\n=== STAGE 3A: PUBLIC NEUTRAL-AUTHOR D_margin_neutral (n=24) B_close vs k325-rebuilt ===")
    for k, v in summary["neutral_n24_vs_rebuilt"].items():
        print(f"  {k:34s}: dB={v['mean']:+.4f} CI={v['ci95']} (+{v['items_higher']}/{v['n']}) arm={v['arm_mean']:+.4f} sig={v['excludes_zero']}")


if __name__ == "__main__":
    main()
