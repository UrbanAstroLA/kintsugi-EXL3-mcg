#!/usr/bin/env python3
"""
Analyze Phase 2 Stage 2B results against Phase 1 k325-rebuilt-r0 and Stage 2A benchmarks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

P1_K4_B_CLOSE_HEADROOM = 0.277259
P1_K4_PB_KL_HEADROOM = 0.0130739534622925

ARMS_META = [
    ("null-shuf-b2", "null", 0),
    ("null-shuf-b5", "null", 0),
    ("b2-role-gate", "role_b2", 1485),
    ("b2-role-up", "role_b2", 1323),
    ("b2-role-down", "role_b2", 1080),
    ("b1b2-hot-route-3888", "hot_3888", 3888),
    ("b1b2-hot-routed-err-3888", "hot_3888", 3888),
    ("iso-b0b5b6-to-b2", "iso_0b", 0),
    ("iso-b0b5b6-to-hot-b1b2", "iso_0b", 0),
    ("b1b2-up-k4", "full_b1b2", 7776),
]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", required=True)
    ap.add_argument("--phase2-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p1 = Path(args.phase1_dir)
    p2 = Path(args.phase2_dir)
    base_m = load_margins(p1 / "margins" / "k325-rebuilt-r0.margins.json")
    ids = sorted(base_m.keys())

    rows = []
    for arm, cat, delta_mib in ARMS_META:
        mp = p2 / "margins" / f"{arm}.margins.json"
        pp = p2 / "bf16ref" / "scores" / "pb" / f"pair-{arm}-vs-k325-rebuilt-r0.json"
        vp = p2 / "bf16ref" / "scores" / "pb" / f"validate-{arm}.json"
        if not mp.exists() or not pp.exists():
            continue
        fm = load_margins(mp)
        db = [fm[i]["comp_b"]["close_margin"] - base_m[i]["comp_b"]["close_margin"] for i in ids]
        b_vals = [fm[i]["comp_b"]["close_margin"] for i in ids]
        b_stat = paired_boot_ci(db)
        pb = json.loads(pp.read_text())["gen"]["overall"]
        dkl = pb["kl"]["B_minus_A"]
        dkl_ci = pb["kl"]["ci95"]
        val_pass = json.loads(vp.read_text()).get("pass") if vp.exists() else None
        gib = delta_mib / 1024.0
        rows.append(
            {
                "arm": arm,
                "category": cat,
                "delta_mib": delta_mib,
                "delta_gib": round(gib, 4),
                "bpw": round(3.0 + (9072 + delta_mib) / 37152.0, 4),
                "pb_gen_kl": round(pb["kl"]["B"], 6),
                "pb_gen_dkl": round(dkl, 6),
                "pb_gen_dkl_ci95": [round(dkl_ci[0], 6), round(dkl_ci[1], 6)],
                "pb_kl_headroom_pct": round(100.0 * (-dkl) / P1_K4_PB_KL_HEADROOM, 2),
                "pb_kl_per_gib": round((-dkl) / gib, 6) if gib > 0 else None,
                "pb_gen_top1": round(pb["top1"]["B"], 6),
                "b_close_mean": round(sum(b_vals) / len(b_vals), 4),
                "delta_b_close": b_stat,
                "b_close_headroom_pct": round(100.0 * b_stat["mean"] / P1_K4_B_CLOSE_HEADROOM, 2),
                "b_close_per_gib": round(b_stat["mean"] / gib, 6) if gib > 0 else None,
                "items_higher_than_base": sum(1 for x in db if x > 0.0),
                "new_neg_items": [i for i in ids if base_m[i]["comp_b"]["close_margin"] >= 0.0 and fm[i]["comp_b"]["close_margin"] < 0.0],
                "pb_validate_pass": val_pass,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "kintsugi-phase2b-summary-v1", "baseline_arm": "k325-rebuilt-r0", "arms": rows}
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== PHASE 2B RESULTS (vs k325-rebuilt-r0) ===")
    for r in rows:
        ci = r["delta_b_close"]["ci95"]
        kci = r["pb_gen_dkl_ci95"]
        print(
            f"  {r['arm']:26s} ({r['delta_gib']:+5.2f}G): "
            f"dKL={r['pb_gen_dkl']:+.6f} [{kci[0]:+.6f},{kci[1]:+.6f}] ({r['pb_kl_headroom_pct']:+5.1f}% hdr) | "
            f"dB={r['delta_b_close']['mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] ({r['b_close_headroom_pct']:+5.1f}% hdr, +{r['items_higher_than_base']}/24) "
            f"val={r['pb_validate_pass']}"
        )


if __name__ == "__main__":
    main()
