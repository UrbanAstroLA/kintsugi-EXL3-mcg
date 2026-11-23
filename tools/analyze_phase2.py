#!/usr/bin/env python3
"""
Analyze Phase 2 Stage 2A single-band flip results against Phase 1 k325-rebuilt-r0.

Computes per-band upward (+3,888 MiB) and downward (-1,296 MiB) deltas on:
  - Primary task metric: B_close (</think> closing token 154842 margin, L1.2)
  - Secondary fidelity metric: P-B gen KL (15,104 rows, 53 clusters)
  - Diagnostic metrics: A_min, B_tail_mean, new fragile (<1.0) and negative (<0) hit-list counts
  - Sensitivity per GiB, headroom recovery %, Spearman rank correlations, and Gate P2 (L2.1/L2.2).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BANDS = [
    ("b0", "L03-L08", 3888, 1296),
    ("b1", "L09-L14", 3888, 1296),
    ("b2", "L15-L20", 3888, 1296),
    ("b3", "L21-L26", 3888, 1296),
    ("b4", "L27-L32", 3888, 1296),
    ("b5", "L33-L38", 3888, 1296),
    ("b6", "L39-L44", 3888, 1296),
]

# Phase 1 K4-bf16core vs K3.25-rebuilt headroom anchors ([E1.2], [E1.4])
P1_K4_B_CLOSE_HEADROOM = 0.277259
P1_K4_PB_KL_HEADROOM = 0.0130739534622925


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


def rank_array(vals: List[float]) -> List[float]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    rx = rank_array(xs)
    ry = rank_array(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def load_margins(path: Path) -> Dict[str, dict]:
    d = json.loads(path.read_text())
    return {it["item_id"]: it for it in d["items"]}


def compare_flip(
    arm: str,
    band: str,
    layers: str,
    direction: str,
    delta_mib: int,
    base_m: Dict[str, dict],
    flip_m: Dict[str, dict],
    pb_pair_path: Path,
    pb_val_path: Path,
) -> dict:
    ids = sorted(base_m.keys())
    assert ids == sorted(flip_m.keys()), f"Item mismatch for {arm}"

    db_close = [flip_m[i]["comp_b"]["close_margin"] - base_m[i]["comp_b"]["close_margin"] for i in ids]
    db_tail = [flip_m[i]["comp_b"]["tail_mean_margin"] - base_m[i]["comp_b"]["tail_mean_margin"] for i in ids]
    da_min = [flip_m[i]["comp_a"]["min_margin"] - base_m[i]["comp_a"]["min_margin"] for i in ids]

    b_vals = [flip_m[i]["comp_b"]["close_margin"] for i in ids]
    base_vals = [base_m[i]["comp_b"]["close_margin"] for i in ids]

    new_neg = [i for i in ids if base_m[i]["comp_b"]["close_margin"] >= 0.0 and flip_m[i]["comp_b"]["close_margin"] < 0.0]
    recovered_neg = [i for i in ids if base_m[i]["comp_b"]["close_margin"] < 0.0 and flip_m[i]["comp_b"]["close_margin"] >= 0.0]
    new_frag = [i for i in ids if base_m[i]["comp_b"]["close_margin"] >= 1.0 and flip_m[i]["comp_b"]["close_margin"] < 1.0]
    recovered_frag = [i for i in ids if base_m[i]["comp_b"]["close_margin"] < 1.0 and flip_m[i]["comp_b"]["close_margin"] >= 1.0]

    b_close_stat = paired_boot_ci(db_close)
    delta_gib = delta_mib / 1024.0
    abs_gib = abs(delta_gib)

    pb_pair = json.loads(pb_pair_path.read_text())
    pb_gen = pb_pair["gen"]["overall"]
    pb_val = json.loads(pb_val_path.read_text()) if pb_val_path.exists() else {"pass": None}

    dkl = pb_gen["kl"]["B_minus_A"]
    dkl_ci = pb_gen["kl"]["ci95"]

    # Positive "gain_per_gib" means beneficial sensitivity:
    #   upward:  +ΔB_close / GiB  and  -ΔKL_PB / GiB
    #   downward (damage rate per GiB saved): -ΔB_close / GiB  and  +ΔKL_PB / GiB
    if direction == "up":
        b_sens_gib = b_close_stat["mean"] / abs_gib
        kl_sens_gib = (-dkl) / abs_gib
        b_headroom_pct = 100.0 * b_close_stat["mean"] / P1_K4_B_CLOSE_HEADROOM
        kl_headroom_pct = 100.0 * (-dkl) / P1_K4_PB_KL_HEADROOM
    else:
        b_sens_gib = (-b_close_stat["mean"]) / abs_gib
        kl_sens_gib = dkl / abs_gib
        b_headroom_pct = None
        kl_headroom_pct = None

    return {
        "arm": arm,
        "band": band.upper(),
        "layers": layers,
        "direction": direction,
        "delta_mib": delta_mib,
        "delta_gib": round(delta_gib, 4),
        "b_close_mean": round(sum(b_vals) / len(b_vals), 4),
        "b_close_fragile_lt1": sum(1 for v in b_vals if v < 1.0),
        "b_close_neg_lt0": sum(1 for v in b_vals if v < 0.0),
        "delta_b_close": b_close_stat,
        "items_higher_than_base": sum(1 for d in db_close if d > 0.0),
        "items_lower_than_base": sum(1 for d in db_close if d < 0.0),
        "new_neg_items": new_neg,
        "recovered_neg_items": recovered_neg,
        "new_frag_items": new_frag,
        "recovered_frag_items": recovered_frag,
        "b_close_sens_per_gib": round(b_sens_gib, 6),
        "b_close_headroom_pct": round(b_headroom_pct, 2) if b_headroom_pct is not None else None,
        "delta_b_tail": paired_boot_ci(db_tail),
        "delta_a_min": paired_boot_ci(da_min),
        "pb_gen_kl": round(pb_gen["kl"]["B"], 6),
        "pb_gen_dkl": round(dkl, 6),
        "pb_gen_dkl_ci95": [round(dkl_ci[0], 6), round(dkl_ci[1], 6)],
        "pb_gen_top1": round(pb_gen["top1"]["B"], 6),
        "pb_kl_sens_per_gib": round(kl_sens_gib, 6),
        "pb_kl_headroom_pct": round(kl_headroom_pct, 2) if kl_headroom_pct is not None else None,
        "pb_validate_pass": pb_val.get("pass"),
    }


def ratio_safe(hi: float, lo: float) -> Optional[float]:
    if lo <= 0.0:
        return None if hi <= 0.0 else float("inf")
    return hi / lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", required=True)
    ap.add_argument("--phase2-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p1 = Path(args.phase1_dir)
    p2 = Path(args.phase2_dir)

    base_m = load_margins(p1 / "margins" / "k325-rebuilt-r0.margins.json")
    up_rows: List[dict] = []
    dn_rows: List[dict] = []

    for band, layers, k3_cnt, k4_cnt in BANDS:
        up_arm = f"{band}-up-k4"
        up_m_path = p2 / "margins" / f"{up_arm}.margins.json"
        if up_m_path.exists():
            up_rows.append(
                compare_flip(
                    arm=up_arm,
                    band=band,
                    layers=layers,
                    direction="up",
                    delta_mib=+k3_cnt,
                    base_m=base_m,
                    flip_m=load_margins(up_m_path),
                    pb_pair_path=p2 / "bf16ref" / "scores" / "pb" / f"pair-{up_arm}-vs-k325-rebuilt-r0.json",
                    pb_val_path=p2 / "bf16ref" / "scores" / "pb" / f"validate-{up_arm}.json",
                )
            )
        dn_arm = f"{band}-down-k3"
        dn_m_path = p2 / "margins" / f"{dn_arm}.margins.json"
        if dn_m_path.exists():
            dn_rows.append(
                compare_flip(
                    arm=dn_arm,
                    band=band,
                    layers=layers,
                    direction="down",
                    delta_mib=-k4_cnt,
                    base_m=base_m,
                    flip_m=load_margins(dn_m_path),
                    pb_pair_path=p2 / "bf16ref" / "scores" / "pb" / f"pair-{dn_arm}-vs-k325-rebuilt-r0.json",
                    pb_val_path=p2 / "bf16ref" / "scores" / "pb" / f"validate-{dn_arm}.json",
                )
            )

    # Rankings & Gate P2 (L2.1 / L2.2)
    up_by_b = sorted(up_rows, key=lambda r: r["b_close_sens_per_gib"], reverse=True)
    up_by_kl = sorted(up_rows, key=lambda r: r["pb_kl_sens_per_gib"], reverse=True)
    safe_dn = [r for r in dn_rows if len(r["new_neg_items"]) == 0]
    dn_by_b_cheap = sorted(safe_dn, key=lambda r: r["b_close_sens_per_gib"])
    dn_by_kl_cheap = sorted(safe_dn, key=lambda r: r["pb_kl_sens_per_gib"])

    rho_up_b_vs_kl = None
    rho_up_vs_dn_b = None
    rho_up_vs_dn_kl = None
    if len(up_rows) == 7:
        rho_up_b_vs_kl = spearman_rho(
            [r["b_close_sens_per_gib"] for r in up_rows],
            [r["pb_kl_sens_per_gib"] for r in up_rows],
        )
    if len(up_rows) == 7 and len(dn_rows) == 7:
        rho_up_vs_dn_b = spearman_rho(
            [r["b_close_sens_per_gib"] for r in up_rows],
            [r["b_close_sens_per_gib"] for r in dn_rows],
        )
        rho_up_vs_dn_kl = spearman_rho(
            [r["pb_kl_sens_per_gib"] for r in up_rows],
            [r["pb_kl_sens_per_gib"] for r in dn_rows],
        )

    # Sensitivity spreads (max/min upward AND best upward sink vs cheapest safe source)
    up_kl_spread = ratio_safe(up_by_kl[0]["pb_kl_sens_per_gib"], up_by_kl[-1]["pb_kl_sens_per_gib"]) if up_by_kl else None
    up_b_spread = ratio_safe(up_by_b[0]["b_close_sens_per_gib"], up_by_b[-1]["b_close_sens_per_gib"]) if up_by_b else None
    sink_vs_src_kl = (
        ratio_safe(up_by_kl[0]["pb_kl_sens_per_gib"], dn_by_kl_cheap[0]["pb_kl_sens_per_gib"])
        if (up_by_kl and dn_by_kl_cheap)
        else None
    )
    sink_vs_src_b = (
        ratio_safe(up_by_b[0]["b_close_sens_per_gib"], dn_by_b_cheap[0]["b_close_sens_per_gib"])
        if (up_by_b and dn_by_b_cheap)
        else None
    )

    def ge2(x: Optional[float]) -> bool:
        return x is not None and (math.isinf(x) or x >= 2.0)

    l2_1_safe_source_exists = len(safe_dn) >= 1
    l2_1_spread_ge_2 = any(ge2(x) for x in (up_kl_spread, up_b_spread, sink_vs_src_kl, sink_vs_src_b))
    l2_1_orig_pass = bool(l2_1_safe_source_exists and l2_1_spread_ge_2)

    # Tightened Gate P2 (L2.1-v2, pre-data Amendment 2A.1):
    # 1a: upward P-B KL max/min >= 1.5x with min > 0 and min's paired 95% CI vs k325-rebuilt excluding 0
    #     (direct paired top-vs-bottom P-B KL cluster-bootstrap CI verified via score_pb.py pair on CPU)
    # 1b: cross-direction consistency rho_up_down_KL >= 0.60
    min_up_kl_pos_sig = bool(
        up_by_kl
        and up_by_kl[-1]["pb_kl_sens_per_gib"] > 0.0
        and up_by_kl[-1]["pb_gen_dkl_ci95"][1] < 0.0
    )
    l2_1_v2_kl_spread_ge_1_5 = bool(
        min_up_kl_pos_sig
        and up_kl_spread is not None
        and not math.isinf(up_kl_spread)
        and up_kl_spread >= 1.5
    )
    l2_1_v2_rho_ge_0_60 = bool(rho_up_vs_dn_kl is not None and rho_up_vs_dn_kl >= 0.60)
    l2_1_v2_non_overlapping_top_bot = bool(
        up_by_kl
        and up_by_kl[0]["pb_gen_dkl_ci95"][1] < up_by_kl[-1]["pb_gen_dkl_ci95"][0]
    )
    l2_1_v2_pass_prelim = bool(l2_1_v2_kl_spread_ge_1_5 and l2_1_v2_rho_ge_0_60)

    def fmt_r(x: Optional[float]):
        if x is None:
            return None
        if math.isinf(x):
            return "inf_neg_denom"
        return round(x, 4)

    summary = {
        "schema": "kintsugi-phase2a-summary-v1",
        "baseline_arm": "k325-rebuilt-r0",
        "upward_flips": up_rows,
        "downward_flips": dn_rows,
        "rankings": {
            "upward_by_b_close": [r["band"] for r in up_by_b],
            "upward_by_pb_kl": [r["band"] for r in up_by_kl],
            "safe_sources_hit0": [r["band"] for r in safe_dn],
            "cheapest_safe_by_b_close": [r["band"] for r in dn_by_b_cheap],
            "cheapest_safe_by_pb_kl": [r["band"] for r in dn_by_kl_cheap],
        },
        "correlations": {
            "spearman_up_b_close_vs_pb_kl": round(rho_up_b_vs_kl, 4) if rho_up_b_vs_kl is not None else None,
            "spearman_up_vs_down_b_close": round(rho_up_vs_dn_b, 4) if rho_up_vs_dn_b is not None else None,
            "spearman_up_vs_down_pb_kl": round(rho_up_vs_dn_kl, 4) if rho_up_vs_dn_kl is not None else None,
        },
        "gates": {
            "L2_1_v2_active_pass_prelim": l2_1_v2_pass_prelim,
            "L2_1_v2_kl_spread_ge_1_5": l2_1_v2_kl_spread_ge_1_5,
            "L2_1_v2_min_up_kl_pos_sig": min_up_kl_pos_sig,
            "L2_1_v2_non_overlapping_top_bot_ci": l2_1_v2_non_overlapping_top_bot,
            "L2_1_v2_rho_up_down_kl_ge_0_60": l2_1_v2_rho_ge_0_60,
            "L2_1_orig_archived_pass": l2_1_orig_pass,
            "L2_1_safe_source_exists": l2_1_safe_source_exists,
            "L2_1_upward_b_close_spread": fmt_r(up_b_spread),
            "L2_1_upward_pb_kl_spread": fmt_r(up_kl_spread),
            "L2_1_sink_vs_safe_src_b_close_spread": fmt_r(sink_vs_src_b),
            "L2_1_sink_vs_safe_src_pb_kl_spread": fmt_r(sink_vs_src_kl),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== PHASE 2A UPWARD FLIPS (+3.7969 GiB -> 100% K4) ===")
    for r in up_rows:
        ci = r["delta_b_close"]["ci95"]
        print(
            f"  {r['band']} ({r['layers']}): dB={r['delta_b_close']['mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
            f"({r['b_close_sens_per_gib']:+.4f}/GiB, {r['b_close_headroom_pct']:+5.1f}% hdr) | "
            f"dKL_PB={r['pb_gen_dkl']:+.6f} ({r['pb_kl_sens_per_gib']:+.6f}/GiB, {r['pb_kl_headroom_pct']:+5.1f}% hdr) | "
            f"frag={r['b_close_fragile_lt1']} neg={r['b_close_neg_lt0']} val={r['pb_validate_pass']}"
        )
    print("\n=== PHASE 2A DOWNWARD FLIPS (-1.2656 GiB -> 0% K4) ===")
    for r in dn_rows:
        ci = r["delta_b_close"]["ci95"]
        print(
            f"  {r['band']} ({r['layers']}): dB={r['delta_b_close']['mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
            f"(dmg={r['b_close_sens_per_gib']:+.4f}/GiB) | "
            f"dKL_PB={r['pb_gen_dkl']:+.6f} (dmg={r['pb_kl_sens_per_gib']:+.6f}/GiB) | "
            f"new_neg={len(r['new_neg_items'])} new_frag={len(r['new_frag_items'])} val={r['pb_validate_pass']}"
        )
    print("\n=== CORRELATIONS & GATE P2 ===")
    print(json.dumps({"correlations": summary["correlations"], "gates": summary["gates"], "rankings": summary["rankings"]}, indent=2))


if __name__ == "__main__":
    main()
