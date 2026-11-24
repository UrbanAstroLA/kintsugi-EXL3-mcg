#!/usr/bin/env python3
"""
Pre-data Amendment 3B.1 analyzer for Phase 3 Stage 3B 4K + 127K Context-Probe.

Companion to `tools/analyze_phase3b.py` (which is kept untouched to preserve
`stage3b/FROZEN.json` SHA-256 verification during live runs). Adds:
  1. Paired item-cluster bootstrap 95% CIs (10,000 resamples, seed 20261123)
     for `acc_recovery_share_of_G_anchor` and `unf_recovery_share_of_K4`.
  2. Unfinished-reduction prerequisite on `L3.3`: the unfinished path counts
     ONLY if `k4-bf16core`'s own paired unfinished-rate reduction vs
     `k325-native` is itself detected (95% CI strictly < 0).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_arm_tasks(tasks_root: Path, arm: str, run_tag: str) -> Dict[Tuple[str, str, int], dict]:
    out: Dict[Tuple[str, str, int], dict] = {}
    base = tasks_root / arm / run_tag
    if not base.exists():
        return out
    for rep_dir in sorted(base.glob("single-rep*")):
        p = rep_dir / "requests.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("final", True) or r.get("status") != "ok":
                continue
            out[(str(r["length"]), str(r["item"]), int(r["rep"]))] = r
    return out


def item_score(r: dict) -> float:
    sc = r.get("score") or {}
    if "score" in sc and sc["score"] is not None:
        return float(sc["score"])
    fields = sc.get("fields") or 0
    correct = sc.get("correct") or 0
    return float(correct) / float(fields) if fields else 0.0


def summarize_arm(rows: Dict[Tuple[str, str, int], dict], length: str) -> dict:
    sub = {k: v for k, v in rows.items() if k[0] == length}
    n = len(sub)
    if n == 0:
        return {"n": 0}
    scores = [item_score(v) for v in sub.values()]
    fc = Counter(v.get("finish_class", "unknown") for v in sub.values())
    fin = [item_score(v) for v in sub.values() if v.get("finish_class") == "stop"]
    return {
        "n": n,
        "mean_score": round(sum(scores) / n, 6),
        "fin_only_n": len(fin),
        "fin_only_score": round(sum(fin) / len(fin), 6) if fin else None,
        "unfinished": n - fc.get("stop", 0),
        "unfinished_rate": round((n - fc.get("stop", 0)) / n, 6),
        "finish_classes": dict(sorted(fc.items())),
    }


def item_series(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    length: str,
) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    common = sorted(k for k in rows_a if k[0] == length and k in rows_b)
    by_item: Dict[str, List[Tuple[float, float, int, int]]] = defaultdict(list)
    for k in common:
        ra, rb = rows_a[k], rows_b[k]
        ua = 0 if ra.get("finish_class") == "stop" else 1
        ub = 0 if rb.get("finish_class") == "stop" else 1
        by_item[k[1]].append((item_score(ra), item_score(rb), ua, ub))
    items = sorted(by_item.keys())
    sa = [sum(a for a, _, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sb = [sum(b for _, b, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sd = [b - a for a, b in zip(sa, sb)]
    ud = [sum(ub - ua for _, _, ua, ub in by_item[i]) / len(by_item[i]) for i in items]
    return items, sa, sb, sd, ud


def paired_item_boot(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    length: str,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    items, sa, sb, sd, ud = item_series(rows_a, rows_b, length)
    n_items = len(items)
    if n_items == 0:
        return {"n_items": 0}
    rng = random.Random(seed)
    bd, bu = [], []
    for _ in range(n_boot):
        s_d = s_u = 0.0
        for _ in range(n_items):
            idx = rng.randrange(n_items)
            s_d += sd[idx]
            s_u += ud[idx]
        bd.append(s_d / n_items)
        bu.append(s_u / n_items)
    bd.sort()
    bu.sort()
    lo_d, hi_d = bd[int(0.025 * n_boot)], bd[int(0.975 * n_boot)]
    lo_u, hi_u = bu[int(0.025 * n_boot)], bu[int(0.975 * n_boot)]
    return {
        "n_items": n_items,
        "mean_A": round(sum(sa) / n_items, 6),
        "mean_B": round(sum(sb) / n_items, 6),
        "B_minus_A": round(sum(sd) / n_items, 6),
        "ci95": [round(lo_d, 6), round(hi_d, 6)],
        "excludes_zero": bool(lo_d > 0.0 or hi_d < 0.0),
        "unf_rate_B_minus_A": round(sum(ud) / n_items, 6),
        "unf_rate_ci95": [round(lo_u, 6), round(hi_u, 6)],
        "unf_excludes_zero": bool(lo_u > 0.0 or hi_u < 0.0),
    }


def share_boot_ci(
    num_diffs: List[float],
    den_diffs: List[float],
    n_boot: int = 10000,
    seed: int = 20261123,
) -> Tuple[Optional[float], Optional[List[float]]]:
    n = len(num_diffs)
    den_mean = sum(den_diffs) / n
    if abs(den_mean) < 1e-12:
        return None, None
    pt = (sum(num_diffs) / n) / den_mean
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        sn = sd = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            sn += num_diffs[idx]
            sd += den_diffs[idx]
        if abs(sd) > 1e-12:
            ratios.append(sn / sd)
    if len(ratios) < 100:
        return round(pt, 4), None
    ratios.sort()
    return round(pt, 4), [round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios))], 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3b-dir", required=True)
    ap.add_argument("--run-tag", default="p3b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s3b = Path(args.stage3b_dir)
    tasks_root = s3b / "tasks"
    arm_names = ["k325-native", "k4-bf16core", "k345-b1b2", "k325-iso-hot"]
    loaded = {a: load_arm_tasks(tasks_root, a, args.run_tag) for a in arm_names}
    present = [a for a in arm_names if loaded[a]]

    levels = {a: {"4k": summarize_arm(loaded[a], "4k"), "128k": summarize_arm(loaded[a], "128k")} for a in present}
    contrasts = {"4k": {}, "128k": {}}
    for length in ("4k", "128k"):
        if "k325-native" in present:
            for a in present:
                if a != "k325-native":
                    contrasts[length][f"{a}_vs_k325-native"] = paired_item_boot(loaded["k325-native"], loaded[a], length)
        if "k4-bf16core" in present:
            for a in ("k345-b1b2", "k325-iso-hot"):
                if a in present:
                    contrasts[length][f"{a}_vs_k4-bf16core"] = paired_item_boot(loaded["k4-bf16core"], loaded[a], length)

    g_anchor = contrasts["128k"].get("k4-bf16core_vs_k325-native")
    anchor_detected = bool(g_anchor and g_anchor.get("ci95", [0.0, 0.0])[0] > 0.0)
    k4_unf_detected = bool(g_anchor and g_anchor.get("unf_rate_ci95", [0.0, 0.0])[1] < 0.0)

    l3_3_v2 = {}
    if "k325-native" in present and "k4-bf16core" in present:
        items_k4, _, _, sd_k4, ud_k4 = item_series(loaded["k325-native"], loaded["k4-bf16core"], "128k")
        neg_ud_k4 = [-u for u in ud_k4]
        for cand in ("k345-b1b2", "k325-iso-hot"):
            if cand not in present:
                continue
            items_c, _, _, sd_c, ud_c = item_series(loaded["k325-native"], loaded[cand], "128k")
            assert items_c == items_k4, f"Item set mismatch for {cand}"
            acc_share, acc_share_ci = share_boot_ci(sd_c, sd_k4)
            unf_share, unf_share_ci = share_boot_ci([-u for u in ud_c], neg_ud_k4)
            c4 = contrasts["4k"][f"{cand}_vs_k325-native"]
            no_4k_reg = bool(c4["B_minus_A"] >= -0.015 or not c4["excludes_zero"])
            fin_ok = bool(
                levels[cand]["128k"]["fin_only_score"] is not None
                and levels["k325-native"]["128k"]["fin_only_score"] is not None
                and levels[cand]["128k"]["fin_only_score"] >= levels["k325-native"]["128k"]["fin_only_score"] - 0.01
            )
            acc_pass = bool(acc_share is not None and acc_share >= 0.50)
            unf_pass = bool(k4_unf_detected and unf_share is not None and unf_share >= 0.50)
            l3_3_v2[cand] = {
                "acc_recovery_share": acc_share,
                "acc_recovery_share_ci95": acc_share_ci,
                "unf_recovery_share": unf_share,
                "unf_recovery_share_ci95": unf_share_ci,
                "k4_unf_reduction_detected": k4_unf_detected,
                "no_4k_regression": no_4k_reg,
                "fin_only_non_inferior": fin_ok,
                "L3_3_v2_pass": bool(anchor_detected and no_4k_reg and fin_ok and (acc_pass or unf_pass)),
            }

    summary = {
        "schema": "kintsugi-phase3b-summary-v2-amendment-3b1",
        "arms_present": present,
        "levels": levels,
        "contrasts": contrasts,
        "anchor_gate": {
            "G_anchor_128k": g_anchor,
            "anchor_gap_detected_ci_gt_0": anchor_detected,
            "k4_unf_reduction_detected_ci_lt_0": k4_unf_detected,
        },
        "L3_3_v2_verdicts": l3_3_v2,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"anchor_gate": summary["anchor_gate"], "L3_3_v2_verdicts": l3_3_v2}, indent=2))


if __name__ == "__main__":
    main()
