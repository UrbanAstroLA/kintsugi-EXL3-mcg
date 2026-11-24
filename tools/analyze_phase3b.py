#!/usr/bin/env python3
"""
Analyze Phase 3 Stage 3B 4K + 127K Context-Probe task confirmation results.

Computes per-arm overall accuracy, finished-only accuracy, unfinished counts
(cap / repetition / other), paired item-clustered bootstrap 95% CIs (10,000
resamples, seed 20261123) across all 60 measured items x 4 passes (240 replies/length),
evaluates the pre-registered `G_anchor = k4-bf16core - k325-native` prerequisite
gate (`Amendment 3A.1`), and evaluates `L3.3` advancement into Phase 4.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_arm_tasks(tasks_root: Path, arm: str, run_tag: str) -> Dict[Tuple[str, str, int], dict]:
    """Return {(length, item_id, rep): row} for all final ok records."""
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
            key = (str(r["length"]), str(r["item"]), int(r["rep"]))
            out[key] = r
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
        "finish_classes": dict(sorted(fc.items())),
    }


def paired_item_boot(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    length: str,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    """Cluster bootstrap over the 60 items (averaging matched reps within each item)."""
    common = sorted(k for k in rows_a if k[0] == length and k in rows_b)
    by_item: Dict[str, List[Tuple[float, float, int, int]]] = defaultdict(list)
    for k in common:
        ra, rb = rows_a[k], rows_b[k]
        ua = 0 if ra.get("finish_class") == "stop" else 1
        ub = 0 if rb.get("finish_class") == "stop" else 1
        by_item[k[1]].append((item_score(ra), item_score(rb), ua, ub))

    items = sorted(by_item.keys())
    n_items = len(items)
    if n_items == 0:
        return {"n_pairs": 0, "n_items": 0}

    item_diffs = [sum(b - a for a, b, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    item_a = [sum(a for a, _, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    item_b = [sum(b for _, b, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    unf_diffs = [sum(ub - ua for _, _, ua, ub in by_item[i]) / len(by_item[i]) for i in items]

    mean_a = sum(item_a) / n_items
    mean_b = sum(item_b) / n_items
    mean_d = sum(item_diffs) / n_items
    mean_ud = sum(unf_diffs) / n_items

    rng = random.Random(seed)
    boot_d = []
    boot_u = []
    for _ in range(n_boot):
        sd = 0.0
        su = 0.0
        for _ in range(n_items):
            idx = rng.randrange(n_items)
            sd += item_diffs[idx]
            su += unf_diffs[idx]
        boot_d.append(sd / n_items)
        boot_u.append(su / n_items)
    boot_d.sort()
    boot_u.sort()
    lo_d, hi_d = boot_d[int(0.025 * n_boot)], boot_d[int(0.975 * n_boot)]
    lo_u, hi_u = boot_u[int(0.025 * n_boot)], boot_u[int(0.975 * n_boot)]
    return {
        "n_pairs": len(common),
        "n_items": n_items,
        "mean_A": round(mean_a, 6),
        "mean_B": round(mean_b, 6),
        "B_minus_A": round(mean_d, 6),
        "ci95": [round(lo_d, 6), round(hi_d, 6)],
        "excludes_zero": bool(lo_d > 0.0 or hi_d < 0.0),
        "unf_rate_B_minus_A": round(mean_ud, 6),
        "unf_rate_ci95": [round(lo_u, 6), round(hi_u, 6)],
        "unf_excludes_zero": bool(lo_u > 0.0 or hi_u < 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3b-dir", required=True)
    ap.add_argument("--run-tag", default="p3b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-anchor-gate", action="store_true",
                    help="Exit 0 if G_anchor (k4-bf16core - k325-native 128k) 95% CI > 0, else exit 10.")
    args = ap.parse_args()

    s3b = Path(args.stage3b_dir)
    tasks_root = s3b / "tasks"
    arm_names = ["k325-native", "k4-bf16core", "k345-b1b2", "k325-iso-hot"]
    loaded = {a: load_arm_tasks(tasks_root, a, args.run_tag) for a in arm_names}
    present = [a for a in arm_names if loaded[a]]

    levels = {}
    for a in present:
        levels[a] = {
            "4k": summarize_arm(loaded[a], "4k"),
            "128k": summarize_arm(loaded[a], "128k"),
        }

    contrasts = {"4k": {}, "128k": {}}
    for length in ("4k", "128k"):
        if "k325-native" in present:
            for a in present:
                if a == "k325-native":
                    continue
                contrasts[length][f"{a}_vs_k325-native"] = paired_item_boot(loaded["k325-native"], loaded[a], length)
        if "k4-bf16core" in present:
            for a in ("k345-b1b2", "k325-iso-hot"):
                if a in present:
                    contrasts[length][f"{a}_vs_k4-bf16core"] = paired_item_boot(loaded["k4-bf16core"], loaded[a], length)

    g_anchor = contrasts["128k"].get("k4-bf16core_vs_k325-native")
    anchor_detected = bool(g_anchor and g_anchor.get("ci95", [0.0, 0.0])[0] > 0.0)
    anchor_point_ge_2_5pp = bool(g_anchor and g_anchor.get("B_minus_A", 0.0) >= 0.025)

    l3_3 = {}
    if g_anchor and g_anchor.get("B_minus_A", 0.0) > 0:
        g_val = g_anchor["B_minus_A"]
        k4_unf_drop = max(0, levels["k325-native"]["128k"]["unfinished"] - levels["k4-bf16core"]["128k"]["unfinished"])
        for cand in ("k345-b1b2", "k325-iso-hot"):
            if cand not in present:
                continue
            c128 = contrasts["128k"][f"{cand}_vs_k325-native"]
            c4 = contrasts["4k"][f"{cand}_vs_k325-native"]
            acc_share = c128["B_minus_A"] / g_val
            cand_unf_drop = levels["k325-native"]["128k"]["unfinished"] - levels[cand]["128k"]["unfinished"]
            unf_share = (cand_unf_drop / k4_unf_drop) if k4_unf_drop > 0 else None
            no_4k_reg = bool(c4["B_minus_A"] >= -0.015 or not c4["excludes_zero"])
            fin_ok = bool(
                levels[cand]["128k"]["fin_only_score"] is not None
                and levels["k325-native"]["128k"]["fin_only_score"] is not None
                and levels[cand]["128k"]["fin_only_score"] >= levels["k325-native"]["128k"]["fin_only_score"] - 0.01
            )
            l3_3[cand] = {
                "acc_recovery_share_of_G_anchor": round(acc_share, 4),
                "unf_drop_vs_native": cand_unf_drop,
                "unf_recovery_share_of_K4": round(unf_share, 4) if unf_share is not None else None,
                "no_4k_regression": no_4k_reg,
                "fin_only_non_inferior": fin_ok,
                "L3_3_pass": bool(anchor_detected and no_4k_reg and fin_ok and (acc_share >= 0.50 or (unf_share is not None and unf_share >= 0.50))),
            }

    summary = {
        "schema": "kintsugi-phase3b-summary-v1",
        "arms_present": present,
        "levels": levels,
        "contrasts": contrasts,
        "anchor_gate": {
            "G_anchor_128k": g_anchor,
            "anchor_gap_detected_ci_gt_0": anchor_detected,
            "anchor_point_ge_2_5pp": anchor_point_ge_2_5pp,
        },
        "L3_3_verdicts": l3_3,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== STAGE 3B ARM LEVELS (4K & 128K, 60 items x 4 passes) ===")
    for a in present:
        l4, l128 = levels[a]["4k"], levels[a]["128k"]
        print(
            f"  {a:14s} | 4K: {100*l4['mean_score']:5.2f}% (unf={l4['unfinished']}/{l4['n']}) | "
            f"128K: {100*l128['mean_score']:5.2f}% (fin={100*(l128['fin_only_score'] or 0):5.2f}%, unf={l128['unfinished']}/{l128['n']}, fc={l128['finish_classes']})"
        )
    print("\n=== STAGE 3B 128K PAIRED CONTRASTS (60-item cluster bootstrap) ===")
    for k, v in contrasts["128k"].items():
        print(
            f"  {k:32s}: {100*v['B_minus_A']:+6.2f} pp CI=[{100*v['ci95'][0]:+6.2f}, {100*v['ci95'][1]:+6.2f}] pp "
            f"sig={v['excludes_zero']} | dUnf={100*v['unf_rate_B_minus_A']:+5.2f} pp"
        )
    print("\n=== ANCHOR GATE & L3.3 ===")
    print(json.dumps({"anchor_gate": summary["anchor_gate"], "L3_3_verdicts": l3_3}, indent=2))

    if args.check_anchor_gate and not anchor_detected:
        sys.exit(10)


if __name__ == "__main__":
    main()
