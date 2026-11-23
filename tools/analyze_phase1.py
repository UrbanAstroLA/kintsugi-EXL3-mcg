#!/usr/bin/env python3
"""
Phase 1 analyzer: evaluates pre-registered Gate P1 lines (L1.1 determinism,
L1.2 two-component probe validity, L1.3 library calibration offset) from
frozen margin receipts and P-A/P-B score summaries.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def paired_bootstrap_mean(diffs: List[float], draws: int = 10000, seed: int = 20261123) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    boots = []
    for _ in range(draws):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * draws)]
    hi = boots[int(0.975 * draws)]
    return {
        "n": n,
        "mean": round(mean, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
    }


def load_margins(path: Path) -> Dict[str, dict]:
    doc = json.loads(path.read_text())
    return {it["item_id"]: it for it in doc["items"]}


def compare_margins(b_map: Dict[str, dict], a_map: Dict[str, dict]) -> dict:
    ids = sorted(b_map.keys())
    assert ids == sorted(a_map.keys())
    metrics = {
        "comp_a_min": lambda x: x["comp_a"]["min_margin"],
        "comp_a_mean": lambda x: x["comp_a"]["mean_margin"],
        "comp_b_close": lambda x: x["comp_b"]["close_margin"],
        "comp_b_tail_mean": lambda x: x["comp_b"]["tail_mean_margin"],
        "comp_b_combined_min": lambda x: x["comp_b"]["combined_min_margin"],
    }
    out = {}
    for name, fn in metrics.items():
        diffs = [fn(b_map[i]) - fn(a_map[i]) for i in ids]
        out[name] = paired_bootstrap_mean(diffs)
    # Also compute max abs spread and row-hash equality
    max_a = max(abs(b_map[i]["comp_a"]["min_margin"] - a_map[i]["comp_a"]["min_margin"]) for i in ids)
    max_b = max(abs(b_map[i]["comp_b"]["close_margin"] - a_map[i]["comp_b"]["close_margin"]) for i in ids)
    hash_eq = sum(1 for i in ids if b_map[i]["comp_b"]["close_hash"] == a_map[i]["comp_b"]["close_hash"])
    out["max_abs_diff_a_min"] = round(max_a, 8)
    out["max_abs_diff_b_close"] = round(max_b, 8)
    out["b_close_hash_equal"] = f"{hash_eq}/{len(ids)}"
    return out


def load_pb_kl(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    # score_pb.py summary schema
    for k in ("gen", "generated", "overall"):
        if k in d and isinstance(d[k], dict) and "kl" in d[k]:
            return float(d[k]["kl"])
    if "kl" in d:
        return float(d["kl"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    mdir = rd / "margins"
    pb_dir = rd / "bf16ref/scores/pb"
    pa_dir = rd / "bf16ref/scores/pa"

    arms = {}
    for arm_tag in ("k325-native-r0", "k325-native-r1", "k325-rebuilt-r0", "k4-bf16core-r0", "k3-bf16core-r0"):
        p = mdir / f"{arm_tag}.margins.json"
        if p.exists():
            arms[arm_tag] = json.loads(p.read_text())

    by_item = {k: {it["item_id"]: it for it in v["items"]} for k, v in arms.items()}

    # L1.1 Determinism: k325-native-r1 vs k325-native-r0
    l11_margins = compare_margins(by_item["k325-native-r1"], by_item["k325-native-r0"])
    max_spread = max(l11_margins["max_abs_diff_a_min"], l11_margins["max_abs_diff_b_close"])

    # L1.2 Probe validity: k4-bf16core-r0 vs k325-rebuilt-r0 (and vs k3-bf16core-r0)
    k4_vs_reb = compare_margins(by_item["k4-bf16core-r0"], by_item["k325-rebuilt-r0"])
    k4_vs_k3 = compare_margins(by_item["k4-bf16core-r0"], by_item["k3-bf16core-r0"])
    reb_vs_nat = compare_margins(by_item["k325-rebuilt-r0"], by_item["k325-native-r0"])

    # Load P-B / P-A pair JSONs if present
    def read_json_if(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    pb_pairs = {
        "r1_vs_r0": read_json_if(pb_dir / "pair-k325-native-r1-vs-k325-native-r0.json"),
        "rebuilt_vs_native": read_json_if(pb_dir / "pair-k325-rebuilt-r0-vs-k325-native-r0.json"),
        "k4_vs_rebuilt": read_json_if(pb_dir / "pair-k4-bf16core-r0-vs-k325-rebuilt-r0.json"),
        "k3_vs_rebuilt": read_json_if(pb_dir / "pair-k3-bf16core-r0-vs-k325-rebuilt-r0.json"),
        "k4_vs_k3": read_json_if(pb_dir / "pair-k4-bf16core-r0-vs-k3-bf16core-r0.json"),
    }
    pa_pairs = {
        "rebuilt_vs_native": read_json_if(pa_dir / "pair-k325-rebuilt-vs-k325-native.json"),
        "k4_vs_rebuilt": read_json_if(pa_dir / "pair-k4-bf16core-vs-k325-rebuilt.json"),
        "k3_vs_rebuilt": read_json_if(pa_dir / "pair-k3-bf16core-vs-k325-rebuilt.json"),
        "k4_vs_k3": read_json_if(pa_dir / "pair-k4-bf16core-vs-k3-bf16core.json"),
    }

    reb_doc = arms["k325-rebuilt-r0"]
    l11_pass = max_spread < 0.05
    l12_a_pass = (
        (k4_vs_reb["comp_a_min"]["ci95"][0] > 0.0 or k4_vs_reb["comp_a_mean"]["ci95"][0] > 0.0)
        and reb_doc["comp_a_fragile_lt1"] >= 5
    )
    l12_b_pass = (
        (k4_vs_reb["comp_b_close"]["ci95"][0] > 0.0 or k4_vs_reb["comp_b_tail_mean"]["ci95"][0] > 0.0)
        and reb_doc["comp_b_close_fragile_lt1"] >= 5
    )

    out = {
        "schema": "kintsugi-phase1-summary-v1",
        "arm_levels": {
            k: {
                "n_items": v["n_items"],
                "wall_s": v["wall_s"],
                "all_finite": v["all_finite"],
                "comp_a_min_mean": round(v["comp_a_min_mean"], 4),
                "comp_a_mean_mean": round(v["comp_a_mean_mean"], 4),
                "comp_a_fragile_lt1": v["comp_a_fragile_lt1"],
                "comp_a_neg_lt0": v["comp_a_neg_lt0"],
                "comp_b_close_mean": round(v["comp_b_close_mean"], 4),
                "comp_b_tail_mean": round(v["comp_b_tail_mean"], 4),
                "comp_b_close_fragile_lt1": v["comp_b_close_fragile_lt1"],
                "comp_b_close_neg_lt0": v["comp_b_close_neg_lt0"],
            }
            for k, v in arms.items()
        },
        "contrasts_margins": {
            "L1_1_r1_vs_r0": l11_margins,
            "L1_2_k4_vs_rebuilt": k4_vs_reb,
            "k4_vs_k3": k4_vs_k3,
            "L1_3_rebuilt_vs_native": reb_vs_nat,
        },
        "pb_pairs": pb_pairs,
        "pa_pairs": pa_pairs,
        "gates": {
            "L1_1_margin_spread_lt_0_05": l11_pass,
            "L1_1_max_spread": max_spread,
            "L1_2_comp_a_pass": l12_a_pass,
            "L1_2_comp_b_pass": l12_b_pass,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["arm_levels"], indent=2))
    print("\n--- Gates ---")
    print(json.dumps(out["gates"], indent=2))
    print("\n--- K4 vs K3.25-rebuilt margins ---")
    print(json.dumps(k4_vs_reb, indent=2))
    print("\n--- K3.25-rebuilt vs K3.25-native margins ---")
    print(json.dumps(reb_vs_nat, indent=2))


if __name__ == "__main__":
    main()
