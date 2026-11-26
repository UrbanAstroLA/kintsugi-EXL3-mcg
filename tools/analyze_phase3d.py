#!/usr/bin/env python3
"""
Analyzer for Phase 3 Stage 3D (Step 3A): 7-band single-flip scan (`B0..B6 -> 100% K4`)
plus matched stitched control `k325-rebuilt` on the frozen 72 128K dump prefixes.

Evaluates:
  1. Library calibration offset `k325-rebuilt - k325-native` on `M_class(exit_10)`.
  2. Matched-library single-band effects `b{i}-up-k4 - k325-rebuilt` (item-cluster
     bootstrap, 10,000 resamples, seed 20261123) across `enter`, `exit_5`, `exit_10`
     (primary), `exit_20`, and across `join` / `ledger` and all 4 author strata.
  3. Pre-registered additivity checks:
     (a) `(B1 + B2)` vs measured `(k345-b1b2 - k325-rebuilt)` at each cut.
     (b) `sum(B0..B6)` vs measured `(k4-bf16core - k325-rebuilt)` at each cut.
  4. Spearman rank correlation vs Stage 2A `<= 14.4K` P-B KL band ranking.
  5. Pre-registered Step 3B candidate selection check (`<= 3` bands, `>= 80%` summed
     share at `exit_10`, `>= 60%` on both `join` and `ledger`, distinct from `B1-B2`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BANDS = ["b0", "b1", "b2", "b3", "b4", "b5", "b6"]
BAND_LAYERS = {
    "b0": "L03-L08",
    "b1": "L09-L14",
    "b2": "L15-L20",
    "b3": "L21-L26",
    "b4": "L27-L32",
    "b5": "L33-L38",
    "b6": "L39-L44",
}
# Stage 2A upward -Delta KL_PB (gen) from PHASE2A-SUMMARY.md [E2.1]
STAGE2A_KL_UP = {
    "b0": 0.001688,
    "b1": 0.002583,
    "b2": 0.004713,
    "b3": 0.002217,
    "b4": 0.001544,
    "b5": 0.001093,
    "b6": 0.001150,
}


def load_arm(path: Path) -> Dict[str, dict]:
    doc = json.loads(path.read_text())
    return {r["prefix_id"]: r for r in doc["items"] if r.get("kind") == "dump"}


def boot_diff(
    rows_a: Dict[str, dict],
    rows_b: Dict[str, dict],
    cut: str,
    metric: str,
    pids: List[str],
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    by_item: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for pid in pids:
        ra, rb = rows_a.get(pid), rows_b.get(pid)
        if not (ra and rb):
            continue
        va = (ra["cuts"].get(cut) or {}).get(metric)
        vb = (rb["cuts"].get(cut) or {}).get(metric)
        if va is None or vb is None:
            continue
        by_item[ra["item_id"]].append((float(va), float(vb)))
    items = sorted(by_item)
    if not items:
        return {"n_prefixes": 0, "n_items": 0}
    all_a = [a for i in items for a, _ in by_item[i]]
    all_b = [b for i in items for _, b in by_item[i]]
    n_pref = len(all_a)
    diff = (sum(all_b) - sum(all_a)) / n_pref
    item_sums = [(len(by_item[i]), sum(b - a for a, b in by_item[i])) for i in items]
    n_items = len(items)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sn = sc = 0.0
        for _ in range(n_items):
            cnt, sd = item_sums[rng.randrange(n_items)]
            sc += cnt
            sn += sd
        boots.append(sn / sc)
    boots.sort()
    mu = sum(boots) / n_boot
    se = math.sqrt(sum((x - mu) ** 2 for x in boots) / (n_boot - 1))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    return {
        "n_prefixes": n_pref,
        "n_items": n_items,
        "mean_A": round(sum(all_a) / n_pref, 6),
        "mean_B": round(sum(all_b) / n_pref, 6),
        "B_minus_A": round(diff, 6),
        "boot_se": round(se, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "less_sticky_detected": bool(hi < 0.0),
    }


def spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    def ranks(vs: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = float(pos + 1)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3c-margins-dir", required=True)
    ap.add_argument("--stage3d-margins-dir", required=True)
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m3c = Path(args.stage3c_margins_dir)
    m3d = Path(args.stage3d_margins_dir)

    native = load_arm(m3c / f"k325-native-{args.tag}.loop_margins.json")
    k345 = load_arm(m3c / f"k345-b1b2-{args.tag}.loop_margins.json")
    k4 = load_arm(m3c / f"k4-bf16core-{args.tag}.loop_margins.json")
    rebuilt = load_arm(m3d / f"k325-rebuilt-{args.tag}.loop_margins.json")
    band_arms = {b: load_arm(m3d / f"{b}-up-k4-{args.tag}.loop_margins.json") for b in BANDS}

    pids = sorted(rebuilt.keys())
    authors = sorted({rebuilt[p]["author"] for p in pids})
    families = sorted({rebuilt[p]["family"] for p in pids})

    metric = "m_class"
    lib_offset = {cut: boot_diff(native, rebuilt, cut, metric, pids) for cut in ("enter", "exit_5", "exit_10", "exit_20")}
    k4_vs_rebuilt = {cut: boot_diff(rebuilt, k4, cut, metric, pids) for cut in ("enter", "exit_5", "exit_10", "exit_20")}
    k345_vs_rebuilt = {cut: boot_diff(rebuilt, k345, cut, metric, pids) for cut in ("enter", "exit_5", "exit_10", "exit_20")}
    k4_vs_rebuilt_fam = {
        fam: boot_diff(rebuilt, k4, "exit_10", metric, [p for p in pids if rebuilt[p]["family"] == fam])
        for fam in families
    }

    band_results: Dict[str, dict] = {}
    for b in BANDS:
        by_cut = {cut: boot_diff(rebuilt, band_arms[b], cut, metric, pids) for cut in ("enter", "exit_5", "exit_10", "exit_20")}
        by_fam = {
            fam: boot_diff(rebuilt, band_arms[b], "exit_10", metric, [p for p in pids if rebuilt[p]["family"] == fam])
            for fam in families
        }
        by_auth = {
            auth: boot_diff(rebuilt, band_arms[b], "exit_10", metric, [p for p in pids if rebuilt[p]["author"] == auth])
            for auth in authors
        }
        d_k4 = k4_vs_rebuilt["exit_10"]["B_minus_A"]
        share_10 = round(by_cut["exit_10"]["B_minus_A"] / d_k4, 4) if abs(d_k4) > 1e-12 else None
        share_fam = {
            fam: round(by_fam[fam]["B_minus_A"] / k4_vs_rebuilt_fam[fam]["B_minus_A"], 4)
            if abs(k4_vs_rebuilt_fam[fam]["B_minus_A"]) > 1e-12
            else None
            for fam in families
        }
        band_results[b] = {
            "layers": BAND_LAYERS[b],
            "by_cut": by_cut,
            "exit10_share_of_k4": share_10,
            "by_family_exit10": by_fam,
            "exit10_family_shares": share_fam,
            "by_author_exit10": by_auth,
        }

    # Rankings (most negative B_minus_A = most softening = #1)
    overall_rank = sorted(BANDS, key=lambda b: band_results[b]["by_cut"]["exit_10"]["B_minus_A"])
    author_top_band = {
        auth: min(BANDS, key=lambda b: band_results[b]["by_author_exit10"][auth]["B_minus_A"])
        for auth in authors
    }
    author_rankings = {
        auth: sorted(BANDS, key=lambda b: band_results[b]["by_author_exit10"][auth]["B_minus_A"])
        for auth in authors
    }

    # Pre-registered additivity checks
    additivity = {}
    for cut in ("enter", "exit_5", "exit_10", "exit_20"):
        s7 = sum(band_results[b]["by_cut"][cut]["B_minus_A"] for b in BANDS)
        dk4 = k4_vs_rebuilt[cut]["B_minus_A"]
        ratio_7 = round(s7 / dk4, 4) if abs(dk4) > 1e-12 else None
        b12_sum = band_results["b1"]["by_cut"][cut]["B_minus_A"] + band_results["b2"]["by_cut"][cut]["B_minus_A"]
        d345 = k345_vs_rebuilt[cut]["B_minus_A"]
        ratio_12 = round(b12_sum / d345, 4) if abs(d345) > 1e-12 else None
        if ratio_7 is None:
            verdict = "undefined"
        elif 0.75 <= ratio_7 <= 1.25:
            verdict = "additive (within +-25% of K4)"
        elif ratio_7 < 0.50:
            verdict = "non-additive / distributed (<50% of K4 -> complement arm fallback)"
        else:
            verdict = "partial / super-additive"
        additivity[cut] = {
            "sum_7_bands": round(s7, 6),
            "measured_k4_minus_rebuilt": dk4,
            "sum7_over_k4": ratio_7,
            "reading": verdict,
            "b1_plus_b2_sum": round(b12_sum, 6),
            "measured_k345_minus_rebuilt": d345,
            "b12_sum_over_k345": ratio_12,
        }

    # Spearman rho vs Stage 2A <=14.4K P-B KL upward recovery
    rho_vs_kl = spearman(
        [-band_results[b]["by_cut"]["exit_10"]["B_minus_A"] for b in BANDS],
        [STAGE2A_KL_UP[b] for b in BANDS],
    )

    # Pre-registered Step 3B candidate search (<=3 bands, distinct from {b1, b2})
    add_ratio_10 = additivity["exit_10"]["sum7_over_k4"]
    qualifying = []
    if add_ratio_10 is not None and add_ratio_10 >= 0.50:
        for k in (1, 2, 3):
            for combo in itertools.combinations(BANDS, k):
                if set(combo) == {"b1", "b2"}:
                    continue
                sh_all = sum(band_results[b]["exit10_share_of_k4"] or 0.0 for b in combo)
                sh_join = sum(band_results[b]["exit10_family_shares"]["join"] or 0.0 for b in combo)
                sh_led = sum(band_results[b]["exit10_family_shares"]["ledger"] or 0.0 for b in combo)
                sh_ent = sum(
                    band_results[b]["by_cut"]["enter"]["B_minus_A"] / k4_vs_rebuilt["enter"]["B_minus_A"]
                    for b in combo
                )
                sh_e20 = sum(
                    band_results[b]["by_cut"]["exit_20"]["B_minus_A"] / k4_vs_rebuilt["exit_20"]["B_minus_A"]
                    for b in combo
                )
                if sh_all >= 0.80 and sh_join >= 0.60 and sh_led >= 0.60:
                    qualifying.append(
                        {
                            "bands": list(combo),
                            "n_bands": k,
                            "extra_gib": round(k * 3.7969, 2),
                            "sum_share_exit10": round(sh_all, 4),
                            "sum_share_join": round(sh_join, 4),
                            "sum_share_ledger": round(sh_led, 4),
                            "sum_share_enter": round(sh_ent, 4),
                            "sum_share_exit20": round(sh_e20, 4),
                        }
                    )
        qualifying.sort(key=lambda c: (c["n_bands"], -c["sum_share_exit10"]))

    out = {
        "schema": "kintsugi-phase3d-7band-scan-v1",
        "primary_metric": metric,
        "primary_cut": "exit_10",
        "n_prefixes": len(pids),
        "library_offset_rebuilt_minus_native": lib_offset,
        "anchors_vs_rebuilt": {"k4_bf16core": k4_vs_rebuilt, "k345_b1b2": k345_vs_rebuilt, "k4_by_family_exit10": k4_vs_rebuilt_fam},
        "band_ranking_exit10": overall_rank,
        "author_top_band_exit10": author_top_band,
        "author_rankings_exit10": author_rankings,
        "spearman_rho_vs_stage2a_pb_kl": rho_vs_kl,
        "additivity": additivity,
        "step3b_qualifying_candidates": qualifying[:10],
        "bands": band_results,
    }

    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2) + "\n")

    print("=== STAGE 3D 7-BAND M_class(exit_10) SCAN (vs matched k325-rebuilt, n=72 dumps) ===")
    print(f"  Library offset (rebuilt - native): {lib_offset['exit_10']['B_minus_A']:+.4f} CI {lib_offset['exit_10']['ci95']}")
    print(f"  K4 - rebuilt (exit_10):            {k4_vs_rebuilt['exit_10']['B_minus_A']:+.4f} CI {k4_vs_rebuilt['exit_10']['ci95']}")
    print(f"  K345 - rebuilt (exit_10):          {k345_vs_rebuilt['exit_10']['B_minus_A']:+.4f} CI {k345_vs_rebuilt['exit_10']['ci95']}")
    for b in overall_rank:
        br = band_results[b]
        e10 = br["by_cut"]["exit_10"]
        print(
            f"  {b.upper()} ({br['layers']}): {e10['B_minus_A']:+.4f} CI {e10['ci95']} "
            f"| share={100*(br['exit10_share_of_k4'] or 0):5.1f}% "
            f"| join={100*(br['exit10_family_shares']['join'] or 0):5.1f}% "
            f"| ledger={100*(br['exit10_family_shares']['ledger'] or 0):5.1f}%"
        )
    print(f"\nAdditivity (exit_10): {json.dumps(additivity['exit_10'])}")
    print(f"Spearman rho vs Stage 2A P-B KL: {rho_vs_kl}")
    print(f"Author top bands: {author_top_band}")
    print(f"Top qualifying Step 3B candidates: {qualifying[:3]}")


if __name__ == "__main__":
    main()
