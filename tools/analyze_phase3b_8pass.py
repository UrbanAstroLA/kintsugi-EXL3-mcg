#!/usr/bin/env python3
"""
Pre-data Amendment 3B.2 analyzer for Phase 3 Stage 3B 8-pass extension
(`k325-native`, `k4-bf16core`, `k345-b1b2`: 128K reps 0-7, 4K reps 0-3).

Implements:
  1. Exact inverse-normal two-stage combination test (`Z = (z1 + z2) / sqrt(2) >= 1.96`,
     one-sided alpha = 0.025) combining Stage 1 (`rep 0-3`) and Stage 2 (`rep 4-7`)
     on both 128K overall accuracy and 128K unfinished-reply reduction.
  2. Descriptive pooled 8-pass paired item-cluster bootstrap 95% CIs (10,000 resamples,
     seed 20261123), plus paired bootstrap 95% CIs for `k345-b1b2` recovery shares.
  3. Frozen deterministic unfinished-tail classifier (`record_list` / `line_repeat` /
     `computation`) on the final 20,000 reasoning chars, plus finished-length quantiles.
  4. Leaves `tools/analyze_phase3b.py` and `stage3b/FROZEN.json` untouched.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REC_RE = re.compile(
    r"(courier|parcel|hub|shipment|packet|depot|route|^\s*\d{4,5}\s+[A-Z0-9-]{3,}|"
    r"^\s*[-*\d]+[\.\):]\s|\b[A-Z]{2,}-\d{2,}\b|\bTXN[-_]?\d+|\bACC[-_]?\d+)",
    re.I,
)


def classify_tail(reasoning: str) -> dict:
    tail = (reasoning or "")[-20000:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return {"tail_class": "empty", "zlib_ratio": 1.0, "rec_frac": 0.0, "max_line_dup": 0}
    raw_b = tail.encode("utf-8", "ignore")
    zratio = len(zlib.compress(raw_b)) / max(1, len(raw_b))
    _, top_cnt = Counter(lines).most_common(1)[0]
    rec_frac = sum(1 for ln in lines if REC_RE.search(ln)) / len(lines)
    if rec_frac >= 0.70:
        cls = "record_list"
    elif top_cnt >= 10:
        cls = "line_repeat"
    else:
        cls = "computation"
    return {
        "tail_class": cls,
        "zlib_ratio": round(zratio, 4),
        "rec_frac": round(rec_frac, 4),
        "max_line_dup": int(top_cnt),
    }


def load_arm(tasks_root: Path, arm: str, run_tag: str, reps: List[int]) -> Dict[Tuple[str, str, int], dict]:
    out: Dict[Tuple[str, str, int], dict] = {}
    base = tasks_root / arm / run_tag
    for rep in reps:
        p = base / f"single-rep{rep}" / "requests.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("final", True) or r.get("status") != "ok":
                continue
            r["_rep_dir"] = str(p.parent)
            out[(str(r["length"]), str(r["item"]), int(r["rep"]))] = r
    return out


def item_score(r: dict) -> float:
    sc = r.get("score") or {}
    if sc.get("score") is not None:
        return float(sc["score"])
    fields = sc.get("fields") or 0
    correct = sc.get("correct") or 0
    return float(correct) / float(fields) if fields else 0.0


def quantiles(vals: List[float]) -> Optional[dict]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    def q(p: float) -> float:
        return round(s[min(n - 1, int(p * (n - 1)))], 1)
    return {"n": n, "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def summarize_arm(rows: Dict[Tuple[str, str, int], dict], length: str) -> dict:
    sub = {k: v for k, v in rows.items() if k[0] == length}
    n = len(sub)
    if n == 0:
        return {"n": 0}
    scores = [item_score(v) for v in sub.values()]
    fc = Counter(v.get("finish_class", "unknown") for v in sub.values())
    fin = [v for v in sub.values() if v.get("finish_class") == "stop"]
    unf = [v for v in sub.values() if v.get("finish_class") != "stop"]

    tail_counts: Counter = Counter()
    ended_in_thinking = 0
    unf_by_fam: Counter = Counter()
    for r in unf:
        unf_by_fam[r["family"]] += 1
        if r.get("content_chars", 0) == 0:
            ended_in_thinking += 1
        tf = Path(r["_rep_dir"]) / "texts" / f"{r['unit']}.{r['session']}.json.gz"
        if tf.exists():
            pay = json.loads(gzip.open(tf, "rt").read())
            reas = pay["turns"][0].get("reasoning", "") or ""
            tail_counts[classify_tail(reas)["tail_class"]] += 1
        else:
            tail_counts["missing_text"] += 1

    fin_toks = [float(r["completion_tokens"]) for r in fin if r.get("completion_tokens") is not None]
    fin_rchars = [float(r.get("reasoning_chars", 0)) for r in fin]
    return {
        "n": n,
        "mean_score": round(sum(scores) / n, 6),
        "fin_only_n": len(fin),
        "fin_only_score": round(sum(item_score(v) for v in fin) / len(fin), 6) if fin else None,
        "unfinished": len(unf),
        "unfinished_rate": round(len(unf) / n, 6),
        "ended_in_thinking": ended_in_thinking,
        "finish_classes": dict(sorted(fc.items())),
        "unfinished_tail_classes": dict(sorted(tail_counts.items())),
        "unfinished_by_family": dict(sorted(unf_by_fam.items())),
        "finished_completion_tokens_q": quantiles(fin_toks),
        "finished_reasoning_chars_q": quantiles(fin_rchars),
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
    items = sorted(by_item)
    sa = [sum(a for a, _, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sb = [sum(b for _, b, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sd = [b - a for a, b in zip(sa, sb)]
    ud = [sum(ub - ua for _, _, ua, ub in by_item[i]) / len(by_item[i]) for i in items]
    return items, sa, sb, sd, ud


def paired_boot(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    length: str,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    items, sa, sb, sd, ud = item_series(rows_a, rows_b, length)
    n = len(items)
    if n == 0:
        return {"n_items": 0}
    md, mu = sum(sd) / n, sum(ud) / n
    rng = random.Random(seed)
    bd, bu = [], []
    for _ in range(n_boot):
        s_d = s_u = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            s_d += sd[j]
            s_u += ud[j]
        bd.append(s_d / n)
        bu.append(s_u / n)
    sed = math.sqrt(sum((x - sum(bd) / n_boot) ** 2 for x in bd) / (n_boot - 1))
    seu = math.sqrt(sum((x - sum(bu) / n_boot) ** 2 for x in bu) / (n_boot - 1))
    bd.sort()
    bu.sort()
    lo_d, hi_d = bd[int(0.025 * n_boot)], bd[int(0.975 * n_boot)]
    lo_u, hi_u = bu[int(0.025 * n_boot)], bu[int(0.975 * n_boot)]
    return {
        "n_items": n,
        "mean_A": round(sum(sa) / n, 6),
        "mean_B": round(sum(sb) / n, 6),
        "B_minus_A": round(md, 6),
        "boot_se": round(sed, 6),
        "z": round(md / sed, 4) if sed > 0 else None,
        "ci95": [round(lo_d, 6), round(hi_d, 6)],
        "excludes_zero": bool(lo_d > 0.0 or hi_d < 0.0),
        "unf_rate_B_minus_A": round(mu, 6),
        "unf_boot_se": round(seu, 6),
        "z_unf_reduction": round((-mu) / seu, 4) if seu > 0 else None,
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
            j = rng.randrange(n)
            sn += num_diffs[j]
            sd += den_diffs[j]
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

    tasks_root = Path(args.stage3b_dir) / "tasks"
    arms = ["k325-native", "k4-bf16core", "k345-b1b2"]
    s1 = {a: load_arm(tasks_root, a, args.run_tag, [0, 1, 2, 3]) for a in arms}
    s2 = {a: load_arm(tasks_root, a, args.run_tag, [4, 5, 6, 7]) for a in arms}
    all8 = {a: load_arm(tasks_root, a, args.run_tag, list(range(8))) for a in arms}
    present = [a for a in arms if all8[a]]

    levels = {
        a: {
            "4k_reps0_3": summarize_arm(s1[a], "4k"),
            "128k_reps0_3": summarize_arm(s1[a], "128k"),
            "128k_reps4_7": summarize_arm(s2[a], "128k"),
            "128k_all8": summarize_arm(all8[a], "128k"),
        }
        for a in present
    }

    # Exact inverse-normal two-stage combination test on anchor (k4-bf16core vs k325-native)
    inv_norm = {}
    if s1["k325-native"] and s1["k4-bf16core"] and s2["k325-native"] and s2["k4-bf16core"]:
        st1 = paired_boot(s1["k325-native"], s1["k4-bf16core"], "128k")
        st2 = paired_boot(s2["k325-native"], s2["k4-bf16core"], "128k")
        z_acc = (st1["z"] + st2["z"]) / math.sqrt(2.0) if (st1["z"] is not None and st2["z"] is not None) else None
        z_unf = (
            (st1["z_unf_reduction"] + st2["z_unf_reduction"]) / math.sqrt(2.0)
            if (st1["z_unf_reduction"] is not None and st2["z_unf_reduction"] is not None)
            else None
        )
        inv_norm = {
            "stage1_reps0_3": st1,
            "stage2_reps4_7": st2,
            "Z_acc_combined": round(z_acc, 4) if z_acc is not None else None,
            "G_anchor_detected_Z_ge_1_96": bool(z_acc is not None and z_acc >= 1.96),
            "Z_unf_combined": round(z_unf, 4) if z_unf is not None else None,
            "k4_unf_detected_Z_ge_1_96": bool(z_unf is not None and z_unf >= 1.96),
        }

    # Pooled 8-pass descriptive contrasts & L3.3 verdict
    pooled_128k = {}
    pooled_4k = {}
    for a in present:
        if a != "k325-native" and all8["k325-native"]:
            pooled_128k[f"{a}_vs_k325-native"] = paired_boot(all8["k325-native"], all8[a], "128k")
            pooled_4k[f"{a}_vs_k325-native"] = paired_boot(s1["k325-native"], s1[a], "4k")
    if all8["k4-bf16core"] and all8["k345-b1b2"]:
        pooled_128k["k345-b1b2_vs_k4-bf16core"] = paired_boot(all8["k4-bf16core"], all8["k345-b1b2"], "128k")

    l3_3 = {}
    if all8["k325-native"] and all8["k4-bf16core"] and all8["k345-b1b2"]:
        items_k4, _, _, sd_k4, ud_k4 = item_series(all8["k325-native"], all8["k4-bf16core"], "128k")
        items_c, _, _, sd_c, ud_c = item_series(all8["k325-native"], all8["k345-b1b2"], "128k")
        assert items_c == items_k4
        acc_sh, acc_sh_ci = share_boot_ci(sd_c, sd_k4)
        unf_sh, unf_sh_ci = share_boot_ci([-u for u in ud_c], [-u for u in ud_k4])
        c4 = pooled_4k["k345-b1b2_vs_k325-native"]
        no_4k_reg = bool(c4["B_minus_A"] >= -0.015 or not c4["excludes_zero"])
        fin_ok = bool(
            levels["k345-b1b2"]["128k_all8"]["fin_only_score"] >= levels["k325-native"]["128k_all8"]["fin_only_score"] - 0.01
        )
        g_det = bool(inv_norm.get("G_anchor_detected_Z_ge_1_96"))
        u_det = bool(inv_norm.get("k4_unf_detected_Z_ge_1_96"))
        acc_pass = bool(acc_sh is not None and acc_sh >= 0.50)
        unf_pass = bool(u_det and unf_sh is not None and unf_sh >= 0.50)
        l3_3["k345-b1b2"] = {
            "G_anchor_8pass_detected": g_det,
            "k4_unf_8pass_detected": u_det,
            "acc_recovery_share": acc_sh,
            "acc_recovery_share_ci95": acc_sh_ci,
            "unf_recovery_share": unf_sh,
            "unf_recovery_share_ci95": unf_sh_ci,
            "no_4k_regression": no_4k_reg,
            "fin_only_non_inferior": fin_ok,
            "L3_3_pass": bool(g_det and no_4k_reg and fin_ok and (acc_pass or unf_pass)),
        }

    out = {
        "schema": "kintsugi-phase3b-8pass-amendment-3b2",
        "arms_present": present,
        "levels": levels,
        "inverse_normal_anchor_test": inv_norm,
        "pooled_8pass_128k_contrasts": pooled_128k,
        "reps0_3_4k_contrasts": pooled_4k,
        "L3_3_verdict": l3_3,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")

    print("=== STAGE 3B 8-PASS LEVELS (128K reps 0-7, 480 replies/arm) ===")
    for a in present:
        l = levels[a]["128k_all8"]
        l4 = levels[a]["4k_reps0_3"]
        print(
            f"  {a:14s} | 4K(4p): {100*l4['mean_score']:5.2f}% | "
            f"128K(8p): {100*l['mean_score']:5.2f}% (fin={100*(l['fin_only_score'] or 0):5.2f}%, "
            f"unf={l['unfinished']}/{l['n']}, in_think={l['ended_in_thinking']}, tails={l['unfinished_tail_classes']})"
        )
    print("\n=== INVERSE-NORMAL ANCHOR TEST & POOLED 8-PASS CONTRASTS ===")
    print(json.dumps({"inverse_normal_anchor_test": inv_norm, "pooled_8pass_128k_contrasts": pooled_128k, "L3_3_verdict": l3_3}, indent=2))


if __name__ == "__main__":
    main()
