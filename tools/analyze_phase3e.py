#!/usr/bin/env python3
"""
Analyzer for Phase 3 Stage 3E: Matched-Library Control (`k325-rebuilt`) 8-Pass
127K Task Confirmation (`L3E.1`).

Compares `k325-rebuilt` (`reps 0-7`, 480 128K replies + 240 4K replies) paired
1-to-1 by `(item, length, rep, seed, nonce)` against `k325-native`, `k4-bf16core`,
and `k345-b1b2`:
  - Frozen reading bands on `k325-rebuilt` 128K unfinished count (`<=20`, `21-29`, `>=30`).
  - Paired 60-item cluster bootstrap (10,000 resamples, seed 20261123) on 128K
    overall accuracy, unfinished rate, both-finished accuracy, and 4K accuracy.
  - Stage-split check (`reps 0-3` mixed vs `reps 4-7` 128K-only) + Fisher exact p.
  - Deterministic tail classifier, `join` entering-vs-leaving (`>=15` run) counts,
    family breakdown, and finished completion-token quantiles.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAW_REC_RE = re.compile(r"^(courier C-|parcel P-|hub H-|\d{5}\s+[A-Z0-9-]{3})", re.I)
REC_RE = re.compile(
    r"(courier|parcel|hub|shipment|packet|depot|route|^\s*\d{4,5}\s+[A-Z0-9-]{3,}|"
    r"^\s*[-*\d]+[\.\):]\s|\b[A-Z]{2,}-\d{2,}\b|\bTXN[-_]?\d+|\bACC[-_]?\d+)",
    re.I,
)


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


def classify_tail(reasoning: str) -> str:
    tail = (reasoning or "")[-20000:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return "empty"
    _, top_cnt = Counter(lines).most_common(1)[0]
    rec_frac = sum(1 for ln in lines if REC_RE.search(ln)) / len(lines)
    if rec_frac >= 0.70:
        return "record_list"
    if top_cnt >= 10:
        return "line_repeat"
    return "computation"


def has_run15(reasoning: str) -> bool:
    run = 0
    for ln in reasoning.splitlines():
        if RAW_REC_RE.match(ln.strip()):
            run += 1
            if run >= 15:
                return True
        elif ln.strip():
            run = 0
    return False


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for 2x2 table [[a, b], [c, d]]."""
    r1, r2 = a + b, c + d
    c1 = a + c
    n = r1 + r2
    def hyper(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p_obs = hyper(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p = sum(hyper(x) for x in range(lo, hi + 1) if hyper(x) <= p_obs + 1e-12)
    return round(min(1.0, p), 6)


def quantiles(vals: List[float]) -> Optional[dict]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    def q(p: float) -> float:
        return round(s[min(n - 1, int(p * (n - 1)))], 1)
    return {"n": n, "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def summarize_arm(rows: Dict[Tuple[str, str, int], dict], length: str) -> dict:
    sub = [v for k, v in rows.items() if k[0] == length]
    if not sub:
        return {"n": 0}
    fin = [r for r in sub if r.get("finish_class") == "stop"]
    unf = [r for r in sub if r.get("finish_class") != "stop"]
    tails: Counter = Counter()
    in_think = 0
    join_tot = join_ent = join_left = 0
    for r in sub:
        tf = Path(r["_rep_dir"]) / "texts" / f"{r['unit']}.{r['session']}.json.gz"
        reas = ""
        if tf.exists():
            reas = json.loads(gzip.open(tf, "rt").read())["turns"][0].get("reasoning", "") or ""
        if r.get("finish_class") != "stop":
            if r.get("content_chars", 0) == 0:
                in_think += 1
            tails[classify_tail(reas)] += 1
        if length == "128k" and r["family"] == "join":
            join_tot += 1
            if has_run15(reas):
                join_ent += 1
                if r.get("finish_class") == "stop":
                    join_left += 1

    by_fam = {}
    for fam in ("join", "ledger", "trace"):
        fs = [r for r in sub if r["family"] == fam]
        if not fs:
            continue
        ff = [r for r in fs if r.get("finish_class") == "stop"]
        by_fam[fam] = {
            "n": len(fs),
            "mean_score": round(sum(item_score(r) for r in fs) / len(fs), 6),
            "fin_only_score": round(sum(item_score(r) for r in ff) / len(ff), 6) if ff else None,
            "unfinished": len(fs) - len(ff),
            "unfinished_rate": round((len(fs) - len(ff)) / len(fs), 6),
        }

    per_rep_unf = {
        f"r{rep}": sum(1 for r in unf if int(r["rep"]) == rep)
        for rep in sorted({int(r["rep"]) for r in sub})
    }
    return {
        "n": len(sub),
        "mean_score": round(sum(item_score(r) for r in sub) / len(sub), 6),
        "fin_only_n": len(fin),
        "fin_only_score": round(sum(item_score(r) for r in fin) / len(fin), 6) if fin else None,
        "unfinished": len(unf),
        "unfinished_rate": round(len(unf) / len(sub), 6),
        "ended_in_thinking": in_think,
        "per_rep_unfinished": per_rep_unf,
        "unfinished_tail_classes": dict(sorted(tails.items())),
        "by_family": by_fam,
        "join_enter_leave_ge15": {
            "join_replies": join_tot,
            "entered_ge15": join_ent,
            "enter_rate": round(join_ent / join_tot, 4) if join_tot else None,
            "left_and_finished": join_left,
            "leave_rate": round(join_left / join_ent, 4) if join_ent else None,
        } if length == "128k" else None,
        "finished_completion_tokens_q": quantiles([float(r["completion_tokens"]) for r in fin if r.get("completion_tokens") is not None]),
    }


def paired_boot(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    length: str,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    common = sorted(k for k in rows_a if k[0] == length and k in rows_b)
    by_item: Dict[str, List[Tuple[float, float, int, int]]] = defaultdict(list)
    bf_by_item: Dict[str, List[float]] = defaultdict(list)
    for k in common:
        ra, rb = rows_a[k], rows_b[k]
        ua = 0 if ra.get("finish_class") == "stop" else 1
        ub = 0 if rb.get("finish_class") == "stop" else 1
        sa, sb = item_score(ra), item_score(rb)
        by_item[k[1]].append((sa, sb, ua, ub))
        if ua == 0 and ub == 0:
            bf_by_item[k[1]].append(sb - sa)
    items = sorted(by_item)
    n = len(items)
    if n == 0:
        return {"n_items": 0}
    sa_i = [sum(a for a, _, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sb_i = [sum(b for _, b, _, _ in by_item[i]) / len(by_item[i]) for i in items]
    sd = [b - a for a, b in zip(sa_i, sb_i)]
    ud = [sum(ub - ua for _, _, ua, ub in by_item[i]) / len(by_item[i]) for i in items]
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
    bd.sort()
    bu.sort()
    sed = math.sqrt(sum((x - sum(bd) / n_boot) ** 2 for x in bd) / (n_boot - 1))
    seu = math.sqrt(sum((x - sum(bu) / n_boot) ** 2 for x in bu) / (n_boot - 1))

    # Both-finished paired contrast
    bf_items = sorted(bf_by_item)
    bf_ds = [sum(bf_by_item[i]) / len(bf_by_item[i]) for i in bf_items]
    rng2 = random.Random(seed)
    bf_b = sorted(sum(bf_ds[rng2.randrange(len(bf_ds))] for _ in range(len(bf_ds))) / len(bf_ds) for _ in range(n_boot)) if bf_ds else []

    return {
        "n_items": n,
        "mean_A": round(sum(sa_i) / n, 6),
        "mean_B": round(sum(sb_i) / n, 6),
        "B_minus_A": round(md, 6),
        "boot_se": round(sed, 6),
        "z": round(md / sed, 4) if sed > 0 else None,
        "ci95": [round(bd[int(0.025 * n_boot)], 6), round(bd[int(0.975 * n_boot)], 6)],
        "excludes_zero": bool(bd[int(0.025 * n_boot)] > 0.0 or bd[int(0.975 * n_boot)] < 0.0),
        "unf_rate_B_minus_A": round(mu, 6),
        "unf_boot_se": round(seu, 6),
        "unf_rate_ci95": [round(bu[int(0.025 * n_boot)], 6), round(bu[int(0.975 * n_boot)], 6)],
        "unf_excludes_zero": bool(bu[int(0.025 * n_boot)] > 0.0 or bu[int(0.975 * n_boot)] < 0.0),
        "both_finished_B_minus_A": round(sum(bf_ds) / len(bf_ds), 6) if bf_ds else None,
        "both_finished_ci95": [round(bf_b[int(0.025 * n_boot)], 6), round(bf_b[int(0.975 * n_boot)], 6)] if bf_b else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3b-dir", required=True)
    ap.add_argument("--run-tag", default="p3b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks_root = Path(args.stage3b_dir) / "tasks"
    arms = ["k325-native", "k325-rebuilt", "k345-b1b2", "k4-bf16core"]
    s1 = {a: load_arm(tasks_root, a, args.run_tag, [0, 1, 2, 3]) for a in arms}
    s2 = {a: load_arm(tasks_root, a, args.run_tag, [4, 5, 6, 7]) for a in arms}
    all8 = {a: load_arm(tasks_root, a, args.run_tag, list(range(8))) for a in arms}

    levels = {}
    for a in arms:
        if not all8[a]:
            continue
        l1 = summarize_arm(s1[a], "128k")
        l2 = summarize_arm(s2[a], "128k")
        p_halves = (
            fisher_exact_two_sided(l1["unfinished"], l1["n"] - l1["unfinished"], l2["unfinished"], l2["n"] - l2["unfinished"])
            if l1["n"] and l2["n"]
            else None
        )
        levels[a] = {
            "4k_reps0_3": summarize_arm(s1[a], "4k"),
            "128k_reps0_3": l1,
            "128k_reps4_7": l2,
            "128k_all8": summarize_arm(all8[a], "128k"),
            "halves_unfinished_fisher_p": p_halves,
        }

    reb_unf = levels["k325-rebuilt"]["128k_all8"]["unfinished"]
    if reb_unf >= 30:
        reading = "k325-rebuilt ~= k325-native (>=30/480): K4 task gain is K4-specific; M_class offset was library artifact"
    elif reb_unf <= 20:
        reading = "k325-rebuilt ~= k4-bf16core (<=20/480): native K3.25-v1 runaway rate is checkpoint-specific; k345-b1b2 regressed vs base"
    else:
        reading = "unresolved / intermediate (21-29/480)"

    contrasts_128k = {
        "k4-bf16core_vs_k325-rebuilt": paired_boot(all8["k325-rebuilt"], all8["k4-bf16core"], "128k"),
        "k325-rebuilt_vs_k325-native": paired_boot(all8["k325-native"], all8["k325-rebuilt"], "128k"),
        "k345-b1b2_vs_k325-rebuilt": paired_boot(all8["k325-rebuilt"], all8["k345-b1b2"], "128k"),
        "k4-bf16core_vs_k325-native": paired_boot(all8["k325-native"], all8["k4-bf16core"], "128k"),
    }
    contrasts_4k = {
        "k325-rebuilt_vs_k325-native": paired_boot(s1["k325-native"], s1["k325-rebuilt"], "4k"),
        "k4-bf16core_vs_k325-rebuilt": paired_boot(s1["k325-rebuilt"], s1["k4-bf16core"], "4k"),
    }

    out = {
        "schema": "kintsugi-phase3e-rebuilt-control-v1",
        "frozen_L3E_1_reading": reading,
        "levels": levels,
        "contrasts_128k_8pass": contrasts_128k,
        "contrasts_4k_4pass": contrasts_4k,
    }
    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2) + "\n")

    print("=== STAGE 3E MATCHED-LIBRARY CONTROL (8-PASS 128K, n=480/arm) ===")
    print(f"FROZEN L3E.1 READING: {reading}")
    for a in arms:
        if a not in levels:
            continue
        l = levels[a]["128k_all8"]
        l4 = levels[a]["4k_reps0_3"]
        print(
            f"  {a:14s} | 4K: {100*l4['mean_score']:5.2f}% | "
            f"128K: {100*l['mean_score']:5.2f}% (fin={100*l['fin_only_score']:5.2f}%, "
            f"unf={l['unfinished']}/480 [{levels[a]['128k_reps0_3']['unfinished']}+{levels[a]['128k_reps4_7']['unfinished']}, p={levels[a]['halves_unfinished_fisher_p']}], "
            f"join_enter={l['join_enter_leave_ge15']['entered_ge15']}/{l['join_enter_leave_ge15']['join_replies']})"
        )
    print("\n=== PAIRED 8-PASS 128K CONTRASTS ===")
    print(json.dumps(contrasts_128k, indent=2))


if __name__ == "__main__":
    main()
