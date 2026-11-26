#!/usr/bin/env python3
"""
Analyzer for Phase 3 Stage 3F: Complement Graft (`k377-not-b1b2`: B0 + B3-B6 at K4,
B1-B2 kept at K3.25-rebuilt) 8-Pass `join + ledger` 127K Task Confirmation (`L3F.1`).

Compares `k377-not-b1b2` (`reps 0-7`, 320 `join + ledger` 128K replies) paired 1-to-1
by `(item, length, rep, seed, nonce)` against `k325-rebuilt`, `k4-bf16core`,
`k325-native`, and `k345-b1b2` on the exact same 40 `join + ledger` 128K items:
  - Frozen `L3F.1` reading bands on `k377-not-b1b2` unfinished count (out of 320):
      <= 19/320 : localized outside B1-B2 (`k377-not-b1b2 ~= k4-bf16core`)
      >= 25/320 : diffuse / super-additive (`k377-not-b1b2 ~= k325-rebuilt`)
      20-24/320 : intermediate / partial recovery
  - Paired 40-item cluster bootstrap (10,000 resamples, seed 20261123) on `join + ledger`
    128K overall accuracy, unfinished rate, and both-finished accuracy.
  - Halves consistency (`reps 0-3` vs `reps 4-7`, Fisher exact p), family breakdown
    (`join` vs `ledger`), deterministic tail classes, `join` entering-vs-leaving
    (`>= 15` run), and finished completion-token quantiles.
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

FAMILIES_3F = ("join", "ledger")

RAW_REC_RE = re.compile(r"^(courier C-|parcel P-|hub H-|\d{5}\s+[A-Z0-9-]{3})", re.I)
REC_RE = re.compile(
    r"(courier|parcel|hub|shipment|packet|depot|route|^\s*\d{4,5}\s+[A-Z0-9-]{3,}|"
    r"^\s*[-*\d]+[\.\):]\s|\b[A-Z]{2,}-\d{2,}\b|\bTXN[-_]?\d+|\bACC[-_]?\d+)",
    re.I,
)


def load_arm(
    tasks_root: Path,
    arm: str,
    run_tag: str,
    reps: List[int],
    families: Tuple[str, ...] = FAMILIES_3F,
) -> Dict[Tuple[str, str, int], dict]:
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
            if str(r.get("length")) != "128k":
                continue
            if str(r.get("family")) not in families:
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


def summarize_arm(rows: Dict[Tuple[str, str, int], dict]) -> dict:
    sub = list(rows.values())
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
        if r["family"] == "join":
            join_tot += 1
            if has_run15(reas):
                join_ent += 1
                if r.get("finish_class") == "stop":
                    join_left += 1

    by_fam = {}
    for fam in FAMILIES_3F:
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
        },
        "finished_completion_tokens_q": quantiles(
            [float(r["completion_tokens"]) for r in fin if r.get("completion_tokens") is not None]
        ),
    }


def paired_boot(
    rows_a: Dict[Tuple[str, str, int], dict],
    rows_b: Dict[Tuple[str, str, int], dict],
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    common = sorted(k for k in rows_a if k in rows_b)
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

    bf_items = sorted(bf_by_item)
    bf_ds = [sum(bf_by_item[i]) / len(bf_by_item[i]) for i in bf_items]
    rng2 = random.Random(seed)
    bf_b = (
        sorted(
            sum(bf_ds[rng2.randrange(len(bf_ds))] for _ in range(len(bf_ds))) / len(bf_ds)
            for _ in range(n_boot)
        )
        if bf_ds
        else []
    )

    return {
        "n_items": n,
        "n_paired_replies": len(common),
        "mean_A": round(sum(sa_i) / n, 6),
        "mean_B": round(sum(sb_i) / n, 6),
        "B_minus_A": round(md, 6),
        "boot_se": round(sed, 6),
        "z": round(md / sed, 4) if sed > 0 else None,
        "ci95": [round(bd[int(0.025 * n_boot)], 6), round(bd[int(0.975 * n_boot)], 6)],
        "excludes_zero": bool(bd[int(0.025 * n_boot)] > 0.0 or bd[int(0.975 * n_boot)] < 0.0),
        "unf_rate_B_minus_A": round(mu, 6),
        "unf_boot_se": round(seu, 6),
        "unf_z": round(mu / seu, 4) if seu > 0 else None,
        "unf_rate_ci95": [round(bu[int(0.025 * n_boot)], 6), round(bu[int(0.975 * n_boot)], 6)],
        "unf_excludes_zero": bool(bu[int(0.025 * n_boot)] > 0.0 or bu[int(0.975 * n_boot)] < 0.0),
        "both_finished_B_minus_A": round(sum(bf_ds) / len(bf_ds), 6) if bf_ds else None,
        "both_finished_ci95": (
            [round(bf_b[int(0.025 * n_boot)], 6), round(bf_b[int(0.975 * n_boot)], 6)]
            if bf_b
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3b-dir", required=True)
    ap.add_argument("--run-tag", default="p3b")
    ap.add_argument("--arm", default="k377-not-b1b2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks_root = Path(args.stage3b_dir) / "tasks"
    arms = ["k325-native", "k325-rebuilt", "k345-b1b2", args.arm, "k4-bf16core"]
    s1 = {a: load_arm(tasks_root, a, args.run_tag, [0, 1, 2, 3]) for a in arms}
    s2 = {a: load_arm(tasks_root, a, args.run_tag, [4, 5, 6, 7]) for a in arms}
    all8 = {a: load_arm(tasks_root, a, args.run_tag, list(range(8))) for a in arms}

    levels = {}
    for a in arms:
        if not all8[a]:
            continue
        l1 = summarize_arm(s1[a])
        l2 = summarize_arm(s2[a])
        p_halves = (
            fisher_exact_two_sided(
                l1["unfinished"],
                l1["n"] - l1["unfinished"],
                l2["unfinished"],
                l2["n"] - l2["unfinished"],
            )
            if l1["n"] and l2["n"]
            else None
        )
        levels[a] = {
            "128k_jl_reps0_3": l1,
            "128k_jl_reps4_7": l2,
            "128k_jl_all8": summarize_arm(all8[a]),
            "halves_unfinished_fisher_p": p_halves,
        }

    reading = "pending"
    if args.arm in levels and levels[args.arm]["128k_jl_all8"]["n"] == 320:
        c_unf = levels[args.arm]["128k_jl_all8"]["unfinished"]
        if c_unf <= 19:
            reading = (
                f"{args.arm} ~= k4-bf16core (<=19/320, obs {c_unf}/320): "
                "128K runaway stability is localized outside B1-B2"
            )
        elif c_unf >= 25:
            reading = (
                f"{args.arm} ~= k325-rebuilt (>=25/320, obs {c_unf}/320): "
                "neither B1-B2 nor its 5/7 complement recovers K4 stability alone (diffuse / super-additive)"
            )
        else:
            reading = f"intermediate / partial recovery (20-24/320, obs {c_unf}/320)"

    contrasts = {
        "k4-bf16core_vs_k325-rebuilt": paired_boot(all8["k325-rebuilt"], all8["k4-bf16core"]),
        "k4-bf16core_vs_k325-native": paired_boot(all8["k325-native"], all8["k4-bf16core"]),
        "k325-rebuilt_vs_k325-native": paired_boot(all8["k325-native"], all8["k325-rebuilt"]),
        "k345-b1b2_vs_k325-rebuilt": paired_boot(all8["k325-rebuilt"], all8["k345-b1b2"]),
    }
    if all8[args.arm]:
        contrasts.update(
            {
                f"{args.arm}_vs_k325-rebuilt": paired_boot(all8["k325-rebuilt"], all8[args.arm]),
                f"{args.arm}_vs_k4-bf16core": paired_boot(all8["k4-bf16core"], all8[args.arm]),
                f"{args.arm}_vs_k325-native": paired_boot(all8["k325-native"], all8[args.arm]),
                f"{args.arm}_vs_k345-b1b2": paired_boot(all8["k345-b1b2"], all8[args.arm]),
            }
        )

    out = {
        "schema": "kintsugi-phase3f-complement-v1",
        "families": list(FAMILIES_3F),
        "target_arm": args.arm,
        "frozen_L3F_1_reading": reading,
        "levels": levels,
        "contrasts_128k_jl_8pass": contrasts,
    }
    if args.out:
        p_out = Path(args.out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_out.write_text(json.dumps(out, indent=2) + "\n")

    print("=== STAGE 3F COMPLEMENT GRAFT (8-PASS 128K join+ledger, n=320/arm) ===")
    print(f"FROZEN L3F.1 READING: {reading}")
    for a in arms:
        if a not in levels:
            continue
        l = levels[a]["128k_jl_all8"]
        l1 = levels[a]["128k_jl_reps0_3"]
        l2 = levels[a]["128k_jl_reps4_7"]
        print(
            f"  {a:16s} | n={l['n']:3d} | acc={100*l['mean_score']:5.2f}% "
            f"(fin={100*l['fin_only_score']:5.2f}%, "
            f"unf={l['unfinished']:2d}/{l['n']} [{l1['unfinished']}+{l2['unfinished']}, p={levels[a]['halves_unfinished_fisher_p']}], "
            f"join_unf={l['by_family']['join']['unfinished']}, ledger_unf={l['by_family']['ledger']['unfinished']}, "
            f"join_enter={l['join_enter_leave_ge15']['entered_ge15']}/{l['join_enter_leave_ge15']['join_replies']})"
        )
    print("\n=== PAIRED 8-PASS 128K join+ledger CONTRASTS ===")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
