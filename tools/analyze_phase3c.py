#!/usr/bin/env python3
"""
Analyzer for Phase 3 Stage 3C:
  - Stage 3C-A (`--stage 3ca`): Loop Entry/Exit Margin Probe (`M_enter`, `M_exit(5,10,20)`).
    Evaluates the single frozen primary statistic (`exit_10` on dump prefixes, all authors
    pooled, paired item-cluster bootstrap, 10,000 resamples, seed 20261123), the anchor
    separation gate (`k4-bf16core - k325-native` CI < 0), and writes the recorded
    `k345-b1b2` recovery-share prediction (with 95% CI) BEFORE Stage 3C-B begins.
  - Stage 3C-B (`--stage 3cb`): `k345-b1b2` mixed-regime replication (`reps 8-11`, 4K+128K).
    Evaluates the frozen unfinished thresholds (`<=12` fluctuation / `13-17` unresolved /
    `>=18` mixed-regime interaction), descriptive pooled 12-rep summary (`L3.3` not
    reopened), scores the Step 1 recorded prediction against the replication's unfinished
    recovery share vs the anchors' 8-pass rates, and runs the frozen tail classifier +
    stage-split onset forensics.
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


def item_cluster_boot_diff(
    rows_a: Dict[str, dict],
    rows_b: Dict[str, dict],
    cut: str,
    metric: str,
    prefix_ids: List[str],
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    """Paired prefix differences clustered by `item_id` (equal weight per prefix inside
    each item cluster, resampling items with replacement)."""
    by_item: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for pid in prefix_ids:
        ra, rb = rows_a.get(pid), rows_b.get(pid)
        if not ra or not rb:
            continue
        ca, cb = ra["cuts"].get(cut), rb["cuts"].get(cut)
        if not ca or not cb:
            continue
        va, vb = ca.get(metric), cb.get(metric)
        if va is None or vb is None:
            continue
        by_item[ra["item_id"]].append((float(va), float(vb)))

    items = sorted(by_item)
    if not items:
        return {"n_prefixes": 0, "n_items": 0}

    all_a = [a for i in items for a, _ in by_item[i]]
    all_b = [b for i in items for _, b in by_item[i]]
    n_pref = len(all_a)
    mean_a = sum(all_a) / n_pref
    mean_b = sum(all_b) / n_pref
    diff = mean_b - mean_a

    # Resample items with replacement, weighting by prefix count per sampled item
    item_sums = [(len(by_item[i]), sum(b - a for a, b in by_item[i])) for i in items]
    n_items = len(items)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s_num = 0.0
        s_cnt = 0
        for _ in range(n_items):
            cnt, sdiff = item_sums[rng.randrange(n_items)]
            s_num += sdiff
            s_cnt += cnt
        boots.append(s_num / s_cnt)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    mu_b = sum(boots) / n_boot
    se = math.sqrt(sum((x - mu_b) ** 2 for x in boots) / (n_boot - 1))
    return {
        "n_prefixes": n_pref,
        "n_items": n_items,
        "mean_A": round(mean_a, 6),
        "mean_B": round(mean_b, 6),
        "B_minus_A": round(diff, 6),
        "boot_se": round(se, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "k4_less_sticky_detected": bool(hi < 0.0),
    }


def recovery_share_boot(
    rows_base: Dict[str, dict],
    rows_cand: Dict[str, dict],
    rows_k4: Dict[str, dict],
    cut: str,
    metric: str,
    prefix_ids: List[str],
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    by_item: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for pid in prefix_ids:
        r0, rc, r4 = rows_base.get(pid), rows_cand.get(pid), rows_k4.get(pid)
        if not (r0 and rc and r4):
            continue
        v0 = (r0["cuts"].get(cut) or {}).get(metric)
        vc = (rc["cuts"].get(cut) or {}).get(metric)
        v4 = (r4["cuts"].get(cut) or {}).get(metric)
        if v0 is None or vc is None or v4 is None:
            continue
        by_item[r0["item_id"]].append((float(vc) - float(v0), float(v4) - float(v0)))

    items = sorted(by_item)
    if not items:
        return {"share": None, "ci95": None}
    num = sum(n for i in items for n, _ in by_item[i])
    den = sum(d for i in items for _, d in by_item[i])
    if abs(den) < 1e-12:
        return {"share": None, "ci95": None}
    pt = num / den

    item_sums = [(sum(n for n, _ in by_item[i]), sum(d for _, d in by_item[i])) for i in items]
    n_items = len(items)
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        sn = sd = 0.0
        for _ in range(n_items):
            dn, dd = item_sums[rng.randrange(n_items)]
            sn += dn
            sd += dd
        if abs(sd) > 1e-12:
            ratios.append(sn / sd)
    ratios.sort()
    ci = [round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios))], 4)] if len(ratios) >= 100 else None
    return {"share": round(pt, 4), "ci95": ci, "n_items": n_items, "n_prefixes": sum(len(v) for v in by_item.values())}


def dump_minus_control_boot(
    rows: Dict[str, dict],
    cut: str,
    metric: str,
    n_boot: int = 10000,
    seed: int = 20261123,
) -> dict:
    d_by_item: Dict[str, List[float]] = defaultdict(list)
    c_by_item: Dict[str, List[float]] = defaultdict(list)
    for r in rows.values():
        v = (r["cuts"].get(cut) or {}).get(metric)
        if v is None:
            continue
        if r["kind"] == "dump":
            d_by_item[r["item_id"]].append(float(v))
        else:
            c_by_item[r["item_id"]].append(float(v))
    d_all = [x for vs in d_by_item.values() for x in vs]
    c_all = [x for vs in c_by_item.values() for x in vs]
    if not d_all or not c_all:
        return {"n_dump": len(d_all), "n_control": len(c_all)}
    md, mc = sum(d_all) / len(d_all), sum(c_all) / len(c_all)
    d_items = [(len(v), sum(v)) for v in d_by_item.values()]
    c_items = [(len(v), sum(v)) for v in c_by_item.values()]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        dn = dc = cn = cc = 0.0
        for _ in range(len(d_items)):
            k, s = d_items[rng.randrange(len(d_items))]
            dc += k
            dn += s
        for _ in range(len(c_items)):
            k, s = c_items[rng.randrange(len(c_items))]
            cc += k
            cn += s
        boots.append((dn / dc) - (cn / cc))
    boots.sort()
    return {
        "n_dump": len(d_all),
        "n_control": len(c_all),
        "mean_dump": round(md, 6),
        "mean_control": round(mc, 6),
        "dump_minus_control": round(md - mc, 6),
        "ci95": [round(boots[int(0.025 * n_boot)], 6), round(boots[int(0.975 * n_boot)], 6)],
    }


def run_stage_3ca(args) -> None:
    mdir = Path(args.margins_dir)
    arms = ["k325-native", "k345-b1b2", "k4-bf16core"]
    by_arm: Dict[str, Dict[str, dict]] = {}
    for arm in arms:
        p = mdir / f"{arm}-{args.tag}.loop_margins.json"
        doc = json.loads(p.read_text())
        by_arm[arm] = {r["prefix_id"]: r for r in doc["items"]}

    all_pids = sorted(by_arm["k325-native"].keys())
    dump_pids = [pid for pid in all_pids if by_arm["k325-native"][pid]["kind"] == "dump"]
    authors = sorted({by_arm["k325-native"][pid]["author"] for pid in dump_pids})
    families = sorted({by_arm["k325-native"][pid]["family"] for pid in dump_pids})

    out: dict = {
        "schema": "kintsugi-phase3c-a-summary-v1",
        "primary_metric": args.primary_metric,
        "primary_cut": "exit_10",
        "metrics": {},
    }

    for metric in ("m_stick", "m_raw"):
        primary_anchor = item_cluster_boot_diff(by_arm["k325-native"], by_arm["k4-bf16core"], "exit_10", metric, dump_pids)
        k345_vs_native = item_cluster_boot_diff(by_arm["k325-native"], by_arm["k345-b1b2"], "exit_10", metric, dump_pids)
        k345_vs_k4 = item_cluster_boot_diff(by_arm["k4-bf16core"], by_arm["k345-b1b2"], "exit_10", metric, dump_pids)
        k345_share = recovery_share_boot(
            by_arm["k325-native"], by_arm["k345-b1b2"], by_arm["k4-bf16core"], "exit_10", metric, dump_pids
        )
        by_cut = {
            cut: {
                "k4_vs_k325": item_cluster_boot_diff(by_arm["k325-native"], by_arm["k4-bf16core"], cut, metric, dump_pids),
                "k345_vs_k325": item_cluster_boot_diff(by_arm["k325-native"], by_arm["k345-b1b2"], cut, metric, dump_pids),
                "k345_share": recovery_share_boot(by_arm["k325-native"], by_arm["k345-b1b2"], by_arm["k4-bf16core"], cut, metric, dump_pids),
            }
            for cut in ("enter", "exit_5", "exit_10", "exit_20")
        }
        by_author = {
            auth: {
                "k4_vs_k325": item_cluster_boot_diff(
                    by_arm["k325-native"], by_arm["k4-bf16core"], "exit_10", metric,
                    [pid for pid in dump_pids if by_arm["k325-native"][pid]["author"] == auth],
                ),
                "k345_vs_k325": item_cluster_boot_diff(
                    by_arm["k325-native"], by_arm["k345-b1b2"], "exit_10", metric,
                    [pid for pid in dump_pids if by_arm["k325-native"][pid]["author"] == auth],
                ),
            }
            for auth in authors
        }
        by_family = {
            fam: {
                "k4_vs_k325": item_cluster_boot_diff(
                    by_arm["k325-native"], by_arm["k4-bf16core"], "exit_10", metric,
                    [pid for pid in dump_pids if by_arm["k325-native"][pid]["family"] == fam],
                ),
                "k345_vs_k325": item_cluster_boot_diff(
                    by_arm["k325-native"], by_arm["k345-b1b2"], "exit_10", metric,
                    [pid for pid in dump_pids if by_arm["k325-native"][pid]["family"] == fam],
                ),
            }
            for fam in families
        }
        dump_vs_ctrl = {
            arm: {cut: dump_minus_control_boot(by_arm[arm], cut, metric) for cut in ("enter", "exit_5", "exit_10", "exit_20")}
            for arm in arms
        }
        out["metrics"][metric] = {
            "primary_anchor_exit10": primary_anchor,
            "k345_vs_native_exit10": k345_vs_native,
            "k345_vs_k4_exit10": k345_vs_k4,
            "k345_recovery_share_exit10": k345_share,
            "by_cut": by_cut,
            "by_author_exit10": by_author,
            "by_family_exit10": by_family,
            "dump_minus_control": dump_vs_ctrl,
        }

    prim = out["metrics"][args.primary_metric]
    out["gate"] = {
        "primary_metric": args.primary_metric,
        "primary_cut": "exit_10",
        "k4_less_sticky_detected": prim["primary_anchor_exit10"]["k4_less_sticky_detected"],
        "recorded_k345_prediction": prim["k345_recovery_share_exit10"],
    }

    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["gate"], indent=2))


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


def find_dump_onset(reasoning: str) -> Optional[int]:
    lines = reasoning.splitlines(keepends=True)
    pos = 0
    starts = []
    for ln in lines:
        starts.append(pos)
        pos += len(ln)
    run = 0
    for idx, ln in enumerate(lines):
        if RAW_REC_RE.match(ln.strip()):
            run += 1
            if run == 15:
                return starts[idx - 14]
        elif ln.strip():
            run = 0
    return None


def summarize_reps(arm_dir: Path, reps: List[int], length: str) -> dict:
    rows = []
    for rep in reps:
        p = arm_dir / f"single-rep{rep}" / "requests.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("final", True) or r.get("status") != "ok" or str(r.get("length")) != length:
                continue
            r["_rep_dir"] = str(p.parent)
            rows.append(r)
    if not rows:
        return {"n": 0}
    def sc(r):
        s = r.get("score") or {}
        if s.get("score") is not None:
            return float(s["score"])
        return float(s.get("correct", 0)) / float(s.get("fields", 1))

    fin = [r for r in rows if r.get("finish_class") == "stop"]
    unf = [r for r in rows if r.get("finish_class") != "stop"]
    tails: Counter = Counter()
    onsets: List[int] = []
    per_rep_unf: Dict[str, int] = {}
    for rep in reps:
        per_rep_unf[f"r{rep}"] = sum(1 for r in unf if int(r["rep"]) == rep)
    for r in unf:
        tf = Path(r["_rep_dir"]) / "texts" / f"{r['unit']}.{r['session']}.json.gz"
        if tf.exists():
            reas = json.loads(gzip.open(tf, "rt").read())["turns"][0].get("reasoning", "") or ""
            tails[classify_tail(reas)] += 1
            oc = find_dump_onset(reas)
            if oc is not None:
                onsets.append(oc)
    onsets.sort()
    return {
        "n": len(rows),
        "mean_score": round(sum(sc(r) for r in rows) / len(rows), 6),
        "fin_only_score": round(sum(sc(r) for r in fin) / len(fin), 6) if fin else None,
        "unfinished": len(unf),
        "unfinished_rate": round(len(unf) / len(rows), 6),
        "per_rep_unfinished": per_rep_unf,
        "unfinished_by_family": dict(sorted(Counter(r["family"] for r in unf).items())),
        "unfinished_tail_classes": dict(sorted(tails.items())),
        "dump_onsets_ge15": len(onsets),
        "onset_char_min_med_max": [onsets[0], onsets[len(onsets) // 2], onsets[-1]] if onsets else None,
    }


def run_stage_3cb(args) -> None:
    s3b_k345 = Path(args.stage3b_tasks_dir) / "k345-b1b2" / "p3b"
    s3b_nat = Path(args.stage3b_tasks_dir) / "k325-native" / "p3b"
    s3b_k4 = Path(args.stage3b_tasks_dir) / "k4-bf16core" / "p3b"
    s3c_k345 = Path(args.stage3c_tasks_dir) / "k345-b1b2" / args.run_tag

    rep_128 = summarize_reps(s3c_k345, [8, 9, 10, 11], "128k")
    rep_4k = summarize_reps(s3c_k345, [8, 9, 10, 11], "4k")
    s1_128 = summarize_reps(s3b_k345, [0, 1, 2, 3], "128k")
    s2_128 = summarize_reps(s3b_k345, [4, 5, 6, 7], "128k")
    nat_8 = summarize_reps(s3b_nat, list(range(8)), "128k")
    k4_8 = summarize_reps(s3b_k4, list(range(8)), "128k")

    u = rep_128["unfinished"]
    if u <= 12:
        reading = "reps 0-3 were a fluctuation (<=12/240)"
    elif u >= 18:
        reading = "mixed regime interacts with k345-b1b2 (>=18/240)"
    else:
        reading = "unresolved (13-17/240)"

    # Pooled 12-rep descriptive summary (L3.3 NOT reopened)
    tot_n = s1_128["n"] + s2_128["n"] + rep_128["n"]
    tot_unf = s1_128["unfinished"] + s2_128["unfinished"] + rep_128["unfinished"]
    pooled_12_score = (
        s1_128["mean_score"] * s1_128["n"] + s2_128["mean_score"] * s2_128["n"] + rep_128["mean_score"] * rep_128["n"]
    ) / max(1, tot_n)

    # Replication unfinished recovery share vs anchors' 8-pass rates
    nat_r = nat_8["unfinished_rate"]
    k4_r = k4_8["unfinished_rate"]
    rep_r = rep_128["unfinished_rate"]
    unf_gap = nat_r - k4_r
    rep_unf_share = round((nat_r - rep_r) / unf_gap, 4) if abs(unf_gap) > 1e-12 else None

    pred_check = {}
    if args.prediction_json and Path(args.prediction_json).exists():
        pred = json.loads(Path(args.prediction_json).read_text())
        gate = pred.get("gate", {})
        rec_pred = gate.get("recorded_k345_prediction", {})
        ci = rec_pred.get("ci95")
        in_ci = bool(ci and rep_unf_share is not None and ci[0] <= rep_unf_share <= ci[1])
        pred_check = {
            "proxy_primary_metric": gate.get("primary_metric"),
            "proxy_anchor_gate_passed": gate.get("k4_less_sticky_detected"),
            "proxy_predicted_k345_share": rec_pred.get("share"),
            "proxy_predicted_k345_ci95": ci,
            "replication_unf_recovery_share_vs_8p_anchors": rep_unf_share,
            "replication_share_within_proxy_ci95": in_ci,
        }

    out = {
        "schema": "kintsugi-phase3c-b-summary-v1",
        "L3_3_verdict_unchanged": "FAIL (not reopened)",
        "frozen_reading": reading,
        "k345_reps8_11_4k": rep_4k,
        "k345_reps8_11_128k": rep_128,
        "k345_stage_split_128k": {"reps0_3_mixed": s1_128, "reps4_7_128k_only": s2_128, "reps8_11_mixed": rep_128},
        "k345_pooled_12rep_descriptive": {
            "n": tot_n,
            "mean_score": round(pooled_12_score, 6),
            "unfinished": tot_unf,
            "unfinished_rate": round(tot_unf / max(1, tot_n), 6),
        },
        "anchors_8pass_reference": {"k325-native": nat_8, "k4-bf16core": k4_8},
        "step1_prediction_score": pred_check,
    }
    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"frozen_reading": reading, "reps8_11_128k": rep_128, "step1_prediction_score": pred_check}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("3ca", "3cb"), required=True)
    ap.add_argument("--margins-dir", default="")
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--primary-metric", choices=("m_stick", "m_raw"), default="m_stick")
    ap.add_argument("--stage3b-tasks-dir", default="")
    ap.add_argument("--stage3c-tasks-dir", default="")
    ap.add_argument("--run-tag", default="p3c")
    ap.add_argument("--prediction-json", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.stage == "3ca":
        run_stage_3ca(args)
    else:
        run_stage_3cb(args)


if __name__ == "__main__":
    main()
