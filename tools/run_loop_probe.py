#!/usr/bin/env python3
"""
Run the frozen Stage 3C-A Loop Entry/Exit Margin Probe (`probe_loop/manifest.json`)
against a running CuteAFD v6.0.0 score-mode server via `/v1/bench/probe`.

For every cut position (`enter`, `exit_5`, `exit_10`, `exit_20`, and `leave` on
controls), records:
  - `m_class`: logsumexp(logp(record_start_tokens in pool))
               - logsumexp(logp(non_record_tokens in pool))
               (primary symmetric class log-odds over pool = top_k=10 U want).
  - `m_stick`: logsumexp(logp(record_start_tokens in pool))
               - max(logp(non_record_tokens in pool))
               (secondary max-exit variant).
  - `m_raw`:   logp(trajectory_next_token) - max(logp(other token in top_5))
               (secondary single-token formula matching `B_close`).
  - `pool_mass`: exp(logsumexp(logp(all tokens in pool))) coverage diagnostic.
And at each newline boundary `pos - 1` (`nl_cuts`), records:
  - `m_nl`:    logp('\n' [198]) - logp('\n\n' [271]).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set


JOIN_RECORD_START_IDS: Set[int] = {66384, 95115, 26596, 19280, 27649}
NL_SINGLE_ID = 198
NL_DOUBLE_ID = 271


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def api_headers(key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def get_model_id(base_url: str, key: Optional[str]) -> str:
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers=api_headers(key))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def post_probe(base_url: str, key: Optional[str], body: dict, spec: dict, timeout: float) -> tuple[dict, float]:
    payload = json.dumps({"body": body, "spec": spec}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/bench/probe",
        data=payload,
        method="POST",
        headers=api_headers(key),
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:600]!r}") from e
    dt = time.perf_counter() - t0
    pr = reply.get("probe") or {}
    if pr.get("error") or not pr.get("engine"):
        raise RuntimeError(f"probe rejected/failed: {pr.get('error')!r}")
    return reply, dt


def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def is_record_start_token(family: str, tid: int, tok) -> bool:
    if family == "join":
        return tid in JOIN_RECORD_START_IDS
    # ledger: any pure ASCII digit token (0, 00, 09, 001, etc.)
    s = tok.decode([tid])
    return len(s) > 0 and s.isascii() and s.isdigit()


def compute_cut_metrics(row: dict, target_id: int, family: str, tok) -> dict:
    wanted_map = {int(tid): float(lp) for tid, lp in row.get("wanted", [])}
    top_all = [(int(tid), float(lp)) for tid, lp in row.get("top", [])]
    if target_id not in wanted_map:
        raise RuntimeError(f"target_id {target_id} missing from wanted at pos {row.get('position')}")
    lp_target = wanted_map[target_id]

    top5 = top_all[:5]
    others5 = [lp for tid, lp in top5 if tid != target_id]
    if not others5:
        raise RuntimeError(f"no runner-up token in top_5 at pos {row.get('position')}")
    lp_best_other5 = max(others5)
    m_raw = lp_target - lp_best_other5

    # Combined candidate pool for class-level stickiness margin `m_stick`
    pool: Dict[int, float] = {}
    for tid, lp in top_all:
        pool[tid] = lp
    for tid, lp in wanted_map.items():
        pool.setdefault(tid, lp)

    rec_lps = [lp for tid, lp in pool.items() if is_record_start_token(family, tid, tok)]
    non_rec_lps = [lp for tid, lp in pool.items() if not is_record_start_token(family, tid, tok)]
    all_lps = list(pool.values())
    pool_log_mass = logsumexp(all_lps) if all_lps else None
    pool_mass = min(1.0, math.exp(pool_log_mass)) if pool_log_mass is not None else None
    m_class = (logsumexp(rec_lps) - logsumexp(non_rec_lps)) if (rec_lps and non_rec_lps) else None
    m_stick = (logsumexp(rec_lps) - max(non_rec_lps)) if (rec_lps and non_rec_lps) else None

    top1_tid = int(row["argmax"])
    return {
        "position": int(row["position"]),
        "target_id": target_id,
        "argmax": top1_tid,
        "argmax_is_record": is_record_start_token(family, top1_tid, tok),
        "top1_match": top1_tid == target_id,
        "lp_target": round(lp_target, 6),
        "lp_best_other5": round(lp_best_other5, 6),
        "m_class": round(m_class, 6) if m_class is not None else None,
        "m_stick": round(m_stick, 6) if m_stick is not None else None,
        "m_raw": round(m_raw, 6),
        "pool_mass": round(pool_mass, 6) if pool_mass is not None else None,
        "pool_log_mass": round(pool_log_mass, 6) if pool_log_mass is not None else None,
        "top5": [[tid, round(lp, 5)] for tid, lp in top5],
        "hash": row.get("hash", ""),
        "finite": bool(row.get("finite", True)),
    }


def score_prefix_response(rec: dict, probe_rec: dict, elapsed_s: float, tok) -> dict:
    rows_by_pos = {int(r["position"]): r for r in probe_rec.get("rows", [])}
    prompt_ids = rec["prompt_ids"]
    family = rec["family"]

    cuts_out = {}
    all_finite = True
    for cname, pos in rec["cuts"].items():
        cm = compute_cut_metrics(rows_by_pos[int(pos)], int(prompt_ids[int(pos)]), family, tok)
        cuts_out[cname] = cm
        all_finite = all_finite and cm["finite"]

    nl_out = {}
    for cname, pos in rec.get("nl_cuts", {}).items():
        row = rows_by_pos[int(pos)]
        w = {int(tid): float(lp) for tid, lp in row.get("wanted", [])}
        if NL_SINGLE_ID in w and NL_DOUBLE_ID in w:
            nl_out[cname] = round(w[NL_SINGLE_ID] - w[NL_DOUBLE_ID], 6)
        all_finite = all_finite and bool(row.get("finite", True))

    return {
        "prefix_id": rec["prefix_id"],
        "kind": rec["kind"],
        "author": rec["author"],
        "item_id": rec["item_id"],
        "family": family,
        "rep": rec["rep"],
        "run_len": rec["run_len"],
        "n_tokens": rec["n_tokens"],
        "elapsed_s": round(elapsed_s, 3),
        "all_finite": all_finite,
        "cuts": cuts_out,
        "nl_margins": nl_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    from tokenizers import Tokenizer  # type: ignore

    tok = Tokenizer.from_file(args.tokenizer)
    probe_dir = Path(args.probe_dir)
    manifest = json.loads((probe_dir / "manifest.json").read_text())
    items_meta = manifest["items"]

    key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    model_id = get_model_id(args.base_url, key)

    out_dir = Path(args.out_dir)
    rec_dir = out_dir / "items" / f"{args.arm}-{args.tag}"
    rec_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_all = time.monotonic()
    for idx, meta in enumerate(items_meta, 1):
        pid = meta["prefix_id"]
        rec_path = rec_dir / f"{pid}.json"
        if rec_path.exists():
            res = json.loads(rec_path.read_text())
            results.append(res)
            e10 = res["cuts"].get("exit_10", {})
            print(
                f"[{idx:03d}/{len(items_meta):03d}] {pid:40s} CACHED "
                f"cls10={e10.get('m_class')} stk10={e10.get('m_stick')} raw10={e10.get('m_raw')} cov={e10.get('pool_mass')}",
                flush=True,
            )
            continue

        item_file = probe_dir / meta["file"]
        if sha256_file(item_file) != meta["sha256"]:
            raise RuntimeError(f"SHA mismatch on frozen loop-probe prefix {item_file}")
        rec = json.loads(gzip.open(item_file, "rt").read())

        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 1,
        }
        spec = {
            "cold": True,
            "prompt_ids": rec["prompt_ids"],
            "score_from": int(rec["score_from"]),
            "score_path": "prefill",
            "score_positions": [int(p) for p in rec["score_positions"]],
            "top_k": int(args.top_k),
            "want": {int(k): v for k, v in rec["want"].items()},
        }
        reply, dt = post_probe(args.base_url, key, body, spec, args.timeout)
        res = score_prefix_response(rec, reply["probe"], dt, tok)
        tmp = rec_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        tmp.replace(rec_path)
        results.append(res)
        e10 = res["cuts"].get("exit_10", {})
        print(
            f"[{idx:03d}/{len(items_meta):03d}] {pid:40s} {dt:5.1f}s "
            f"cls10={e10.get('m_class')} stk10={e10.get('m_stick')} raw10={e10.get('m_raw')} cov={e10.get('pool_mass')}",
            flush=True,
        )

    summary = {
        "schema": "kintsugi-loop-margin-run-v1",
        "arm": args.arm,
        "tag": args.tag,
        "model_id": model_id,
        "n_prefixes": len(results),
        "wall_s": round(time.monotonic() - t_all, 2),
        "all_finite": all(r["all_finite"] for r in results),
        "items": results,
    }
    out_summary = out_dir / f"{args.arm}-{args.tag}.loop_margins.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {args.arm}-{args.tag} (n={len(results)}) in {summary['wall_s']:.1f}s -> {out_summary}")


if __name__ == "__main__":
    main()
