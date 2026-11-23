#!/usr/bin/env python3
"""
Run a frozen two-component teacher-forced margin probe (`D_margin_build` or
`D_margin_heldout`) against a running CuteAFD v6.0.0 score-mode server via
`/v1/bench/probe` (inline top-k + wanted logprobs, zero disk dumps).

Outputs one crash-safe `.json` receipt per item plus an aggregated
`<arm>-<tag>.margins.json` summary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


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


def compute_row_margin(row: dict, target_id: int) -> dict:
    wanted_map = {int(tid): float(lp) for tid, lp in row.get("wanted", [])}
    if target_id not in wanted_map:
        raise RuntimeError(f"target_id {target_id} missing from wanted at pos {row.get('position')}")
    lp_target = wanted_map[target_id]
    top = [(int(tid), float(lp)) for tid, lp in row.get("top", [])]
    others = [lp for tid, lp in top if tid != target_id]
    if not others:
        raise RuntimeError(f"no runner-up token in top_k at pos {row.get('position')}")
    lp_best_other = max(others)
    margin = lp_target - lp_best_other
    return {
        "position": int(row["position"]),
        "target_id": target_id,
        "argmax": int(row["argmax"]),
        "top1_match": int(row["argmax"]) == target_id,
        "lp_target": lp_target,
        "lp_best_other": lp_best_other,
        "margin": margin,
        "hash": row.get("hash", ""),
        "finite": bool(row.get("finite", True)),
    }


def score_item_response(item_rec: dict, probe_rec: dict, elapsed_s: float) -> dict:
    rows_by_pos = {int(r["position"]): r for r in probe_rec.get("rows", [])}
    prompt_ids = item_rec["prompt_ids"]

    comp_a_rows = [
        compute_row_margin(rows_by_pos[pos], int(prompt_ids[pos]))
        for pos in item_rec["comp_a_positions"]
    ]
    comp_b_close = compute_row_margin(
        rows_by_pos[item_rec["comp_b_close_position"]],
        int(prompt_ids[item_rec["comp_b_close_position"]]),
    )
    comp_b_tail = [
        compute_row_margin(rows_by_pos[pos], int(prompt_ids[pos]))
        for pos in item_rec["comp_b_tail_positions"]
    ]

    a_margins = [r["margin"] for r in comp_a_rows]
    b_tail_margins = [r["margin"] for r in comp_b_tail]
    all_rows = comp_a_rows + [comp_b_close] + comp_b_tail

    return {
        "item_id": item_rec["item_id"],
        "family": item_rec["family"],
        "split": item_rec["split"],
        "k325_hist_unf": item_rec["k325_hist_unf"],
        "n_tokens": item_rec["n_tokens"],
        "elapsed_s": round(elapsed_s, 3),
        "all_finite": all(r["finite"] for r in all_rows),
        "comp_a": {
            "n_tokens": len(a_margins),
            "min_margin": min(a_margins),
            "mean_margin": sum(a_margins) / len(a_margins),
            "top1_acc": sum(1 for r in comp_a_rows if r["top1_match"]) / len(comp_a_rows),
            "neg_tokens": sum(1 for m in a_margins if m < 0.0),
            "rows": comp_a_rows,
        },
        "comp_b": {
            "close_margin": comp_b_close["margin"],
            "close_top1": comp_b_close["top1_match"],
            "close_hash": comp_b_close["hash"],
            "tail_min_margin": min(b_tail_margins),
            "tail_mean_margin": sum(b_tail_margins) / len(b_tail_margins),
            "combined_min_margin": min(b_tail_margins + [comp_b_close["margin"]]),
            "close_row": comp_b_close,
            "tail_rows": comp_b_tail,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--split", choices=("build", "heldout", "all"), default="build")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    manifest = json.loads((probe_dir / "manifest.json").read_text())
    items_meta = [
        m for m in manifest["items"]
        if args.split == "all" or m["split"] == args.split
    ]

    key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    model_id = get_model_id(args.base_url, key)

    out_dir = Path(args.out_dir)
    rec_dir = out_dir / "items" / f"{args.arm}-{args.tag}"
    rec_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_all = time.monotonic()
    for idx, meta in enumerate(items_meta, 1):
        item_id = meta["item_id"]
        rec_path = rec_dir / f"{item_id}.json"
        if rec_path.exists():
            res = json.loads(rec_path.read_text())
            results.append(res)
            print(
                f"[{idx:02d}/{len(items_meta):02d}] {item_id:12s} CACHED "
                f"A_min={res['comp_a']['min_margin']:+.3f} B_close={res['comp_b']['close_margin']:+.3f}",
                flush=True,
            )
            continue

        item_file = probe_dir / meta["file"]
        if sha256_file(item_file) != meta["sha256"]:
            raise RuntimeError(f"SHA mismatch on frozen probe item {item_file}")
        item_rec = json.loads(gzip.open(item_file, "rt").read())

        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 1,
        }
        spec = {
            "cold": True,
            "prompt_ids": item_rec["prompt_ids"],
            "score_from": int(item_rec["score_from"]),
            "score_path": "prefill",
            "score_positions": [int(p) for p in item_rec["score_positions"]],
            "top_k": int(args.top_k),
            "want": {int(k): v for k, v in item_rec["want"].items()},
        }
        reply, dt = post_probe(args.base_url, key, body, spec, args.timeout)
        res = score_item_response(item_rec, reply["probe"], dt)
        tmp = rec_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        tmp.replace(rec_path)
        results.append(res)
        print(
            f"[{idx:02d}/{len(items_meta):02d}] {item_id:12s} {dt:5.1f}s "
            f"A_min={res['comp_a']['min_margin']:+.3f} A_mean={res['comp_a']['mean_margin']:+.3f} "
            f"B_close={res['comp_b']['close_margin']:+.3f} B_tail={res['comp_b']['tail_mean_margin']:+.3f}",
            flush=True,
        )

    summary = {
        "schema": "kintsugi-margin-run-v1",
        "arm": args.arm,
        "tag": args.tag,
        "split": args.split,
        "model_id": model_id,
        "n_items": len(results),
        "wall_s": round(time.monotonic() - t_all, 2),
        "all_finite": all(r["all_finite"] for r in results),
        "comp_a_min_mean": sum(r["comp_a"]["min_margin"] for r in results) / len(results),
        "comp_a_mean_mean": sum(r["comp_a"]["mean_margin"] for r in results) / len(results),
        "comp_a_fragile_lt1": sum(1 for r in results if r["comp_a"]["min_margin"] < 1.0),
        "comp_a_neg_lt0": sum(1 for r in results if r["comp_a"]["min_margin"] < 0.0),
        "comp_b_close_mean": sum(r["comp_b"]["close_margin"] for r in results) / len(results),
        "comp_b_tail_mean": sum(r["comp_b"]["tail_mean_margin"] for r in results) / len(results),
        "comp_b_close_fragile_lt1": sum(1 for r in results if r["comp_b"]["close_margin"] < 1.0),
        "comp_b_close_neg_lt0": sum(1 for r in results if r["comp_b"]["close_margin"] < 0.0),
        "items": results,
    }
    out_summary = out_dir / f"{args.arm}-{args.tag}.margins.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"DONE {args.arm}-{args.tag} ({args.split}, n={len(results)}) in {summary['wall_s']:.1f}s: "
        f"A_min={summary['comp_a_min_mean']:+.4f} (frag<1:{summary['comp_a_fragile_lt1']}, neg<0:{summary['comp_a_neg_lt0']}) | "
        f"B_close={summary['comp_b_close_mean']:+.4f} (frag<1:{summary['comp_b_close_fragile_lt1']}, neg<0:{summary['comp_b_close_neg_lt0']})"
    )


if __name__ == "__main__":
    main()
