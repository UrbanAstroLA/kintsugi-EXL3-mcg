#!/usr/bin/env python3
"""
Stage 3C-B task runner (`reps 8-11`, 4K+128K mixed regime, concurrency 8,
max_tokens 32,768).

Clones the exact streaming, grading, and receipt format of `run_tasks_dc.py`,
while leaving `probelib_dc.py` and `run_tasks_dc.py` untouched and fixing the
`probelib_dc.seed_for` `0 <= rep < 10` limit (where `rep >= 10` would both raise
AssertionError and collide with `SEED_BASE["turns"] = 910_000_000`).

Uses an unused deterministic seed block:
  `SEED_BASE_3C = 950_000_000` (`reps 8-11` -> `958_000_000 .. 961_999_999`)
  `SALT_3C = "CONTEXT-PROBE-2026-11-10:3c"`
proven disjoint from all prior runs (`730M-885M`, `900M-939M`) and `reps 0-7`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SEED_BASE_3C = 950_000_000
SALT_3C = "CONTEXT-PROBE-2026-11-10:3c"


def seed_for_3c(P, rep: int, family: str, index: int, length: str, turn: int = 0) -> int:
    assert 0 <= turn < 64 and 0 <= index < 1000 and 8 <= rep < 20, f"invalid rep={rep}"
    ordinal = (P.FAMILY_INDEX[family] * 1000 + index) * 4 + P.LENGTH_INDEX[length]
    return SEED_BASE_3C + rep * 1_000_000 + ordinal * 64 + turn


def load_units(P, set_dir: Path, lengths: list[str], families: list[str], limit: int) -> tuple[list[dict], dict]:
    index = json.loads((set_dir / "index.json").read_text())
    rows = [r for r in index["items"] if r["length"] in lengths and r["family"] in families]
    rows.sort(key=lambda r: (r["index"], P.FAMILY_INDEX[r["family"]], P.LENGTH_INDEX[r["length"]]))
    if limit:
        keep = sorted({r["index"] for r in rows})[:limit]
        rows = [r for r in rows if r["index"] in keep]
    return rows, index


def load_item(P, set_dir: Path, row: dict) -> dict:
    path = set_dir / row["file"]
    if P.sha256_file(path) != row["file_sha256"]:
        sys.exit(f"item file changed since generation: {path}")
    return json.loads(path.read_text())


def run_unit(P, T, a, client, model: str, key: str | None, item: dict, unit_id: str, stop: threading.Event, log, texts_dir: Path) -> bool:
    session = uuid.uuid4().hex[:12]
    users = [T.single_content(item)]
    seed0 = seed_for_3c(P, a.rep, item["family"], item["index"], item["length"])
    nonce = client.nonce_for(f"{SALT_3C}:single", seed0)
    messages = [{"role": "user", "content": users[0]}]
    body = P.chat_body(model, client.with_nonce(messages, nonce), a.max_tokens, seed0)
    rec = P.stream_chat(a.base_url, body, key=key, read_timeout=a.timeout, stop_event=stop)
    content, reasoning = rec.pop("content"), rec.pop("reasoning")
    out = {
        "unit": unit_id,
        "session": session,
        "item": item["id"],
        "family": item["family"],
        "index": item["index"],
        "length": item["length"],
        "mode": "single",
        "parts": 0,
        "rep": a.rep,
        "turn": 0,
        "final": True,
        "seed": seed0,
        "nonce": nonce,
        "max_tokens": body["max_tokens"],
        "content_sha256": P.sha256_text(content),
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_class": P.finish_class(rec),
        **rec,
    }
    out["score"] = T.score(item["answer"], content) if rec["status"] == "ok" else None
    log.write(out)
    texts = [{"turn": 0, "content": content, "reasoning": reasoning}]
    with gzip.open(texts_dir / f"{unit_id}.{session}.json.gz", "wt") as f:
        json.dump({"unit": unit_id, "session": session, "turns": texts}, f)
    if rec["status"] != "ok":
        stop.set()
        print(f"  {unit_id}: {rec['status']} {rec['error_kind']} {rec['error']}", flush=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drivers-dir", required=True, help="Path containing probelib_dc.py and tasklib_dc.py")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run-tag", default="p3c")
    ap.add_argument("--set", default="measured")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--lengths", nargs="+", default=["4k", "128k"])
    ap.add_argument("--families", nargs="+", default=["ledger", "join", "trace"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out-root", required=True, help="Output root (writes tasks/<arm>/<run-tag>/single-rep<rep>/)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(a.drivers_dir).resolve()))
    import probelib_dc as P  # type: ignore
    import tasklib_dc as T  # type: ignore

    set_dir = P.ITEMS / a.set
    rows, index = load_units(P, set_dir, a.lengths, a.families, a.limit)
    sub = f"single-rep{a.rep}"
    out = Path(a.out_root) / "tasks" / a.arm / a.run_tag / sub
    print(
        f"{a.arm} {a.run_tag} {sub}: {len(rows)} items ({a.lengths}), "
        f"{sum(r['content_tokens'] for r in rows):,} content tokens, concurrency {a.concurrency}",
        flush=True,
    )

    if a.dry_run:
        client = P.stage0_client()
        # Full collision check across reps 0..7 (probelib_dc) + reps 8..11 (seed_for_3c)
        prev_seeds = {
            P.seed_for("single", rep, r["family"], r["index"], r["length"])
            for rep in range(8)
            for r in rows
        }
        new_seeds = {
            seed_for_3c(P, rep, r["family"], r["index"], r["length"])
            for rep in range(8, 12)
            for r in rows
        }
        expected_new = 4 * len(rows)
        assert len(new_seeds) == expected_new, f"internal collision in reps 8-11: {len(new_seeds)} != {expected_new}"
        overlap = prev_seeds & new_seeds
        assert not overlap, f"collision with reps 0-7: {len(overlap)}"
        for r in rows[:4]:
            s = seed_for_3c(P, a.rep, r["family"], r["index"], r["length"])
            print(f"  {r['id']}-{r['length']}: seed {s}, nonce {client.nonce_for(f'{SALT_3C}:single', s)}")
        print(
            f"SEED CHECK PASSED: reps 0-7 ({len(prev_seeds)} seeds in [{min(prev_seeds)}..{max(prev_seeds)}]) "
            f"disjoint from reps 8-11 ({len(new_seeds)} seeds in [{min(new_seeds)}..{max(new_seeds)}]); 0 collisions."
        )
        return 0

    key = P.read_key(a.api_key_file)
    client = P.stage0_client()
    doc = P.models_doc(a.base_url, key)
    model = doc["data"][0]["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts").mkdir(exist_ok=True)
    done = {r["unit"] for r in P.read_jsonl(out / "requests.jsonl") if r.get("final") and r["status"] == "ok"}
    todo = [r for r in rows if f"{r['id']}-{r['length']}" not in done]
    if not todo:
        print(f"skip {a.arm}/{a.run_tag}/{sub}: all {len(rows)} items already completed", flush=True)
        return 0

    log = P.JsonlWriter(out / "requests.jsonl")
    report = {
        "arm": a.arm,
        "run_tag": a.run_tag,
        "mode": "single",
        "parts": 0,
        "rep": a.rep,
        "set": a.set,
        "lengths": a.lengths,
        "args": vars(a),
        "model": model,
        "models_doc": doc,
        "items_index_sha256": P.sha256_file(set_dir / "index.json"),
        "generator_sha256": index["generator_sha256"],
        "driver_sha256": P.sha256_file(__file__),
        "probelib_sha256": P.sha256_file(P.__file__),
        "tasklib_sha256": P.sha256_file(T.__file__),
        "settings": {
            "temperature": P.TEMPERATURE,
            "top_p": P.TOP_P,
            "top_k": P.TOP_K,
            "chat_template_kwargs": P.TEMPLATE_KWARGS,
            "seed_base_3c": SEED_BASE_3C,
            "salt_3c": SALT_3C,
        },
        "started_at": P.now_iso(),
    }

    stop = threading.Event()
    t0 = time.monotonic()
    failed = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = []
        for r in todo:
            item = load_item(P, set_dir, r)
            uid = f"{r['id']}-{r['length']}"
            futs.append((uid, ex.submit(run_unit, P, T, a, client, model, key, item, uid, stop, log, out / "texts")))
        for idx, (uid, fut) in enumerate(futs, 1):
            ok = fut.result()
            if not ok:
                failed += 1
            print(f"  [{idx}/{len(futs)}] {uid} {'ok' if ok else 'FAIL'} ({time.monotonic() - t0:.0f} s)", flush=True)

    report["finished_at"] = P.now_iso()
    report["elapsed_s"] = round(time.monotonic() - t0, 2)
    report["failed"] = failed
    P.write_json_atomic(out / "report.json", report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
