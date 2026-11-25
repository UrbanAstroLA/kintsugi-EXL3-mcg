#!/usr/bin/env python3
"""
Build the frozen Stage 3C-A Loop Entry/Exit Margin Probe (`probe_loop/`) from
saved 128K Context-Probe trajectories across all 4 author arms:
  - `k325-native`  (Stage 3B reps 0-7)
  - `k4-bf16core`  (Stage 3B reps 0-7)
  - `k345-b1b2`    (Stage 3B reps 0-7)
  - `exlr8-k4`     (neutral-quantizer 128K trajectories)

Prefix sets:
  1. `dump`: every 128K unfinished reply whose reasoning contains a pure verbatim
     record-dump run of >= 15 consecutive non-empty lines matching `RAW_REC_RE`.
  2. `control`: deterministically selected stratified sample of finished 128K
     replies that quote >= 3 consecutive pure record lines mid-thinking and then
     exit the run (preferring runs of length >= 11 so `M_exit(10)` is defined on
     every control; falling back to >= 3 only if a stratum has too few).

Cut positions per prefix (scored in a single prefill pass via `/v1/bench/probe`):
  - `onset` (k=0): first token of record line 1 in the run (`M_enter`).
  - `k=5`, `k=10`, `k=20`: first token of record line k+1 (`M_exit(k)`), wherever
    k+1 <= run_len. Also records the preceding newline token (`pos - 1`) for
    diagnostic `M_nl(k) = lp('\n') - lp('\n\n')`.
  - `leave` (controls only): first token of the first non-empty non-record line
    immediately following the exited run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAW_REC_RE = re.compile(r"^(courier C-|parcel P-|hub H-|\d{5}\s+[A-Z0-9-]{3})", re.I)

# Real start and exit token IDs observed empirically in GLM-5.3-Flash 128K traces.
# Included in `/v1/bench/probe` `want` at every cut position (zero extra prefill cost)
# so both `M_raw` (top_k=5 target vs best other) and `M_stick` (record-class vs best
# non-record exit token) are computed from the exact same logits.
JOIN_RECORD_START_IDS = [66384, 95115, 26596, 19280, 27649]  # cour, parcel, hub, Hub, Parcel
COMMON_EXIT_IDS = [
    73022,  # ```
    1112,   # ...
    21340,  # Maybe
    2753,   # No
    9450,   # Yes
    12888,  # Could
    8609,   # Any
    4416,   # So
    14181,  # Wait
    198,    # \n
    271,    # \n\n
    154842, # </think>
]
BASE_WANT_IDS = sorted(set(JOIN_RECORD_START_IDS + COMMON_EXIT_IDS))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def render_single_prompt(item: dict) -> str:
    return item["preamble"] + "\n\n" + item["sep"].join(item["records"]) + "\n\n" + item["question"]


def find_record_runs(reasoning: str) -> List[dict]:
    """Return all maximal runs of consecutive non-empty lines matching RAW_REC_RE."""
    lines = reasoning.splitlines(keepends=True)
    starts: List[int] = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)

    runs: List[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        if RAW_REC_RE.match(lines[i].strip()):
            j = i
            while j < n and (not lines[j].strip() or RAW_REC_RE.match(lines[j].strip())):
                j += 1
            rec_indices = [idx for idx in range(i, j) if RAW_REC_RE.match(lines[idx].strip())]
            # Find first non-empty line after run (if exited)
            leave_idx = None
            k = j
            while k < n:
                if lines[k].strip():
                    leave_idx = k
                    break
                k += 1
            runs.append(
                {
                    "first_line": rec_indices[0],
                    "last_line": rec_indices[-1],
                    "run_len": len(rec_indices),
                    "line_char_starts": [starts[idx] for idx in rec_indices],
                    "line_char_ends": [starts[idx] + len(lines[idx]) for idx in rec_indices],
                    "leave_char_start": starts[leave_idx] if leave_idx is not None else None,
                    "leave_char_end": (starts[leave_idx] + len(lines[leave_idx])) if leave_idx is not None else None,
                }
            )
            i = j
        else:
            i += 1
    return runs


def char_to_tok_idx(offsets: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
    """Return the first token index overlapping or starting at `char_pos`."""
    for idx, (s, e) in enumerate(offsets):
        if e > char_pos and s <= char_pos:
            return idx
        if s >= char_pos:
            return idx
    return None


def scan_author_dir(author: str, arm_dir: Path) -> Tuple[List[dict], List[dict]]:
    dumps: List[dict] = []
    controls: List[dict] = []
    if not arm_dir.exists():
        return dumps, controls

    for rep_dir in sorted(arm_dir.glob("single-rep*")):
        req_p = rep_dir / "requests.jsonl"
        if not req_p.exists():
            continue
        for line in req_p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("final", True) or r.get("status") != "ok" or str(r.get("length")) != "128k":
                continue
            tf = rep_dir / "texts" / f"{r['unit']}.{r['session']}.json.gz"
            if not tf.exists():
                continue
            payload = json.loads(gzip.open(tf, "rt").read())
            reas = payload["turns"][0].get("reasoning", "") or ""
            if not reas:
                continue
            runs = find_record_runs(reas)
            if r.get("finish_class") != "stop":
                long_runs = [rn for rn in runs if rn["run_len"] >= 15]
                if long_runs:
                    dumps.append(
                        {
                            "author": author,
                            "kind": "dump",
                            "item_id": str(r["item"]),
                            "family": str(r["family"]),
                            "rep": int(r["rep"]),
                            "finish_class": str(r.get("finish_class", "")),
                            "nonce": str(r.get("nonce", "")),
                            "reasoning": reas,
                            "run": long_runs[0],
                        }
                    )
            else:
                exited = [rn for rn in runs if rn["run_len"] >= 3 and rn["leave_char_start"] is not None]
                if exited:
                    # Pick the longest exited run in this finished trajectory
                    best_run = max(exited, key=lambda rn: (rn["run_len"], -rn["line_char_starts"][0]))
                    controls.append(
                        {
                            "author": author,
                            "kind": "control",
                            "item_id": str(r["item"]),
                            "family": str(r["family"]),
                            "rep": int(r["rep"]),
                            "finish_class": "stop",
                            "nonce": str(r.get("nonce", "")),
                            "reasoning": reas,
                            "run": best_run,
                        }
                    )
    return dumps, controls


def select_stratified_controls(all_controls: List[dict], per_author: int) -> List[dict]:
    """Deterministically select up to `per_author` controls per author, preferring
    run_len >= 11 (so M_exit(10) is defined) and balancing families (`join`, `ledger`)
    plus distinct items."""
    by_author: Dict[str, List[dict]] = defaultdict(list)
    for c in all_controls:
        by_author[c["author"]].append(c)

    selected: List[dict] = []
    for author in sorted(by_author):
        pool = by_author[author]
        # Prefer moderate-to-long exited runs (>=11 lines, capping preference order around
        # typical clean exits 11..60 before huge 500+ near-runaways, deterministic tiebreak)
        def sort_key(c: dict) -> tuple:
            rl = c["run"]["run_len"]
            tier = 0 if 11 <= rl <= 60 else (1 if rl > 60 else 2)
            return (tier, c["item_id"], c["rep"], -rl)

        by_fam: Dict[str, List[dict]] = defaultdict(list)
        for c in sorted(pool, key=sort_key):
            by_fam[c["family"]].append(c)

        # Round-robin across families and distinct items first
        chosen: List[dict] = []
        used_keys = set()
        used_items = set()
        fams = sorted(by_fam)
        # Pass 1: distinct items, round-robin families
        progressed = True
        while len(chosen) < per_author and progressed:
            progressed = False
            for fam in fams:
                if len(chosen) >= per_author:
                    break
                for c in by_fam[fam]:
                    ck = (c["author"], c["item_id"], c["rep"])
                    if ck not in used_keys and c["item_id"] not in used_items:
                        chosen.append(c)
                        used_keys.add(ck)
                        used_items.add(c["item_id"])
                        progressed = True
                        break
        # Pass 2: allow repeat items if fewer than `per_author` distinct items exist
        progressed = True
        while len(chosen) < per_author and progressed:
            progressed = False
            for fam in fams:
                if len(chosen) >= per_author:
                    break
                for c in by_fam[fam]:
                    ck = (c["author"], c["item_id"], c["rep"])
                    if ck not in used_keys:
                        chosen.append(c)
                        used_keys.add(ck)
                        progressed = True
                        break
        selected.extend(chosen)
    return selected


def build_prefix_record(cand: dict, item_obj: dict, tok) -> dict:
    user_text = render_single_prompt(item_obj)
    if cand["nonce"]:
        user_with_nonce = f"Request {cand['nonce']}.\n" + user_text
    else:
        user_with_nonce = user_text
    prefix_text = f"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{user_with_nonce}<|assistant|><think>"

    run = cand["run"]
    rl = run["run_len"]
    # Truncate reasoning compactly right after the last needed cut line (+1 line margin)
    if cand["kind"] == "dump":
        last_needed_idx = min(rl - 1, 21)
        cut_char_end = run["line_char_ends"][last_needed_idx]
    else:
        if rl <= 22 and run["leave_char_end"] is not None:
            cut_char_end = run["leave_char_end"]
        else:
            last_needed_idx = min(rl - 1, 21)
            cut_char_end = run["line_char_ends"][last_needed_idx]
    reas_slice = cand["reasoning"][:cut_char_end]

    enc_prefix = tok.encode(prefix_text, add_special_tokens=False)
    enc_reason = tok.encode(reas_slice, add_special_tokens=False)
    prompt_ids = enc_prefix.ids + enc_reason.ids
    base = len(enc_prefix.ids)

    # Cut map: cut_name -> token position in `prompt_ids`
    cuts: Dict[str, int] = {}
    nl_cuts: Dict[str, int] = {}

    def add_cut(name: str, char_pos: int, record_nl: bool = False) -> None:
        t_idx = char_to_tok_idx(enc_reason.offsets, char_pos)
        if t_idx is None:
            return
        pos = base + t_idx
        if pos <= base or pos >= len(prompt_ids):
            return
        cuts[name] = pos
        if record_nl and pos - 1 > base:
            nl_cuts[name] = pos - 1

    # k=0 onset (line 1 of run)
    add_cut("enter", run["line_char_starts"][0], record_nl=False)
    # k=5, 10, 20 -> start of record line k+1 (0-indexed: k)
    for k in (5, 10, 20):
        if k < rl and run["line_char_starts"][k] < cut_char_end:
            add_cut(f"exit_{k}", run["line_char_starts"][k], record_nl=True)
    # leave (controls only, if within truncated slice)
    if cand["kind"] == "control" and run["leave_char_start"] is not None and run["leave_char_start"] < cut_char_end:
        add_cut("leave", run["leave_char_start"], record_nl=True)

    all_pos = sorted(set(list(cuts.values()) + list(nl_cuts.values())))
    assert all_pos, f"No valid cut positions for {cand['author']} {cand['item_id']} r{cand['rep']}"
    want = {str(pos): sorted(set([int(prompt_ids[pos])] + BASE_WANT_IDS)) for pos in all_pos}

    prefix_id = f"{cand['kind']}__{cand['author']}__{cand['item_id']}__r{cand['rep']}"
    return {
        "prefix_id": prefix_id,
        "kind": cand["kind"],
        "author": cand["author"],
        "item_id": cand["item_id"],
        "family": cand["family"],
        "rep": cand["rep"],
        "finish_class": cand["finish_class"],
        "run_len": rl,
        "onset_char": run["line_char_starts"][0],
        "n_tokens": len(prompt_ids),
        "n_prefix_tokens": len(enc_prefix.ids),
        "n_reasoning_tokens": len(enc_reason.ids),
        "cuts": cuts,
        "nl_cuts": nl_cuts,
        "cut_tokens": {name: {"id": int(prompt_ids[pos]), "text": tok.decode([int(prompt_ids[pos])])} for name, pos in cuts.items()},
        "score_from": all_pos[0],
        "score_positions": all_pos,
        "want": want,
        "prompt_ids": prompt_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3b-tasks-dir", required=True, help="Directory with k325-native/, k4-bf16core/, k345-b1b2/ subdirs.")
    ap.add_argument("--exlr8-dir", default="", help="Optional directory containing exlr8-k4 single-rep* subdirs.")
    ap.add_argument("--items-dir", required=True, help="Directory containing measured 128k item JSONs by family.")
    ap.add_argument("--tokenizer", required=True, help="Path to GLM-5.3-Flash tokenizer.json.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--controls-per-author", type=int, default=10)
    args = ap.parse_args()

    from tokenizers import Tokenizer  # type: ignore

    tok = Tokenizer.from_file(args.tokenizer)
    items_dir = Path(args.items_dir)
    item_cache: Dict[Tuple[str, str], dict] = {}
    for fam in ("join", "ledger", "trace"):
        for p in sorted((items_dir / fam).glob("*-128k.json")):
            item_id = p.name.replace("-128k.json", "")
            item_cache[(fam, item_id)] = json.loads(p.read_text())

    stage3b = Path(args.stage3b_tasks_dir)
    sources = [
        ("k325-native", stage3b / "k325-native" / "p3b"),
        ("k4-bf16core", stage3b / "k4-bf16core" / "p3b"),
        ("k345-b1b2", stage3b / "k345-b1b2" / "p3b"),
    ]
    if args.exlr8_dir:
        sources.append(("exlr8-k4", Path(args.exlr8_dir)))

    all_dumps: List[dict] = []
    all_controls: List[dict] = []
    for author, arm_dir in sources:
        d, c = scan_author_dir(author, arm_dir)
        all_dumps.extend(d)
        all_controls.extend(c)

    sel_controls = select_stratified_controls(all_controls, per_author=args.controls_per_author)
    combined = sorted(all_dumps, key=lambda x: (x["author"], x["item_id"], x["rep"])) + sorted(
        sel_controls, key=lambda x: (x["author"], x["item_id"], x["rep"])
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items").mkdir(exist_ok=True)

    manifest_items = []
    counts_by_kind_author: Counter = Counter()
    counts_by_kind_family: Counter = Counter()
    cut_avail: Dict[str, Counter] = defaultdict(Counter)

    for cand in combined:
        item_obj = item_cache[(cand["family"], cand["item_id"])]
        rec = build_prefix_record(cand, item_obj, tok)
        item_out = out_dir / "items" / f"{rec['prefix_id']}.json.gz"
        with gzip.open(item_out, "wt") as f:
            json.dump(rec, f, separators=(",", ":"))
        item_sha = sha256_bytes(item_out.read_bytes())

        counts_by_kind_author[(rec["kind"], rec["author"])] += 1
        counts_by_kind_family[(rec["kind"], rec["family"])] += 1
        for cname in rec["cuts"]:
            cut_avail[rec["kind"]][cname] += 1

        manifest_items.append(
            {
                "prefix_id": rec["prefix_id"],
                "kind": rec["kind"],
                "author": rec["author"],
                "item_id": rec["item_id"],
                "family": rec["family"],
                "rep": rec["rep"],
                "run_len": rec["run_len"],
                "onset_char": rec["onset_char"],
                "n_tokens": rec["n_tokens"],
                "cuts": sorted(rec["cuts"].keys()),
                "file": f"items/{rec['prefix_id']}.json.gz",
                "sha256": item_sha,
            }
        )

    manifest = {
        "schema": "kintsugi-loop-probe-v1",
        "raw_rec_regex": RAW_REC_RE.pattern,
        "join_record_start_ids": JOIN_RECORD_START_IDS,
        "common_exit_ids": COMMON_EXIT_IDS,
        "total_prefixes": len(manifest_items),
        "dump_count": sum(1 for m in manifest_items if m["kind"] == "dump"),
        "control_count": sum(1 for m in manifest_items if m["kind"] == "control"),
        "control_pool_available": len(all_controls),
        "counts_by_kind_author": {f"{k}:{a}": v for (k, a), v in sorted(counts_by_kind_author.items())},
        "counts_by_kind_family": {f"{k}:{f}": v for (k, f), v in sorted(counts_by_kind_family.items())},
        "cut_availability": {k: dict(sorted(v.items())) for k, v in sorted(cut_avail.items())},
        "mean_tokens": round(sum(m["n_tokens"] for m in manifest_items) / max(1, len(manifest_items)), 1),
        "items": manifest_items,
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}, indent=2))
    print(f"manifest_sha256={sha256_bytes(man_path.read_bytes())}")


if __name__ == "__main__":
    main()
