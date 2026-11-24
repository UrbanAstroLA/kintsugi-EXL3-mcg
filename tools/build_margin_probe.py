#!/usr/bin/env python3
"""
Build frozen two-component teacher-forced margin probe suites (`D_margin_build`
and `D_margin_heldout`) from verified 100%-correct finished K4 trajectories on
the 60-item 127K context-probe benchmark (20 join, 20 ledger, 20 trace).

Stratified split (frozen deterministically before any GPU run):
- `build`   (24 items: 8 join, 8 ledger, 8 trace): used in Phases 1 and 2.
- `heldout` (36 items: 12 join, 12 ledger, 12 trace): untouched until Phase 3.
Both splits are balanced across historical K3.25 127K cap-hit items.

Per item, records exact `score_positions` and `want` token ids for:
- Component A (Answer values): exact value tokens on the final `ANSWER: {...}` line.
- Component B (Termination): the `</think>` closing token (154842) plus the 8
  immediately preceding reasoning tokens.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THINK_OPEN_ID = 154841
THINK_CLOSE_ID = 154842
ANSWER_LINE_RE = re.compile(r"^\s*ANSWER:\s*(\{.*\})\s*$", re.M)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def render_single_prompt(item: dict) -> str:
    return item["preamble"] + "\n\n" + item["sep"].join(item["records"]) + "\n\n" + item["question"]


def load_k325_unfinished_counts(runs_root: Path) -> Counter:
    counts: Counter = Counter()
    patterns = [
        runs_root / "2026-11-15-fp8-core-contrasts/tasks/k325-fp8c",
        runs_root / "2026-11-10-context-probe/tasks/k325-fp8",
    ]
    for base in patterns:
        if not base.exists():
            continue
        for p in sorted(base.rglob("requests.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("final", True) and str(r.get("length")) == "128k" and r.get("finish_class") != "stop":
                    counts[r["item"]] += 1
    return counts


def find_perfect_k4_trajectory(item_id: str, runs_root: Path, source_mode: str = "default") -> Tuple[dict, str]:
    """Find a 100%-correct (or highest-scoring) finished trajectory at 128k."""
    if source_mode == "neutral-public":
        sources = [
            ("exlr8-k4", runs_root / "2026-11-10-context-probe/tasks/exlr8-k4/m0"),
            ("k325-fp8c", runs_root / "2026-11-15-fp8-core-contrasts/tasks/k325-fp8c/m0"),
        ]
    else:
        sources = [
            ("v6-bf16core", runs_root / "2026-11-20-v6-context/tasks/v6-bf16core/c0"),
            ("v6-bf16core-bridge", runs_root / "2026-11-19-v6-bridge/tasks/v6-bf16core/b0"),
            ("v6-hqa", runs_root / "2026-11-19-v6-bridge/tasks/v6-hqa/b0"),
            ("fp8c", runs_root / "2026-11-15-fp8-core-contrasts/tasks/fp8c/m0"),
            ("hqa", runs_root / "2026-11-10-context-probe/tasks/hqa/m0"),
            ("anchor", runs_root / "2026-11-10-context-probe/tasks/anchor/m0"),
        ]
    best: Optional[Tuple[float, dict, str]] = None
    for src_label, base in sources:
        if not base.exists():
            continue
        for rep_dir in sorted(base.glob("single-rep*")):
            req_p = rep_dir / "requests.jsonl"
            if not req_p.exists():
                continue
            for line in req_p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("item") != item_id or str(r.get("length")) != "128k":
                    continue
                if r.get("finish_class") != "stop":
                    continue
                sc = r.get("score") or {}
                fields = sc.get("fields") or 0
                correct = sc.get("correct") or 0
                if not fields:
                    continue
                frac = correct / fields
                texts = sorted((rep_dir / "texts").glob(f"{item_id}-128k.*.json.gz"))
                if not texts:
                    continue
                payload = json.loads(gzip.open(texts[0], "rt").read())
                t0 = payload["turns"][0]
                if not t0.get("reasoning") or not t0.get("content"):
                    continue
                if not ANSWER_LINE_RE.search(t0["content"]):
                    continue
                cand = {
                    "reasoning": t0["reasoning"],
                    "content": t0["content"].strip(),
                    "nonce": r.get("nonce", ""),
                    "source_arm": src_label,
                    "source_rep": rep_dir.name,
                    "source_text_file": texts[0].name,
                    "source_score": frac,
                }
                if frac == 1.0:
                    return cand, src_label
                if best is None or frac > best[0]:
                    best = (frac, cand, src_label)
    if best is not None:
        return best[1], best[2]
    raise RuntimeError(f"No finished K4 128k trajectory found for {item_id}")


def find_value_char_spans(content: str, expected_answer: dict) -> List[Tuple[int, int, str]]:
    """Locate character spans of every correctly answered VALUE on the final ANSWER: {...} line."""
    matches = list(ANSWER_LINE_RE.finditer(content))
    if not matches:
        raise ValueError("No ANSWER: {...} line in content")
    m = matches[-1]
    json_start = m.start(1)
    json_str = m.group(1)
    parsed = json.loads(json_str)

    spans: List[Tuple[int, int, str]] = []
    for key, want in expected_answer.items():
        if key not in parsed:
            continue
        val = parsed[key]
        if str(val).strip() != str(want).strip():
            continue
        if isinstance(val, str):
            pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*"(' + re.escape(val) + r')"')
        else:
            pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*(' + re.escape(str(val)) + r')(?=[,\}\s])')
        vm = pat.search(json_str)
        if not vm:
            continue
        spans.append((json_start + vm.start(1), json_start + vm.end(1), key))
    if not spans:
        raise ValueError(f"No matching answer value spans found in {json_str!r}")
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, help="Root directory containing prior context-probe task runs.")
    ap.add_argument("--items-dir", required=True, help="Directory containing measured item JSON files by family.")
    ap.add_argument("--tokenizer", required=True, help="Path to GLM-5.3-Flash tokenizer.json.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--build-per-family", type=int, default=8)
    ap.add_argument("--tail-tokens", type=int, default=8)
    ap.add_argument("--source-mode", default="default", choices=("default", "neutral-public"))
    args = ap.parse_args()

    from tokenizers import Tokenizer  # type: ignore

    runs_root = Path(args.runs_root)
    items_dir = Path(args.items_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items").mkdir(exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    k325_unf = load_k325_unfinished_counts(runs_root)

    # Load all 60 128k items grouped by family
    by_family: Dict[str, List[Tuple[int, str, Path]]] = {"join": [], "ledger": [], "trace": []}
    for fam in ("join", "ledger", "trace"):
        for p in sorted((items_dir / fam).glob("*-128k.json")):
            item_id = p.name.replace("-128k.json", "")
            by_family[fam].append((k325_unf[item_id], item_id, p))
        assert len(by_family[fam]) == 20, f"Expected 20 items in {fam}, got {len(by_family[fam])}"

    # Deterministic stratified split: sort each family by (-k325_unf, item_id) and alternate
    # indices so both `build` (8/family = 24) and `heldout` (12/family = 36) receive balanced
    # K3.25 cap-hit exposure.
    build_ids: List[str] = []
    heldout_ids: List[str] = []
    item_paths: Dict[str, Path] = {}
    for fam in ("join", "ledger", "trace"):
        ranked = sorted(by_family[fam], key=lambda x: (-x[0], x[1]))
        b_fam, h_fam = [], []
        for idx, (unf_cnt, item_id, p) in enumerate(ranked):
            item_paths[item_id] = p
            # Alternate until build hits 8 (even indices 0,2,4,6,8,10,12,14 -> 8 items)
            if idx % 2 == 0 and len(b_fam) < args.build_per_family:
                b_fam.append(item_id)
            else:
                h_fam.append(item_id)
        assert len(b_fam) == args.build_per_family and len(h_fam) == 20 - args.build_per_family
        build_ids.extend(b_fam)
        heldout_ids.extend(h_fam)

    manifest_items = []
    for split, id_list in (("build", build_ids), ("heldout", heldout_ids)):
        for item_id in id_list:
            item_obj = json.loads(item_paths[item_id].read_text())
            traj, src_arm = find_perfect_k4_trajectory(item_id, runs_root, source_mode=args.source_mode)

            user_text = render_single_prompt(item_obj)
            # Match exact tasklib/client nonce prepending if nonce existed on trajectory
            # Note: saved trajectory was generated with its nonce; we include the exact prefix
            # `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{user_text}<|assistant|><think>`
            prefix_text = f"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{user_text}<|assistant|><think>"
            if traj["nonce"]:
                user_with_nonce = f"Request {traj['nonce']}.\n" + user_text
                prefix_text = f"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{user_with_nonce}<|assistant|><think>"

            reasoning_text = traj["reasoning"]
            content_text = traj["content"]

            enc_prefix = tok.encode(prefix_text, add_special_tokens=False)
            enc_reason = tok.encode(reasoning_text, add_special_tokens=False)
            enc_content = tok.encode(content_text, add_special_tokens=False)

            prompt_ids = enc_prefix.ids + enc_reason.ids + [THINK_CLOSE_ID] + enc_content.ids
            pos_think_close = len(enc_prefix.ids) + len(enc_reason.ids)
            assert prompt_ids[pos_think_close] == THINK_CLOSE_ID

            # Component B positions: last `tail_tokens` of reasoning + `</think>`
            tail_start = max(len(enc_prefix.ids), pos_think_close - args.tail_tokens)
            comp_b_tail_pos = list(range(tail_start, pos_think_close))
            comp_b_close_pos = pos_think_close

            # Component A positions: exact tokens overlapping every answer VALUE span in `content_text`
            val_spans = find_value_char_spans(content_text, item_obj["answer"])
            content_base = pos_think_close + 1
            comp_a_pos: List[int] = []
            comp_a_by_field: Dict[str, List[int]] = {}
            for c_start, c_end, field_key in val_spans:
                f_pos = []
                for tok_idx, (t_s, t_e) in enumerate(enc_content.offsets):
                    if t_e > c_start and t_s < c_end:
                        f_pos.append(content_base + tok_idx)
                assert f_pos, f"No tokens overlapped span ({c_start},{c_end}) for {item_id}/{field_key}"
                comp_a_by_field[field_key] = f_pos
                comp_a_pos.extend(f_pos)
            comp_a_pos = sorted(set(comp_a_pos))

            # Combined score_positions & want map:
            # In CuteAFD ProbeSpec, `score_positions` contains the predicted token indices `pos`
            # (where row `pos` predicts `prompt_ids[pos]` from prefix `prompt_ids[:pos]`).
            all_pos = sorted(set(comp_b_tail_pos + [comp_b_close_pos] + comp_a_pos))
            want = {str(pos): [int(prompt_ids[pos])] for pos in all_pos}

            rec = {
                "item_id": item_id,
                "family": item_obj["family"],
                "split": split,
                "k325_hist_unf": int(k325_unf[item_id]),
                "source_arm": src_arm,
                "source_rep": traj["source_rep"],
                "n_tokens": len(prompt_ids),
                "n_prefix_tokens": len(enc_prefix.ids),
                "n_reasoning_tokens": len(enc_reason.ids),
                "n_content_tokens": len(enc_content.ids),
                "score_from": all_pos[0],
                "score_positions": all_pos,
                "want": want,
                "comp_a_positions": comp_a_pos,
                "comp_a_by_field": comp_a_by_field,
                "comp_b_close_position": comp_b_close_pos,
                "comp_b_tail_positions": comp_b_tail_pos,
                "prompt_ids": prompt_ids,
            }
            item_out = out_dir / "items" / f"{item_id}.json.gz"
            with gzip.open(item_out, "wt") as f:
                json.dump(rec, f, separators=(",", ":"))
            item_sha = sha256_bytes(item_out.read_bytes())

            manifest_items.append(
                {
                    "item_id": item_id,
                    "family": item_obj["family"],
                    "split": split,
                    "k325_hist_unf": int(k325_unf[item_id]),
                    "source_arm": src_arm,
                    "n_tokens": len(prompt_ids),
                    "n_comp_a": len(comp_a_pos),
                    "n_comp_b": 1 + len(comp_b_tail_pos),
                    "file": f"items/{item_id}.json.gz",
                    "sha256": item_sha,
                }
            )

    manifest = {
        "schema": "kintsugi-margin-probe-v1",
        "build_count": len(build_ids),
        "heldout_count": len(heldout_ids),
        "build_k325_unf_sum": sum(m["k325_hist_unf"] for m in manifest_items if m["split"] == "build"),
        "heldout_k325_unf_sum": sum(m["k325_hist_unf"] for m in manifest_items if m["split"] == "heldout"),
        "items": manifest_items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Built margin probe in {out_dir}: build={len(build_ids)} "
        f"(hist_k325_unf={manifest['build_k325_unf_sum']}), heldout={len(heldout_ids)} "
        f"(hist_k325_unf={manifest['heldout_k325_unf_sum']}), "
        f"mean_tokens={sum(m['n_tokens'] for m in manifest_items)/len(manifest_items):.0f}, "
        f"mean_comp_a_pos={sum(m['n_comp_a'] for m in manifest_items)/len(manifest_items):.1f}"
    )


if __name__ == "__main__":
    main()
