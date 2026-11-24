#!/usr/bin/env python3
"""
Deterministic bit-allocation map generator for Phase 2 Stage 2B (`protocols/PHASE2B.md`).

Generates exact `{stem: 3|4}` JSON maps for `tools/stitch_exl3.py --map-json`
from `K3.25-v1/quantize_config.json` and `K3-v1/ds4rt-exl3-error-ledger.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
)


def parse_stem(stem: str) -> Tuple[int, int, str]:
    m = EXPERT_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized stem: {stem}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load_ledger(path: Path) -> Dict[str, Tuple[float, float]]:
    """Return {stem: (gate_sq_mass_frac, hessian_rel_err)} for all 37,152 projections."""
    out: Dict[str, Tuple[float, float]] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("record_kind") != "projection":
                continue
            mod = d["module"]
            g2 = float(d["route_evidence"]["expert_gate_squared_mass_fraction"])
            hrel = float(d["quantizer_metrics"]["hessian_weighted_relative_error"])
            out[mod] = (g2, hrel)
    return out


def make_within_band_shuffle(
    base: Dict[str, int], lo: int, hi: int, seed: int
) -> Dict[str, int]:
    """Randomly permute K4 assignments within each (layer, role) in [lo, hi], exact 0 B delta."""
    out = dict(base)
    rng = random.Random(seed)
    groups: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for stem in sorted(base.keys()):
        lay, _, role = parse_stem(stem)
        if lo <= lay <= hi:
            groups[(lay, role)].append(stem)
    for key in sorted(groups.keys()):
        stems = groups[key]
        k4_cnt = sum(1 for s in stems if base[s] == 4)
        picks = set(rng.sample(stems, k4_cnt))
        for s in stems:
            out[s] = 4 if s in picks else 3
    return out


def make_role_upgrade(base: Dict[str, int], lo: int, hi: int, role_target: str) -> Dict[str, int]:
    out = dict(base)
    for stem in base:
        lay, _, role = parse_stem(stem)
        if lo <= lay <= hi and role == role_target:
            out[stem] = 4
    return out


def make_band_override(base: Dict[str, int], ranges: List[Tuple[int, int, int]]) -> Dict[str, int]:
    out = dict(base)
    for stem in base:
        lay, _, _ = parse_stem(stem)
        for lo, hi, bits in ranges:
            if lo <= lay <= hi:
                out[stem] = bits
    return out


def top_k3_in_range(
    base: Dict[str, int],
    ledger: Dict[str, Tuple[float, float]],
    lo: int,
    hi: int,
    k: int,
    mode: str,
) -> List[str]:
    cand: List[Tuple[float, str]] = []
    for stem, bits in base.items():
        if bits != 3:
            continue
        lay, _, _ = parse_stem(stem)
        if lo <= lay <= hi:
            g2, hrel = ledger[stem]
            score = g2 if mode == "route" else (g2 * hrel)
            cand.append((score, stem))
    cand.sort(key=lambda x: (-x[0], x[1]))
    if len(cand) < k:
        raise ValueError(f"Requested {k} K3 candidates in L{lo}-L{hi}, only {len(cand)} available")
    return [stem for _, stem in cand[:k]]


def summarize(name: str, bmap: Dict[str, int], base: Dict[str, int]) -> dict:
    k4 = sum(1 for v in bmap.values() if v == 4)
    base_k4 = sum(1 for v in base.values() if v == 4)
    k3_to_k4 = sum(1 for s, v in bmap.items() if base[s] == 3 and v == 4)
    k4_to_k3 = sum(1 for s, v in bmap.items() if base[s] == 4 and v == 3)
    by_band = Counter()
    for s, v in bmap.items():
        if base[s] != v:
            lay, _, _ = parse_stem(s)
            by_band[f"B{(lay - 3) // 6}:{'3->4' if v == 4 else '4->3'}"] += 1
    return {
        "arm": name,
        "k4_count": k4,
        "delta_mib": k4 - base_k4,
        "bpw": round(3.0 + k4 / len(bmap), 4),
        "flips_3_to_4": k3_to_k4,
        "flips_4_to_3": k4_to_k3,
        "by_band": dict(sorted(by_band.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k325-qcfg", required=True)
    ap.add_argument("--k3-ledger", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    qcfg = json.loads(Path(args.k325_qcfg).read_text())
    base = {k: int(v["bits_per_weight"]) for k, v in qcfg["tensor_storage"].items()}
    assert len(base) == 37152, f"Expected 37152 stems, got {len(base)}"
    assert sum(1 for v in base.values() if v == 4) == 9072

    ledger = load_ledger(Path(args.k3_ledger))
    assert set(base.keys()).issubset(set(ledger.keys())), "Missing stems in K3 error ledger"

    maps: Dict[str, Dict[str, int]] = {}
    maps["null-shuf-b2"] = make_within_band_shuffle(base, 15, 20, seed=20261123)
    maps["null-shuf-b5"] = make_within_band_shuffle(base, 33, 38, seed=20261124)
    maps["b2-role-gate"] = make_role_upgrade(base, 15, 20, "gate_proj")
    maps["b2-role-up"] = make_role_upgrade(base, 15, 20, "up_proj")
    maps["b2-role-down"] = make_role_upgrade(base, 15, 20, "down_proj")

    hot_route = top_k3_in_range(base, ledger, 9, 20, 3888, mode="route")
    m_hr = dict(base)
    for s in hot_route:
        m_hr[s] = 4
    maps["b1b2-hot-route-3888"] = m_hr

    hot_err = top_k3_in_range(base, ledger, 9, 20, 3888, mode="routed_err")
    m_he = dict(base)
    for s in hot_err:
        m_he[s] = 4
    maps["b1b2-hot-routed-err-3888"] = m_he

    maps["iso-b0b5b6-to-b2"] = make_band_override(
        base, [(3, 8, 3), (33, 38, 3), (39, 44, 3), (15, 20, 4)]
    )

    m_iso_hot = make_band_override(base, [(3, 8, 3), (33, 38, 3), (39, 44, 3)])
    for s in hot_err:
        m_iso_hot[s] = 4
    maps["iso-b0b5b6-to-hot-b1b2"] = m_iso_hot

    maps["b1b2-up-k4"] = make_band_override(base, [(9, 20, 4)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, bmap in maps.items():
        (out_dir / f"{name}.map.json").write_text(json.dumps(bmap, sort_keys=True) + "\n")
        s = summarize(name, bmap, base)
        summaries.append(s)
        print(
            f"  {name:26s} K4={s['k4_count']:5d} dMiB={s['delta_mib']:+5d} "
            f"bpw={s['bpw']:.4f} 3->4={s['flips_3_to_4']:4d} 4->3={s['flips_4_to_3']:4d} {s['by_band']}"
        )

    (out_dir / "maps_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
