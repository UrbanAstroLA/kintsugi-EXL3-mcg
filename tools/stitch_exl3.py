#!/usr/bin/env python3
"""
Fast, zero-dependency Hugging Face safetensors stitcher for projection-mixed
EXL3-MCG routed experts on GLM-5.3-Flash (K in {3, 4}, BF16 dense core).

Design:
- Writes 1 shard for the BF16 non-expert core (`core.safetensors`, ~10 GiB) and
  1 shard per MoE layer (`layer-03.safetensors` .. `layer-45.safetensors`,
  ~2.9-3.6 GiB each).
- Supports incremental hardlinking (`--reuse-from <existing_stitched_dir>`):
  any layer whose bit-allocation hash and tensor sources match the base dir is
  hardlinked in milliseconds, rewriting only modified layers.
- Enforces CuteAFD v6.0.0 (`exl3_contract.rs`) and vLLM v0.9.2
  (`port-exl3-projection-mixed-glm53.py`) invariants at build time.
- Includes `--verify` mode to check shapes, dtypes, byte budget, and SHA-256
  fidelity against source checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
)
TENSOR_SUFFIXES = (".trellis", ".suh", ".svh", ".mcg")
EXL3_MCG_MULTIPLIER = 3417055213
COPY_CHUNK = 16 * 1024 * 1024  # 16 MiB streaming buffer

NON_SAFETENSOR_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class SafetensorsReader:
    """Caches parsed safetensors headers for a checkpoint directory."""

    def __init__(self, snap_dir: Path) -> None:
        self.snap_dir = snap_dir
        idx_path = snap_dir / "model.safetensors.index.json"
        self.index = json.loads(idx_path.read_text())
        self.weight_map: Dict[str, str] = self.index["weight_map"]
        self._headers: Dict[str, Tuple[int, dict]] = {}
        self._qcfg: Optional[dict] = None

    @property
    def quantize_config(self) -> dict:
        if self._qcfg is None:
            self._qcfg = json.loads((self.snap_dir / "quantize_config.json").read_text())
        return self._qcfg

    def _get_header(self, shard_name: str) -> Tuple[int, dict]:
        if shard_name not in self._headers:
            path = self.snap_dir / shard_name
            with path.open("rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self._headers[shard_name] = (8 + hlen, hdr)
        return self._headers[shard_name]

    def tensor_info(self, tname: str) -> Tuple[Path, int, int, str, List[int]]:
        shard_name = self.weight_map[tname]
        data_start, hdr = self._get_header(shard_name)
        meta = hdr[tname]
        s, e = meta["data_offsets"]
        return self.snap_dir / shard_name, data_start + s, e - s, meta["dtype"], meta["shape"]

    def tensor_sha256(self, tname: str) -> str:
        path, offset, nbytes, _, _ = self.tensor_info(tname)
        h = hashlib.sha256()
        with path.open("rb") as f:
            f.seek(offset)
            rem = nbytes
            while rem > 0:
                chunk = f.read(min(COPY_CHUNK, rem))
                if not chunk:
                    raise EOFError(f"Unexpected EOF reading {tname} from {path}")
                h.update(chunk)
                rem -= len(chunk)
        return h.hexdigest()


def write_safetensors_shard(
    out_path: Path,
    entries: List[Tuple[str, SafetensorsReader]],
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """Streams tensors from source readers into a single safetensors file."""
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata

    plan: List[Tuple[str, Path, int, int]] = []
    cur = 0
    for tname, reader in entries:
        src_path, src_off, nbytes, dtype, shape = reader.tensor_info(tname)
        header[tname] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cur, cur + nbytes],
        }
        plan.append((tname, src_path, src_off, nbytes))
        cur += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment for clean mmap alignment.
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    open_files: Dict[Path, object] = {}
    try:
        with tmp_path.open("wb") as out_f:
            out_f.write(struct.pack("<Q", len(header_bytes)))
            out_f.write(header_bytes)
            for _, src_path, src_off, nbytes in plan:
                src_f = open_files.get(src_path)
                if src_f is None:
                    src_f = src_path.open("rb")
                    open_files[src_path] = src_f
                src_f.seek(src_off)
                rem = nbytes
                while rem > 0:
                    buf = src_f.read(min(COPY_CHUNK, rem))
                    if not buf:
                        raise EOFError(f"Unexpected EOF copying from {src_path}")
                    out_f.write(buf)
                    rem -= len(buf)
        tmp_path.replace(out_path)
    finally:
        for f in open_files.values():
            f.close()
        if tmp_path.exists():
            tmp_path.unlink()
    return cur


def parse_override_band(spec: str) -> Tuple[int, int, int, Optional[str]]:
    """Parse '<start>-<end>:<bits>[:<role>]', e.g. '3-8:4' or '3-8:3:down_proj'."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid --override-band '{spec}'; expected start-end:bits[:role]")
    rng = parts[0].split("-")
    if len(rng) == 1:
        lo = hi = int(rng[0])
    elif len(rng) == 2:
        lo, hi = int(rng[0]), int(rng[1])
    else:
        raise ValueError(f"Invalid layer range in '{spec}'")
    bits = int(parts[1])
    if bits not in (3, 4):
        raise ValueError(f"Unsupported bits={bits} in '{spec}'; must be 3 or 4")
    role = parts[2] if len(parts) == 3 else None
    if role is not None and role not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(f"Invalid role '{role}' in '{spec}'")
    return lo, hi, bits, role


def build_bit_map(
    k325_qcfg: dict,
    preset: str,
    map_json: Optional[Path],
    overrides: List[str],
) -> Dict[str, int]:
    ts = k325_qcfg["tensor_storage"]
    stems = sorted(ts.keys())
    if map_json is not None:
        raw = json.loads(map_json.read_text())
        if set(raw.keys()) != set(stems):
            raise ValueError(
                f"map-json keys ({len(raw)}) do not match expected {len(stems)} routed expert stems"
            )
        bmap = {k: int(raw[k]) for k in stems}
    elif preset == "k325-native":
        bmap = {k: int(ts[k]["bits_per_weight"]) for k in stems}
    elif preset == "all-k3":
        bmap = {k: 3 for k in stems}
    elif preset == "all-k4":
        bmap = {k: 4 for k in stems}
    else:
        raise ValueError(f"Unknown preset '{preset}'")

    for ov in overrides:
        lo, hi, bits, role_filter = parse_override_band(ov)
        for stem in stems:
            m = EXPERT_RE.match(stem)
            if not m:
                continue
            layer = int(m.group(1))
            role = m.group(3)
            if lo <= layer <= hi and (role_filter is None or role == role_filter):
                bmap[stem] = bits

    # Validate invariant: every layer uses only {3}, {4}, or {3, 4}
    by_layer: Dict[int, set] = defaultdict(set)
    for stem, b in bmap.items():
        if b not in (3, 4):
            raise ValueError(f"Invalid bits={b} for {stem}; must be 3 or 4")
        m = EXPERT_RE.match(stem)
        if not m:
            raise ValueError(f"Unexpected non-expert stem in tensor_storage: {stem}")
        by_layer[int(m.group(1))].add(b)
    for layer, bset in sorted(by_layer.items()):
        if not bset.issubset({3, 4}):
            raise ValueError(f"Layer {layer} violates consecutive-tier rule: {bset}")
    return bmap


def layer_signature(
    layer: int,
    stems: List[str],
    bmap: Dict[str, int],
    k3_src_id: str,
    k4_src_id: str,
) -> str:
    h = hashlib.sha256()
    h.update(f"layer={layer}|k3={k3_src_id}|k4={k4_src_id}\n".encode())
    for stem in stems:
        h.update(f"{stem}={bmap[stem]}\n".encode())
    return h.hexdigest()


def core_signature(core_src_id: str, core_tensors: List[str]) -> str:
    h = hashlib.sha256()
    h.update(f"core={core_src_id}|n={len(core_tensors)}\n".encode())
    for t in core_tensors:
        h.update(t.encode() + b"\n")
    return h.hexdigest()


def stitch_checkpoint(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    core_dir = Path(args.core_src).resolve()
    k3_dir = Path(args.k3_src).resolve()
    k4_dir = Path(args.k4_src).resolve()
    out_dir = Path(args.out).resolve()
    reuse_dir = Path(args.reuse_from).resolve() if args.reuse_from else None

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {out_dir} (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    core_reader = SafetensorsReader(core_dir)
    k3_reader = core_reader if k3_dir == core_dir else SafetensorsReader(k3_dir)
    k4_reader = core_reader if k4_dir == core_dir else SafetensorsReader(k4_dir)

    base_qcfg = core_reader.quantize_config
    bmap = build_bit_map(base_qcfg, args.map_preset, Path(args.map_json) if args.map_json else None, args.override_band)

    # Group routed expert stems by layer
    layer_stems: Dict[int, List[str]] = defaultdict(list)
    expert_tensor_names = set()
    for stem in sorted(bmap.keys()):
        layer = int(EXPERT_RE.match(stem).group(1))
        layer_stems[layer].append(stem)
        for suf in TENSOR_SUFFIXES:
            expert_tensor_names.add(stem + suf)

    core_tensors = sorted(t for t in core_reader.weight_map.keys() if t not in expert_tensor_names)
    layers_sorted = sorted(layer_stems.keys())

    # Load reuse signatures if available
    reuse_sigs: Dict[str, str] = {}
    reuse_index: Optional[dict] = None
    if reuse_dir and (reuse_dir / "stitch_manifest.json").exists():
        rm = json.loads((reuse_dir / "stitch_manifest.json").read_text())
        reuse_sigs = rm.get("shard_signatures", {})
        if (reuse_dir / "model.safetensors.index.json").exists():
            reuse_index = json.loads((reuse_dir / "model.safetensors.index.json").read_text())

    new_sigs: Dict[str, str] = {}
    weight_map: Dict[str, str] = {}
    total_size = 0
    reused_shards = 0
    written_shards = 0

    def shard_tensor_bytes(shard_path: Path) -> int:
        with shard_path.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        return sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in hdr.items() if k != "__metadata__")

    # 1. Core BF16 shard
    core_shard = "core.safetensors"
    csig = core_signature(str(core_dir), core_tensors)
    new_sigs[core_shard] = csig
    core_out = out_dir / core_shard
    if reuse_dir and reuse_sigs.get(core_shard) == csig and (reuse_dir / core_shard).exists():
        if core_out.exists():
            core_out.unlink()
        os.link(reuse_dir / core_shard, core_out)
        total_size += shard_tensor_bytes(core_out)
        reused_shards += 1
    else:
        total_size += write_safetensors_shard(
            core_out,
            [(t, core_reader) for t in core_tensors],
            metadata={"format": "pt"},
        )
        written_shards += 1
    for t in core_tensors:
        weight_map[t] = core_shard

    # 2. Per-layer MoE EXL3 shards
    k3_qts = k3_reader.quantize_config["tensor_storage"]
    k4_qts = k4_reader.quantize_config["tensor_storage"]
    new_tensor_storage: Dict[str, dict] = {}

    for layer in layers_sorted:
        stems = layer_stems[layer]
        shard_name = f"layer-{layer:02d}.safetensors"
        lsig = layer_signature(layer, stems, bmap, str(k3_dir), str(k4_dir))
        new_sigs[shard_name] = lsig
        shard_out = out_dir / shard_name

        # Build tensor_storage entries for this layer
        entries: List[Tuple[str, SafetensorsReader]] = []
        for stem in stems:
            bits = bmap[stem]
            src_reader = k3_reader if bits == 3 else k4_reader
            src_entry = k3_qts[stem] if bits == 3 else k4_qts[stem]
            if int(src_entry["bits_per_weight"]) != bits:
                raise ValueError(
                    f"Source for {stem} has bits_per_weight={src_entry['bits_per_weight']}, expected {bits}"
                )
            if int(src_entry["mcg_multiplier"]) != EXL3_MCG_MULTIPLIER:
                raise ValueError(f"Unexpected mcg_multiplier on {stem}: {src_entry['mcg_multiplier']}")
            # Normalize tensor_storage entry to the exact K3.25-v1 schema
            stored = {}
            for suf in TENSOR_SUFFIXES:
                tname = stem + suf
                stored[tname] = src_entry["stored_tensors"][tname]
                entries.append((tname, src_reader))
                weight_map[tname] = shard_name
            new_tensor_storage[stem] = {
                "stored_tensors": stored,
                "quant_format": "exl3",
                "bits_per_weight": bits,
                "mcg_multiplier": EXL3_MCG_MULTIPLIER,
            }

        if reuse_dir and reuse_sigs.get(shard_name) == lsig and (reuse_dir / shard_name).exists():
            if shard_out.exists():
                shard_out.unlink()
            os.link(reuse_dir / shard_name, shard_out)
            total_size += shard_tensor_bytes(shard_out)
            reused_shards += 1
        else:
            total_size += write_safetensors_shard(shard_out, entries, metadata={"format": "pt"})
            written_shards += 1

    # 3. Copy auxiliary tokenizer / template / generation files
    for fname in NON_SAFETENSOR_FILES:
        src = core_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    # 4. Write model.safetensors.index.json
    index_obj = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index_obj, indent=2) + "\n")

    # 5. Write quantize_config.json, quantization_config.json, and config.json
    # Keep strictly allowlisted top-level keys (CuteAFD v6 exl3_contract.rs:219-227).
    bit_counts = Counter(bmap.values())
    k4_by_layer = Counter()
    for stem, b in bmap.items():
        if b == 4:
            k4_by_layer[int(EXPERT_RE.match(stem).group(1))] += 1

    stitch_meta = {
        "stitcher": "kintsugi-EXL3-mcg/tools/stitch_exl3.py",
        "map_preset": args.map_preset if not args.map_json else "custom-json",
        "overrides": list(args.override_band),
        "k3_count": bit_counts[3],
        "k4_count": bit_counts[4],
        "effective_expert_bpw": round((3 * bit_counts[3] + 4 * bit_counts[4]) / len(bmap), 6),
    }

    qcfg_out = {
        "bits": min(bit_counts.keys()),
        "group_size": -1,
        "desc_act": False,
        "lm_head": False,
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "pack_dtype": "int32",
        "codebook": "mcg",
        "out_scales": "auto",
        "module_include": base_qcfg["module_include"],
        "meta": stitch_meta,
        "tensor_storage": new_tensor_storage,
    }
    qcfg_text = json.dumps(qcfg_out, indent=2) + "\n"
    (out_dir / "quantize_config.json").write_text(qcfg_text)
    (out_dir / "quantization_config.json").write_text(qcfg_text)

    cfg = json.loads((core_dir / "config.json").read_text())
    cfg["quantization_config"] = {k: v for k, v in qcfg_out.items() if k != "tensor_storage"}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # 6. Write stitch_manifest.json (for fast incremental hardlink reuse & audit)
    manifest = {
        "total_size_bytes": total_size,
        "total_tensors": len(weight_map),
        "core_tensors": len(core_tensors),
        "expert_projections": len(bmap),
        "k3_projections": bit_counts[3],
        "k4_projections": bit_counts[4],
        "effective_expert_bpw": stitch_meta["effective_expert_bpw"],
        "k4_by_layer": {f"L{l:02d}": k4_by_layer[l] for l in layers_sorted},
        "reused_shards": reused_shards,
        "written_shards": written_shards,
        "shard_signatures": new_sigs,
    }
    (out_dir / "stitch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    dt = time.monotonic() - t0
    print(
        f"Stitched {out_dir.name}: {len(weight_map)} tensors, "
        f"total_size={total_size:,} B ({total_size / (1024**3):.2f} GiB), "
        f"K3={bit_counts[3]} K4={bit_counts[4]} ({stitch_meta['effective_expert_bpw']:.4f} bpw), "
        f"shards written={written_shards} reused={reused_shards} in {dt:.1f}s"
    )


def verify_checkpoint(args: argparse.Namespace) -> None:
    out_dir = Path(args.verify_dir).resolve()
    core_dir = Path(args.core_src).resolve()
    k3_dir = Path(args.k3_src).resolve()
    k4_dir = Path(args.k4_src).resolve()

    out_reader = SafetensorsReader(out_dir)
    core_reader = SafetensorsReader(core_dir)
    k3_reader = core_reader if k3_dir == core_dir else SafetensorsReader(k3_dir)
    k4_reader = core_reader if k4_dir == core_dir else SafetensorsReader(k4_dir)

    out_qcfg = out_reader.quantize_config
    ts = out_qcfg["tensor_storage"]
    idx_total = out_reader.index["metadata"]["total_size"]
    core_total = core_reader.index["metadata"]["total_size"]
    delta_bytes = idx_total - core_total
    delta_pct = 100.0 * delta_bytes / core_total

    # 1. Exhaustive header check: every tensor's dtype, shape, and byte length matches its source
    expert_tensors = set()
    header_mismatches = 0
    for stem, entry in ts.items():
        bits = int(entry["bits_per_weight"])
        src_reader = k3_reader if bits == 3 else k4_reader
        for suf in TENSOR_SUFFIXES:
            tname = stem + suf
            expert_tensors.add(tname)
            _, _, ob, od, os_ = out_reader.tensor_info(tname)
            _, _, sb, sd, ss = src_reader.tensor_info(tname)
            if (ob, od, os_) != (sb, sd, ss):
                header_mismatches += 1
                if header_mismatches <= 5:
                    print(f"MISMATCH expert {tname}: out={(ob, od, os_)} src={(sb, sd, ss)}")

    core_tensors = sorted(t for t in out_reader.weight_map.keys() if t not in expert_tensors)
    for tname in core_tensors:
        _, _, ob, od, os_ = out_reader.tensor_info(tname)
        _, _, sb, sd, ss = core_reader.tensor_info(tname)
        if (ob, od, os_) != (sb, sd, ss):
            header_mismatches += 1
            if header_mismatches <= 5:
                print(f"MISMATCH core {tname}: out={(ob, od, os_)} src={(sb, sd, ss)}")

    # 2. Content SHA-256 verification (sampled or full)
    sha_mismatches = 0
    checked_sha = 0
    stems_sorted = sorted(ts.keys())
    if args.full_sha:
        check_stems = stems_sorted
        check_core = core_tensors
    else:
        step = max(1, len(stems_sorted) // max(1, args.sample_experts))
        check_stems = stems_sorted[::step]
        cstep = max(1, len(core_tensors) // max(1, args.sample_core))
        check_core = core_tensors[::cstep]

    for stem in check_stems:
        bits = int(ts[stem]["bits_per_weight"])
        src_reader = k3_reader if bits == 3 else k4_reader
        for suf in TENSOR_SUFFIXES:
            tname = stem + suf
            if out_reader.tensor_sha256(tname) != src_reader.tensor_sha256(tname):
                sha_mismatches += 1
            checked_sha += 1

    for tname in check_core:
        if out_reader.tensor_sha256(tname) != core_reader.tensor_sha256(tname):
            sha_mismatches += 1
        checked_sha += 1

    print(
        f"Verify {out_dir.name}: tensors={len(out_reader.weight_map)} "
        f"(core={len(core_tensors)}, expert_stems={len(ts)}), "
        f"total_size={idx_total:,} B (delta vs core_src: {delta_bytes:+,} B / {delta_pct:+.6f}%), "
        f"header_mismatches={header_mismatches}, sha_checked={checked_sha}, sha_mismatches={sha_mismatches}"
    )
    if header_mismatches or sha_mismatches:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Stitch and verify projection-mixed EXL3-MCG checkpoints.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Stitch a mixed K3/K4 checkpoint.")
    pb.add_argument("--core-src", required=True, help="BF16-core checkpoint snapshot (also default K3.25 map source).")
    pb.add_argument("--k3-src", required=True, help="K3 EXL3 source snapshot.")
    pb.add_argument("--k4-src", required=True, help="K4 EXL3 source snapshot.")
    pb.add_argument("--out", required=True, help="Output checkpoint directory.")
    pb.add_argument("--map-preset", default="k325-native", choices=("k325-native", "all-k3", "all-k4"))
    pb.add_argument("--map-json", default=None, help="Optional JSON mapping each expert stem to 3 or 4.")
    pb.add_argument(
        "--override-band",
        action="append",
        default=[],
        help="Repeatable band override: 'start-end:bits[:role]', e.g. '3-8:4' or '3-8:3:down_proj'.",
    )
    pb.add_argument("--reuse-from", default=None, help="Existing stitched dir to hardlink unchanged shards from.")
    pb.add_argument("--overwrite", action="store_true", help="Allow writing into an existing directory.")

    pv = sub.add_parser("verify", help="Verify a stitched checkpoint against its source snapshots.")
    pv.add_argument("--verify-dir", required=True, help="Stitched checkpoint directory to verify.")
    pv.add_argument("--core-src", required=True)
    pv.add_argument("--k3-src", required=True)
    pv.add_argument("--k4-src", required=True)
    pv.add_argument("--sample-experts", type=int, default=200, help="Number of expert stems to SHA-256 check.")
    pv.add_argument("--sample-core", type=int, default=100, help="Number of core tensors to SHA-256 check.")
    pv.add_argument("--full-sha", action="store_true", help="SHA-256 check all 150,226 tensors.")

    args = p.parse_args()
    if args.cmd == "build":
        stitch_checkpoint(args)
    else:
        verify_checkpoint(args)


if __name__ == "__main__":
    main()
