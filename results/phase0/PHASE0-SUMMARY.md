# Phase 0 Results: Stitcher Verification & CuteAFD `v6.0.0` CPU Planner (`2026-11-23`)

**Zero GPU time used.** All checks ran on CPU + NVMe using [`tools/stitch_exl3.py`](../../tools/stitch_exl3.py) and `ghcr.io/tpurtell/cuteafd-coordinator:v6.0.0@sha256:f2bd29c76f8810e078eff0e1aed390427908fae944932cc6ca7dfb98071fa415` (`cuteafd plan --layout --rtx 2 --spark-ranks 0`).

## 1. Stitcher & Integrity Checks (`L0.1`, `L0.2`)

| Variant | Sources (`K3` / `K4` / Core) | K3 / K4 Experts (`bpw`) | Tensor Bytes (`GiB`) | Shards Written / Reused | Build Time | Header / SHA Mismatches | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `k325-selfcheck` | `K3.25` / `K3.25` / `K3.25` | `28,080` / `9,072` (`3.2442`) | `146,178,965,496` (`136.14`) | `44` / `0` | `109.6 s` | `0` / `0` (`2,215` SHA) | **PASS (`L0.1`)** |
| `k325-rebuilt` | `K3-v1` / `K4-K6` / `K3.25` | `28,080` / `9,072` (`3.2442`) | `146,178,965,496` (`136.14`) | `44` / `0` | `178.4 s` | `0` / `0` (`2,215` SHA) | **PASS (`L0.2`)** |
| `test-b0-k4` (`L03–08` → K4) | `K3-v1` / `K4-K6` / `K3.25` | `24,192` / `12,960` (`3.3488`) | `150,255,828,984` (`139.94`) | `6` / `38` | `35.9 s` | `0` / `0` (`1,314` SHA) | **PASS (`L0.2b`)** |
| `test-b0-k3` (`L03–08` → K3) | `K3-v1` / `K4-K6` / `K3.25` | `29,376` / `7,776` (`3.2093`) | `144,820,011,000` (`134.87`) | `6` / `38` | `26.9 s` | `0` / `0` (`1,314` SHA) | **PASS (`L0.2b`)** |

*Note on byte total:* Both `k325-selfcheck` and `k325-rebuilt` sum to the exact same `146,178,965,496 B` (`0 B` difference between rebuilt and self-rebuilt tensor payloads; `−19,975,456 B` / `−0.0137%` relative to `K3.25-v1`'s `model.safetensors.index.json` `total_size` metadata). Hardlink reuse kept the combined disk footprint of `k325-rebuilt` + `test-b0-k4` + `test-b0-k3` at **`172 GiB`** total.

## 2. CuteAFD `v6.0.0` CPU Planner Checks (`L0.3`)

| Checkpoint | Verdict | Expert Package | Core Precision (`fp8-prefill`) | Planned Weights (`rtx0` / `rtx1`) | Planned KV Pool (tokens) |
|---|---|---|---|---:|---:|
| `k325-native` (`K3.25-v1@0490d2f7`) | **READY to serve** | `glmf:exl3-k34` | `CheckpointPrecision` (`bf16 x519, f32 x68`; `Unchanged`) | `71,366` / `70,668 MiB` | `6,259,200` |
| `k325-rebuilt` | **READY to serve** | `glmf:exl3-k34` | `CheckpointPrecision` (`bf16 x519, f32 x68`; `Unchanged`) | `71,366` / `70,668 MiB` | `6,259,200` |
| `test-b0-k4` (`L03–08` 100% K4) | **READY to serve** | `glmf:exl3-k34` | `CheckpointPrecision` (`bf16 x519, f32 x68`; `Unchanged`) | `73,310` / `72,612 MiB` | `5,592,064` |
| `test-b0-k3` (`L03–08` 0% K4) | **READY to serve** | `glmf:exl3-k34` | `CheckpointPrecision` (`bf16 x519, f32 x68`; `Unchanged`) | `70,718` / `70,020 MiB` | `6,481,920` |

**Gate P0 outcome:** **PASS (`L0.1`, `L0.2`, `L0.3` all met).** Ready to draft and freeze `protocols/PHASE1.md` pending operator approval for Phase 1 GPU time (`~1.5–1.8 GPU-h`).
