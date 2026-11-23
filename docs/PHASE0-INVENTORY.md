# Phase 0 Inventory: Checkpoints, Allocation Audit & Loader Contracts

Zero-GPU inspection performed `2026-11-23` against local checkpoint snapshots and engine source repositories.

---

## 1. `K3.25-v1` Existing Bit Allocation (`[E0.1]`)

Inspected `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` (`rev 0490d2f7`, `quantize_config.json` and safetensors headers):

- **Non-expert core:** `1,618` unquantized tensors (`1,327` BF16 + `291` F32 norms/scales/biases: dense MLP layers 0–2, all 34 KDA + 11 MLA + 1 MTP attention projections, shared experts, routers, embeddings, `lm_head`).
- **Routed EXL3 experts:** `37,152` projections (`43` MoE layers `L03–L45` × `288` experts × `{gate_proj, up_proj, down_proj}`):
  - **28,080 at K3** (3-bit MCG EXL3)
  - **9,072 at K4** (4-bit MCG EXL3): `1,701 gate_proj`, `2,835 up_proj`, `4,536 down_proj` (exact `3 : 5 : 8` role ratio).
- **Per-layer distribution:** **Strictly uniform across depth.** Every single MoE layer `L03–L44` received either `215` or `217` K4 projections out of `864` (`25.0%` per layer):
  - Even/odd alternating layers: `gate = 40` or `41`, `up = 67` or `68`, `down = 108`.
  - Layer `L45` (MTP): `0` K4 projections (`0%`).
- **Selection rule recorded in metadata:** `base-tier-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1` on `agentic-quant-panel-v1` (1,426 prompts, 1.08M tokens). No inter-layer reallocation was performed.

---

## 2. Local `K ∈ {3, 4}` EXL3-MCG Fit Library (`[E0.2]`)

| Checkpoint | Revision | Routed Expert Coverage | Bits | Trellis Shape (`gate_proj`) | `mcg_multiplier` | Non-Expert Core |
|---|---|---:|---:|---|---:|---|
| `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1` | `319d66a8` | `37,152 / 37,152` | `3` | `[256, 128, 48]` `int16` | `3417055213` | BF16/F32 (`1,618`) |
| `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` | `0490d2f7` | `37,152 / 37,152` | `3` / `4` | `[256, 128, 48]` / `64` `int16` | `3417055213` | BF16/F32 (`1,618`) |
| `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-Core-K6` | `5d3fffe1` | `37,152 / 37,152` | `4` | `[256, 128, 64]` `int16` | `3417055213` | K6 EXL3 (`324` tensors) |

- **Exhaustive schema compatibility:** Verified all `37,152` routed expert keys across `K3-v1`, `K3.25-v1`, and `K4-Core-K6` (`0` mismatches): identical tensor suffixes (`.trellis`, `.suh`, `.svh`, `.mcg`), dtypes (`int16`, `float16`, `float16`, `int32`), `mcg_multiplier = 3417055213` (`0xCBAC1FED`), and exact `1,048,576 B` (`1.000000 MiB`) delta per K3→K4 projection (`256 × 128 × (64 − 48) × 2 B`).
- **Byte-identity check:** Spot-checked SHA-256 hashes of `.trellis` tensors across snapshots:
  - `K3-v1` (`319d66a8` == `1e4abd26`) ≠ `K3.25-v1` (`0490d2f7` == `701cd745`) on K3 projections (`26a8e13f...` vs `b393030c...` on `L03.exp0.gate_proj`).
  - `K4-Core-K6` (`5d3fffe1`) ≠ `K3.25-v1` (`0490d2f7`) ≠ `brandonmusic/tr3-4bpw` (`a5fee929`) on K4 projections (`82f0131d...` vs `7bb30c76...` vs `7ec195df...` on `L03.exp1.gate_proj`).
- **Implication for experimental design:** Because the three checkpoints came from distinct quantization runs, every Kintsugi candidate stitched from `K3-v1` + `K4-Core-K6` is paired against a **matched stitched control (`K3.25-rebuilt`)** built from the same library using `K3.25-v1`'s exact `9,072`-projection map.

---

## 3. Loader Contract Checks (`[E0.3]`, `[E0.4]`)

### CuteAFD `v6.0.0` (`tag 15148d339`)
- **Manifest key allowlist (`rust/crates/cuteafd-loader/src/formats/exl3_contract.rs:219–227`):** Every top-level key in `K3.25-v1/quantize_config.json` passes the closed allowlist.
- **Two-tier mixed EXL3 support (`v41_exl3.rs`, `v41_exl3_residency.rs:262`):** Supports two-tier `{K3, K4}` routed experts (`glmf:exl3-k34`), including uniform 0%-K4 and 100%-K4 bands (`[E0.7]`).
- **Dense core:** Plain BF16/F32 core tensors remain unquantized (`CheckpointPrecision`, `fp8-prefill: Unchanged`; `[E0.7]`).
- **Inline probe margins (`rust/crates/cuteafd-api/src/openai/probe.rs:302–308, :567–569`):** `ProbeRow` returns `{ position, hash, argmax, top, wanted, finite }` inline in JSON when `top_k >= 2` and `wanted` are requested on `score_positions`, enabling exact teacher-forced margin scoring with zero logit dumps.

### tpurtell vLLM Fork `v0.9.2` (`sha256:ea0430aa`)
- **Projection-mixed EXL3 (read from local `v0.9.0` patch `patches/port-exl3-projection-mixed-glm53.py:236–260, :339–349`; `v0.9.2` support inferred from the same lineage and operational `K3.25-v1` serving, to be directly verified in Phase 4):**
  1. Each MoE layer must use either a uniform integer tier (`K2–K6`, taking the ordinary path) or **exactly two consecutive integer tiers** (`{3, 4}`, taking `_prepare_projection_mixed_weights`).
  2. `codebook = "mcg"` and `EP = 1` (`TP = 2` rank-sliced experts).
  3. All `tensor_storage` entries must be standard routed expert projections (`mlp.experts.\d+.(gate_proj|up_proj|down_proj)`), keeping non-expert core tensors unquantized.
- **Local checkpoint loading:** Supported via `MODEL_DIR_OVERRIDE`.
