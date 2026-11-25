# Experimental Plan: Kintsugi Bit Allocation for GLM-5.3-Flash EXL3-MCG K3.25

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB each, 300 W).  
**Primary search engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`; note `v6.0.1` only alters media time in `prefill_share` on GLM, with no text-only effect — we keep `v6.0.0` pinned for consistency).  
**Cross-engine confirmation target:** tpurtell's vLLM fork `v0.9.2` (`sha256:ea0430aa`, `TP=2`, `EP=1`, `DFlash2`).  
**Target checkpoint family:** `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` (unquantized BF16/F32 dense core + 37,152 MCG EXL3 routed experts).

---

## Why CuteAFD v6.0.0 for Search, Followed by vLLM v0.9.2 Carry-Over

1. **Bitwise prefill determinism (Kintsugi Step 0b):** CuteAFD `v5.0.0` demonstrated exact `0.0` KL difference and `1.0` argmax agreement across restarts (`FC-pb-repeat`, `DC-turn-floor`), whereas vLLM `v0.9.2` has `±0.0006` nat chunked-prefill jitter above `2,045` tokens. Phase 1 verifies `v6.0.0`'s determinism directly on `K3.25-v1`.
2. **Faster launch turnaround (~3× faster):** CuteAFD `v6.0.0` starts in `~2 min` vs vLLM `v0.9.2`'s `~6.4 min` warm restart (`382 s`), making a 14-flip layer-band scan practical.
3. **Matched dense-core precision:** Unlike CuteAFD `v5.0.0` (which converted BF16 MLA/FFN weights to FP8 block-128 at load; `cuteafd-eval` `c45`), CuteAFD `v6.0.0` keeps plain BF16/F32 core tensors unquantized (`CheckpointPrecision`, `fp8-prefill: Unchanged`; `[E0.7]`), matching vLLM `v0.9.2`'s BF16/F32 core execution on `K3.25-v1`.
4. **Native inline margin probe:** CuteAFD `v6.0.0`'s `/v1/bench/probe` (`rust/crates/cuteafd-api/src/openai/probe.rs:302–308, :567–569`) returns `ProbeRow { position, hash, argmax, top, wanted, finite }` inline in JSON when `top_k >= 2` and `wanted` (`y_t`) are specified on `score_positions`, giving exact teacher-forced margins $m_t = \log p(y_t) - \max_{y \neq y_t}\log p(y)$ with **zero logit dumps**.
5. **Loop-abort boundary:** CuteAFD's `RepeatGuard` rule R caps its shingle map at `MAX_ENTRIES = 65,536` tokens. All graded task evaluations pin `max_tokens = 32,768` so this bound cannot activate.

---

## Controlling the Fit-Library Confound

Phase 0 tensor hashing (`[E0.2]`) showed that `K3-v1`, `K3.25-v1`, and `K4-Core-K6` came from separate quantization runs (distinct `.trellis` hashes despite identical MCG shapes, dtypes, and `mcg_multiplier = 3417055213`). Therefore:

- Every stitched allocation drawn from `K3-v1` (K3 pool) + `K4-Core-K6` (K4 pool) + `K3.25-v1` (BF16/F32 core) is paired against a **matched stitched control (`K3.25-rebuilt`)** using `K3.25-v1`'s exact `9,072`-projection map.
- This cleanly separates:
  1. **Library offset:** `K3.25-rebuilt − K3.25-v1` (native published checkpoint).
  2. **Pure allocation effect:** `candidate − K3.25-rebuilt` (identical K3/K4 tensor pools).
  3. **Repair target:** Improving upon `K3.25-v1` toward `K4-bf16core` at explicitly measured byte cost.

---

## Staged Phases & Go/No-Go Gates

### Phase 0 — CPU Stitcher, Control Verification & Planner Check (**COMPLETE**, CPU-only)
- Built [`tools/stitch_exl3.py`](../tools/stitch_exl3.py) (1 BF16/F32 core shard + 43 per-layer MoE shards with incremental hardlink reuse).
- Verified `L0.1` (`k325-selfcheck`, `0` header/SHA mismatches), `L0.2` (`k325-rebuilt`, exact `146,178,965,496 B` tensor total; incremental band flips `test-b0-k4` in `35.9 s` and `test-b0-k3` in `26.9 s`), and `L0.3` (CuteAFD `v6.0.0` CPU planner returned `READY to serve` on all 4 checkpoints; `[E0.5]–[E0.7]`).

### Phase 1 — Step 0b Determinism, Library Offset & Two-Component Margin Probe Validation
Prior CuteAFD `v5.0.0` receipts (`2026-11-15-fp8-core-contrasts`) showed that ~4.7 of `k325-fp8c`'s 5.73-pp deficit at 127K came from **unfinished replies (`14` cap hits + `2` loop aborts vs `4` cap hits on `fp8c`)**, while finished-only scores differed by only `1.08 pp` (`0.9325` vs `0.9433`). A pure final-answer margin probe alone may miss termination failures.

1. **Construct a two-component build-side teacher-forced probe (`D_margin_build`)** using `/v1/bench/probe` (`top_k=5`, `wanted=y_t`, zero dumps), strictly split from held-out confirmation items (`D_margin_heldout`):
   - **Component A (Answer margins):** Hard 64K–127K `join`/`ledger`/`trace` answer tokens plus tool-call decision tokens.
   - **Component B (Termination / `</think>` margins):** Closing `</think>` and pre-divergence reasoning tokens along finished `K4` trajectories on the exact items where `K3.25` exhausted the 32,768-token cap.
2. **Baseline launches on CuteAFD `v6.0.0`:**
   - `K3.25-v1` (2 separate launches for Step 0b determinism)
   - `K3.25-rebuilt` (measures library calibration offset vs `K3.25-v1`)
   - `K4-bf16core` (all `37,152` experts from `K4-Core-K6` + BF16/F32 core — library upper bound)
   - `K3-bf16core` (all `37,152` experts from `K3-v1` + BF16/F32 core — library lower bound)
3. Record P-A KL, P-B KL, and both `D_margin_build` components on all four.
- **Gate P1 (`L1.1–L1.3`):** Restart margin spread `< 0.05` nat (`0.0` target) and P-B diff `0.0`; each retained probe component separates `K4-bf16core` above `K3.25-rebuilt` (95% CI excluding `0`) with fragile/negative items on `K3.25-rebuilt`; `|KL_PB(K3.25-rebuilt) − KL_PB(K3.25-v1)| ≤ 0.003` nats (else diagnose calibration offset first).

### Phase 2 — Coarse Damage Map: Layer-Band, Role & Routing Localization
1. **Stage 2A — 7 Layer-Band Scan (14 flips):**
   - Partition `L03–L44` into **7 six-layer bands** (`B0: L03–08`, `B1: L09–14`, `B2: L15–20`, `B3: L21–26`, `B4: L27–32`, `B5: L33–38`, `B6: L39–44`), plus optional `L45` (MTP) check.
   - Starting from `K3.25-rebuilt` (25% K4 per band):
     - **7 upward flips (band → 100% K4, `+3.80 GiB` each):** measure sink recovery ($\Delta m_A/\text{MiB}$, $\Delta m_B/\text{MiB}$, $\Delta\text{KL}/\text{MiB}$).
     - **7 downward flips (band → 0% K4, `−1.27 GiB` each):** identify safe source bands (`0` new negative-margin flips).
   - Report rank correlation $\rho$ (with bootstrap CI) between $\Delta\text{KL}/\text{MiB}$ and $\Delta m/\text{MiB}$ across bands.
2. **Stage 2B — Targeted Role Split & Routing-Targeted Sub-Band Localization (optional):**
   - **Targeted Role Split:** On the top 2 sink bands, test `gate_proj` / `up_proj` / `down_proj` separately.
   - **Routing-Targeted Sub-Band Localization (`[E0.8]`):** Once Stage 2A locates the top 1–2 task-damage bands, exploit the `67.24×` median within-layer routing skew (and the `38.6%` of top-25% routed experts still at K3) by testing:
     1. Upgrading **only the top-routed remaining K3 experts** inside those bands (ranked by calibration `sample_count` / `route_evidence × loss` from `quant_log.csv` and `K3-v1/ds4rt-exl3-error-ledger.jsonl`, costing `~+1.0–2.0 GiB` vs `+3.8–7.6 GiB` full-band).
     2. An **iso-bitrate cold-to-hot swap**, moving K4 off cold experts in safe source bands onto hot K3 experts in top-damage bands at matched bytes.
     3. If calibration routing misses task-specific 127K routing, an optional diagnostic capture of task-specific `expert_counts` on `D_margin_build` can follow.
- **Gate P2 (`L2.1–L2.2`):** `≥ 2.0×` sensitivity spread between best sink and cheapest safe source band in $\Delta m/\text{MiB}$ or $\Delta\text{KL}/\text{MiB}$. If flat across bands within noise, stop early and publish a clean negative result.

### Phase 3 — Targeted Repair & Held-Out Confirmation on CuteAFD `v6.0.0`
1. **Candidates (built from the same `K3-v1` + `K4-Core-K6` library):**
   - **Primary deployment candidate — `K3.35–K3.45-Kintsugi` (targeted band upgrade):** Keep `K3.25-rebuilt` intact and upgrade the top 1–2 Kintsugi-located sink bands to `100%` K4 (`+3.80 GiB` / `3.35 bpw` for 1 band; `+7.59 GiB` / `3.45 bpw` for 2 bands; planner still admits `~4.9–5.6M` KV tokens on CuteAFD `v6.0.0` and `> 3.5M` on vLLM `v0.9.2`). Fewer degrees of freedom, larger expected effect size, and cleaner cross-engine transfer than a fine-grained source/sink exchange.
   - **Secondary scientific candidates (iso-bitrate, exact `3.25 bpw` / `9,072` K4 projections):**
     - `K3.25-Kintsugi-iso`: exchange K4 from cheapest safe source bands (`0%`) into top task-margin sink bands (`75–100%`) at exact matched bytes (`0 B` tensor delta).
     - `K3.25-KL-iso`: same matched byte budget allocated strictly by P-B $\Delta\text{KL}/\text{MiB}$ (tests task-margin vs KL-only allocation).
2. **Power statement (stated in advance):**
   - **High-power continuous gate (`D_margin_heldout` + untouched P-A/P-B):** resolves small margin/KL shifts cleanly (`~±0.05–0.15` nat on margins, `~±0.0015` nat on P-B KL).
   - **127K graded task confirmation (60 items × 4 passes, `max_tokens = 32,768`):** paired CI half-width is `~±3.5–4.0 pp` (so at 4 passes, only shifts `≳ 4 pp` exclude `0`; smaller `2–3 pp` gains read as `"not detected"` unless extended to 8 passes, where half-width tightens to `~±2.5–2.8 pp`). We therefore report **both** `candidate − K3.25` (improvement), `candidate − K4-bf16core` (remaining gap & bound), **unfinished-reply counts**, and **exact GiB cost**.
- **Gate P3 (`L3.1–L3.3`):** Passes KL non-inferiority, shows statistically detected improvement on `D_margin_heldout` (95% CI excluding `0`), and closes the 127K unfinished-reply / task gap toward `K4-bf16core` cleanly enough to justify Phase 4 carry-over testing.

### Phase 4 — Cross-Engine Carry-Over on vLLM Fork `v0.9.2` (gated on Phase 3)
1. Mount the Phase 3 finalist via `MODEL_DIR_OVERRIDE` on unmodified `glm53-flash-exl3-k3.25@tpurtell+v0.9.2` (`sha256:ea0430aa`).
2. Paired run (`finalist` vs native `K3.25-v1`):
   - **P-A + P-B** (2 repeats each for vLLM chunked-prefill jitter).
   - **60-item Context-Probe at 4K & 127K** (4 passes, `max_tokens = 32,768`).
   - **`tool-eval-bench`** (88 scenarios × 3 repeats, seeds 1001–1003; baseline `160/163/160`) + **Block T speed & KV pool check**.
- **Gate P4 (`L4.1–L4.3`):** Held-out 127K improvement / unfinished-reply reduction carries over on vLLM `v0.9.2`, `tool-eval-bench` stays non-inferior (`≥ 159/176` mean), C1 code decode stays `≥ 95%` of baseline (`145–150 tok/s`), and KV pool stays `≥ 3.5M` tokens (`≥ 4.5M` if iso-bitrate).

---

## Methodology Extensions for Large Reasoning MoEs

Within the core Kintsugi search-and-confirmation loop, this study tests three scope-bound extensions for large reasoning MoE checkpoints:

1. **Termination-boundary (`</think>`) margins alongside answer-token margins:** On long-context reasoning models, quantization degradation often presents as runaway thinking (`max_tokens` cap hits) rather than wrong answer tokens alone. Pairing closing-`</think>` / pre-divergence margins (Component B) with answer-token margins (Component A) captures both failure modes directly.
2. **Two-level MoE localization (coarse depth bands → within-layer routing-hot experts):** Stage 2A localizes task damage across 7 six-layer depth bands; Stage 2B refines within the top sink bands using within-layer routing skew (`[E0.8]`, `sample_count` / `route_evidence × loss`) to test whether upgrading or swapping only hot K3 experts recovers most of the full-band margin gain at lower byte cost.
3. **Matched stitched-library controls (`K3.25-rebuilt`):** When reusing independently quantized EXL3 checkpoints (`K3-v1`, `K4-Core-K6`, `K3.25-v1`; `[E0.2]`), pairing every stitched candidate against a byte-matched control built from the same K3/K4 tensor pools separates quantization-run library offset from pure bit-allocation effects.

---

## Phase & Resource Summary

| Phase | Engine | Disk Scratch | Hard Kill Gate |
|---|---|---:|---|
| Phase 0 (Stitcher + planner) | CPU / CuteAFD planner | `172 GiB` | Gate P0 (**PASS**) |
| Phase 1 (Determinism + two-component probe) | CuteAFD `v6.0.0` | `~170 GiB` | Gate P1 |
| Phase 2 (7-band damage map + role/routing split) | CuteAFD `v6.0.0` | `~170 GiB` rolling | Gate P2 |
| Phase 3 (Repair + CuteAFD confirmation) | CuteAFD `v6.0.0` | `~180 GiB` rolling | Gate P3 |
| Phase 4 (vLLM `v0.9.2` carry-over) | vLLM fork `v0.9.2` | `~200 GiB` temp dumps | Gate P4 |
| **Summary** | | **≤ 200 GiB rolling** | Early stop at P1 / P2 / P3 |
