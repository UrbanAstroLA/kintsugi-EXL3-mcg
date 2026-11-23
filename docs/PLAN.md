# Experimental Plan: Kintsugi Bit Allocation for GLM-5.3-Flash EXL3-MCG K3.25

**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB each, 300 W).  
**Primary search engine:** CuteAFD `v6.0.0` (`tag 15148d339`, release image pinned by digest).  
**Cross-engine confirmation target:** tpurtell's vLLM fork `v0.9.2` (`sha256:ea0430aa`, `TP=2`, `EP=1`, `DFlash2`).  
**Target checkpoint family:** `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` (BF16 dense core + 37,152 MCG EXL3 routed experts).

---

## Why CuteAFD v6.0.0 for Search, Followed by vLLM v0.9.2 Carry-Over

1. **Bitwise prefill determinism (Kintsugi Step 0b):** CuteAFD v5.0.0 and v6.0.0 demonstrated exact `0.0` KL difference and `1.0` argmax agreement across restarts (`32,768` fidelity rows bit-equal in `cuteafd-eval` `V6B-gates`; `0.0` KL floor on multi-turn frontiers in `DC-turn-floor`). vLLM v0.9.2 has `±0.0006` nat chunked-prefill jitter above `2,045` tokens.
2. **Fast launch turnaround (~6× faster):** CuteAFD starts in ~1 min vs vLLM v0.9.2's ~6.4 min warm restart (`382 s`), making a 14–21 group damage-map scan practical in **~2–3.5 GPU-hours** instead of ~6+ GPU-hours.
3. **Matched dense-core precision:** Unlike CuteAFD v5.0.0 (which converted BF16 MLA/FFN weights to FP8 block-128 at load; `cuteafd-eval` `c45`), CuteAFD v6.0.0 keeps plain BF16 core tensors in BF16 (`V6S-bf16-precision`), matching vLLM v0.9.2's BF16 core execution on `K3.25-v1`.
4. **Native probe and scoring APIs:** `/v1/bench/probe` exposes exact teacher-forced log-probabilities and margins at arbitrary depths (up to 128K+), alongside `cuteafd bench fidelity run` (P-B) and `glmf-teacher-kl.py` (P-A).

---

## Controlling the Fit-Library Confound

Phase 0 tensor hashing (`[E0.2]`) showed that `K3-v1`, `K3.25-v1`, and `K4-Core-K6` were quantized independently (distinct trellis hashes despite identical EXL3 MCG shapes, dtypes, and `mcg_multiplier = 3417055213`). Therefore:

- Every allocation experiment drawn from the `K3-v1` (K3 pool) + `K4-Core-K6` (K4 pool) + `K3.25-v1` (BF16 core) library **must include a matched stitched control (`K3.25-rebuilt`)** using `K3.25-v1`'s exact `9,072`-projection K4 map.
- This cleanly separates:
  1. **Library / calibration difference:** `K3.25-rebuilt − K3.25-v1` (native published checkpoint).
  2. **Pure Kintsugi allocation effect:** `K3.25-Kintsugi − K3.25-rebuilt` (identical K3/K4 tensor pools, differing *only* in which projections receive K4).
  3. **Ship bar:** Any candidate must beat *both* `K3.25-rebuilt` and native `K3.25-v1` on held-out confirmation.

---

## Staged Phases & Go/No-Go Gates

### Phase 0 — CPU Stitcher, Control Verification & Planner Check (~1 h CPU, 0 GPU-h)
1. Verify all `37,152` expert projections (`L03–L45`, 288 experts × `{gate_proj, up_proj, down_proj}`) across `K3-v1`, `K3.25-v1`, and `K4-Core-K6` share identical shapes, dtypes, and `mcg_multiplier`.
2. Build `tools/stitch_exl3.py`: given any per-projection `{3, 4}` map (obeying the two-consecutive-tier rule per layer), writes a self-contained Hugging Face safetensors checkpoint directory (`quantize_config.json`, `quantization_config.json`, `config.json`, `model.safetensors.index.json`, hardlinked/copied shards).
3. **Writer check (Kintsugi Step 1):** Rebuilding `K3.25-v1` from `K3.25-v1` itself must be byte-exact per tensor; building `K3.25-rebuilt` from `K3-v1` + `K4-Core-K6` + `K3.25-v1` BF16 core must pass both CuteAFD v6.0.0's CPU planner (`READY to serve`, BF16 core) and vLLM v0.9.2's `quantization_config.json` loader contract (`exl3_contract` / `port-exl3-projection-mixed-glm53`).
- **Gate P0 (see [`evidence/LINES.md`](../evidence/LINES.md)):** Planner returns `READY to serve` on `K3.25-v1`, `K3.25-rebuilt`, a 100%-K4 band variant, and a 0%-K4 band variant; `K3.25-rebuilt` matches `K3.25-v1`'s byte budget within `0.1%`.

### Phase 1 — Step 0b Determinism, Library Calibration Check & Margin Probe Validation (~1.5 GPU-h on CuteAFD v6.0.0)
1. **Construct the build-side teacher-forced margin probe (`D_margin_build`):**
   - Select ~30–40 hard items (distinct from the held-out confirmation split!) drawing from long-context retrieval/aggregation prefixes (`64K–127K`, where K3.25 degrades vs K4) and multi-step tool/code prefixes, scoring exact teacher-forced answer-token margin $m_i = \min_{t \in i}(\log p(y_t) - \max_{y \neq y_t}\log p(y))$ via `/v1/bench/probe`.
   - Split explicitly into **build set** (used to rank groups in Phase 2) and **held-out set** (untouched until Phase 3 confirmation).
2. **Baseline launches on CuteAFD v6.0.0:**
   - `K3.25-v1` (2 separate restarts for Step 0b determinism check)
   - `K3.25-rebuilt` (measures library offset vs native `K3.25-v1`)
   - `K4-bf16core` (all 37,152 experts from `K4-Core-K6` + BF16 core from `K3.25-v1` — the upper-bound reference for our library!)
   - `K3-bf16core` (all 37,152 experts from `K3-v1` + BF16 core — the lower-bound reference)
3. Record P-A KL, P-B KL, and `D_margin_build` margins on all four.
- **Gate P1 (Go/No-Go):**
  - Determinism: restart-to-restart margin spread `< 0.05` nat (target `0.0000`) and P-B KL diff `0.0`.
  - Probe validity (Kintsugi Step 2 rule): `D_margin_build` separates `K4-bf16core` above `K3.25-rebuilt` with paired bootstrap 95% CI excluding 0, and contains fragile/negative-margin items on `K3.25-rebuilt`.
  - Library check: `K3.25-rebuilt` sits within `±0.003` nats P-B KL of native `K3.25-v1` (if not, diagnose calibration differences before spending Phase 2 GPU time).

### Phase 2 — Coarse Damage Map: Layer-Band & Targeted Role Localization (~2.5–4 GPU-h on CuteAFD v6.0.0)
1. **Stage 2A — 7 Layer-Band Scan (14 flips, ~2–2.5 GPU-h):**
   - Partition MoE layers `L03–L44` into **7 contiguous 6-layer bands** (`B0: L03–08`, `B1: L09–14`, `B2: L15–20`, `B3: L21–26`, `B4: L27–32`, `B5: L33–38`, `B6: L39–44`), plus optional `L45` (MTP) check.
   - Starting from `K3.25-rebuilt` (25% K4 per band):
     - **7 downward flips (band → 0% K4):** identify cheapest safe **source** bands (zero hit-list regressions, lowest $\Delta\text{KL}/\text{MiB}$ and $\Delta m/\text{MiB}$).
     - **7 upward flips (band → 100% K4):** identify highest-value **sink** bands (largest margin recovery on fragile/failing items and $\Delta\text{KL}/\text{MiB}$).
   - Measure both `D_margin_build` and P-B KL (full or fast build subset) on every flip.
   - Report rank correlation $\rho$ (with bootstrap CI) between $\Delta\text{KL}/\text{MiB}$ and $\Delta m/\text{MiB}$ across bands (Kintsugi Step 3).
2. **Stage 2B — Targeted Role Split (optional, ~1 GPU-h):**
   - On the top 2–3 sink bands and top 2–3 source bands, flip `gate_proj`, `up_proj`, and `down_proj` separately to test whether `K3.25-v1`'s fixed `3:8:5` (`gate:down:up`) ratio should also shift per depth band.
- **Gate P2 (Go/No-Go):**
  - Meaningful sensitivity spread across bands (≥ 2× ratio in $\Delta m/\text{MiB}$ or $\Delta\text{KL}/\text{MiB}$ between best sink and cheapest source bands, with at least one safe source band having 0 hit-list regressions). If all bands are flat within noise, stop and publish a clean negative result (confirming uniform layer allocation is near-optimal on GLM-5.3-Flash).

### Phase 3 — Iso-Bitrate Repair & Held-Out Confirmation on CuteAFD v6.0.0 (~6–8 GPU-h)
1. **Construct candidate allocations (from the same `K3-v1` + `K4-Core-K6` library):**
   - **`K3.25-Kintsugi-iso` (exact 3.25 bpw, 9,072 K4 projections):** Exchange K4 projections from cheapest safe source bands (down to `0%`) into top task-margin sink bands (up to `75–100%`) at equal byte budget (within `0.1%`).
   - **`K3.25-KL-iso` (diagnostic control, exact 3.25 bpw):** Same budget allocated strictly by P-B $\Delta\text{KL}/\text{MiB}$ (directly tests Kintsugi's thesis: task-margin vs average-KL allocation on a 288-expert MoE).
   - **`K3.38-Kintsugi` (optional headroom arm):** Keep `K3.25-rebuilt` intact and upgrade the top 1–2 sink bands to `100%` K4 (`+~6–10` GiB total, preserving >3.5M KV tokens on vLLM).
2. **Held-out confirmation on CuteAFD v6.0.0 (pre-registered in `protocols/` before first request):**
   - Full P-A + P-B KL vs BF16 (confirming no catastrophic regression on untouched windows).
   - **60-item Context-Probe graded tasks at 4K and 127K (4 passes, paired item bootstrap):** Primary endpoint is `127K` ITT field score and unfinished count against `K3.25-v1` and `K3.25-rebuilt` (using held-out items/seeds not in `D_margin_build`).
- **Gate P3 (Ship to Cross-Engine Confirmation):**
  - Passes all KL non-inferiority lines AND resolves a statistically detected gain (95% CI excluding 0) on held-out 127K graded tasks or held-out recall margins against both `K3.25-rebuilt` and native `K3.25-v1`.

### Phase 4 — Cross-Engine Carry-Over Confirmation on vLLM Fork v0.9.2 (~5–7 GPU-h, only if Gate P3 passes)
1. Mount the winning CuteAFD-selected checkpoint via `MODEL_DIR_OVERRIDE` on unmodified `glm53-flash-exl3-k3.25@tpurtell+v0.9.2` (`sha256:ea0430aa`).
2. Run paired comparison (`K3.25-Kintsugi-iso` vs native `K3.25-v1`, identical serving env):
   - **P-A + P-B static check** (2 repeats each to cover vLLM's `±0.0006` nat chunked-prefill jitter).
   - **127K Context-Probe graded tasks** (4 passes) and **`tool-eval-bench`** (88 scenarios, 3 repeats, seeds 1001–1003; baseline `160/163/160`).
   - **Speed & VRAM check:** Block T subset (1K/8K/32K/128K code & reasoning, C1 and C4/C16), DFlash2 acceptance (~51% baseline), and KV pool tokens (`4.71M` baseline).
- **Gate P4 (Final Ship Line):** Held-out task improvement carries over onto vLLM v0.9.2, `tool-eval-bench` stays non-inferior (`≥ 159/176` mean), C1 decode stays within `≥ 95%` of baseline (`~145–150 tok/s`), and KV pool remains `~4.71M` tokens (for iso-bitrate).

---

## Resource Budget Summary

| Phase | Engine | Est. GPU-h | Disk Scratch | Hard Kill Gate |
|---|---|---:|---:|---|
| Phase 0 (CPU audit + stitcher + planner) | CPU / CuteAFD planner | 0 h | ~140 GiB | Gate P0 |
| Phase 1 (Determinism + probe validation) | CuteAFD v6.0.0 | ~1.5 h | ~150 GiB | Gate P1 |
| Phase 2 (Layer-band damage map) | CuteAFD v6.0.0 | ~2.5–4.0 h | ~150 GiB rolling | Gate P2 |
| Phase 3 (Repair + CuteAFD confirmation) | CuteAFD v6.0.0 | ~6–8 h | ~150 GiB rolling | Gate P3 |
| Phase 4 (vLLM v0.9.2 carry-over) | vLLM fork v0.9.2 | ~5–7 h | ~200 GiB temp dumps | Gate P4 |
| **Total (if all gates pass)** | | **~15–20.5 h** | **≤ 200 GiB rolling** | Early stop at 0 h / 1.5 h / 5 h |
