# Phase 3 Protocol — Held-Out Continuous Gate (Stage 3A) & 4K/127K Task Confirmation (Stage 3B)

**Status:** FROZEN before first Phase 3 GPU request (`2026-11-23`)  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `fp8-prefill = auto/Unchanged`)  
**Candidates advanced from Phase 2 Stage 2B (`L2.3`, `Amendment 2B.1`, `[E2.8]–[E2.9]`):**
1. **`K3.45-B1B2`** (`b1b2-up-k4`, `+7.59 GiB`, `3.4535 bpw`) — **Primary deployment candidate** (`B1+B2` `L09–L20` at `100%` K4; `50.2%` of K3.25→K4 P-B KL headroom in `27.7%` of bytes).
2. **`K3.35-B2`** (`b2-up-k4`, `+3.80 GiB`, `3.3488 bpw`) — **Efficiency candidate** (`B2` `L15–L20` at `100%` K4; `36.0%` of K4 P-B KL headroom, `1.44×` KL/GiB vs `K3.45-B1B2`).
3. **`K3.25-iso-hot`** (`iso-b0b5b6-to-hot-b1b2`, `0 B` delta, `3.2442 bpw`) — **Secondary iso-bitrate candidate** (`B0+B5+B6 → 0%` K4, top `3,888` hot K3 projections in `B1+B2` → K4; `16.3%` of K4 P-B KL headroom at zero extra bytes).

**Controls:**
- `k325-native` (`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@0490d2f7`, published baseline)
- `k325-rebuilt` (matched stitched control from the same `K3-v1` + `K4-Core-K6` library)
- `k4-bf16core` (stitched library upper bound, `4.0000 bpw`, `+27.42 GiB` over `K3.25`)

---

## 1. Stage 3A — Held-Out Continuous Gate & Neutral-Author Check (`~2.7` wall-h, `score` mode)

Runs three strictly held-out / neutral evaluations before spending long generation hours in Stage 3B:

1. **Untouched P-A Teacher Panel (`51,175` rows across `25` clusters, NEVER used in Phase 2):**
   - Run on `K3.45-B1B2`, `K3.35-B2`, and `K3.25-iso-hot` and paired directly against Phase 1's untouched `k325-native`, `k325-rebuilt`, and `k4-bf16core` P-A receipts.
2. **Untouched `D_margin_heldout` (`n = 36` items, `127K`, frozen in Phase 1 and NEVER touched in Phases 1–2):**
   - Run inline via `/v1/bench/probe` (`top_k = 5`, zero dumps) across all 6 arms (`k325-native`, `k325-rebuilt`, `k4-bf16core`, `K3.45-B1B2`, `K3.35-B2`, `K3.25-iso-hot`).
3. **Public Neutral-Author `D_margin_neutral` (`build` `n = 24` items, `127K`, ZERO K4-EXL3 authorship):**
   - Built via `tools/build_margin_probe.py --source-mode neutral-public` exclusively from already-public non-K4 runs (`2026-11-10-context-probe` `anchor` and `2026-11-15-fp8-core-contrasts` `k325-fp8c`).
   - Run across all 6 arms to test whether `k4-bf16core > k325-rebuilt` and `candidate > k325-rebuilt` on `B_close` hold when K4 did not author the reasoning trajectories.

### Pre-Registered Stage 3A Gate (`L3.1`, `L3.2`)
- **`L3.1` (Untouched P-A & P-B KL generalization & non-inferiority):**
  - Non-inferiority: candidate P-A KL does not regress by `> +0.003` nats and P-B KL does not regress by `> +0.002` nats vs BOTH `k325-rebuilt` and `k325-native`.
  - Generalization: candidate achieves **statistically detected untouched P-A KL reduction** (`ΔKL_PA < 0`, paired 25-cluster 95% CI excluding `0`) vs `k325-rebuilt` (confirming Stage 2's P-B depth gains transfer to an independent distribution).
- **`L3.2` (Untouched `D_margin_heldout` `n = 36` + Neutral-Author `D_margin_neutral` `n = 24`):**
  - Report paired bootstrap 95% CIs (`10,000` resamples, seed `20261123`) for `B_close` vs BOTH `k325-rebuilt` and `k325-native` on both `D_margin_heldout` (`n = 36`) and `D_margin_neutral` (`n = 24`), plus the neutral-author `k4-bf16core − k325-rebuilt` validity check.
  - Note: Phase 1–2 established single-band `B_close` jitter (`SD = 0.275` nat across two shuffles); `L3.2` passes cleanly if a candidate improves `D_margin_heldout` `B_close` over both controls with 95% CI excluding `0`, and neutral-author `k4 − k325-rebuilt` is reported transparently regardless of outcome.
- **Stage 3A → Stage 3B Advancement Rule:**
  - Advance into Stage 3B: **`k325-native`** (deployment baseline) + **`K3.45-B1B2`** (if it passes `L3.1`) + **`K3.35-B2`** and/or **`K3.25-iso-hot`** if they pass `L3.1` with strong untouched P-A transfer (plus **`k4-bf16core`** as upper anchor, capped at 3–4 total Stage 3B arms = `~9–12` wall-h).

---

## 2. Stage 3B — 4K + 127K Context-Probe Task Confirmation (`~9–12` wall-h, `serve` mode, gated on Stage 3A)

- **Suite:** Full 60-item measured Context-Probe (`20 join`, `20 ledger`, `20 trace`) at **both `4K` and `128K` (`127K` content)**, **`4` passes (`single-rep0..rep3`)**, concurrency `8`, **`max_tokens = 32,768`** (strictly below CuteAFD's `65,536` RepeatGuard shingle cap), temperature `1.0`, `top_p = 0.95`.
- **Measured runtime on CuteAFD `v6.0.0`:** `~47 min / rep` → `~3.1 wall-h / arm` for 4 passes (`~9.3–12.4 wall-h` for 3–4 arms).
- **Power statement (stated in advance):** At 60 items × 4 passes (`240` 128K replies per arm), paired bootstrap 95% CI half-width is `~±3.5–4.0 pp`. Report `candidate − k325-native`, `candidate − k4-bf16core`, finished-only accuracy, unfinished counts (`cap` vs `repetition`), and exact GiB / KV-pool cost.
- **`L3.3` (Phase 3 → Phase 4 Gate):** Advance the strongest candidate to Phase 4 (`vLLM v0.9.2` carry-over) if it passes `L3.1`, shows no 4K regression, and moves 127K task accuracy and/or unfinished-reply count cleanly toward `k4-bf16core` without regression (optionally extending to 8 passes if borderline within `±2.5 pp`).
