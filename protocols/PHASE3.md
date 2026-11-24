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
3. **Neutral-Author `D_margin_neutral` (`build` `n = 24` items, `127K`: `23/24` independent-quantizer `exlr8-k4`, `1/24` K3.25-authored fallback `ledger-118` — `Amendment 3A.1`):**
   - Built via `tools/build_margin_probe.py --source-mode neutral-public` from the published independent-quantizer **`2026-11-10-context-probe` `exlr8-k4`** run (`23/24` build items from `EXLR8 K4`, zero shared weights with EXL3-MCG; `1/24` K3.25-authored fallback `k325-fp8c`, sensitivity-checked with `ledger-118` excluded).
   - *(Pre-data `Amendment 3A.1`, recorded before `margins_neutral/` began: the initial draft sourced `anchor`, which shares `K4-Core-K6` expert weights bit-for-bit with `k4-bf16core`; replaced atomically with `exlr8-k4` before any neutral-probe GPU request ran.)*
   - Run across all 6 arms to test whether `k4-bf16core > k325-rebuilt` and `candidate > k325-rebuilt` on `B_close` hold under independent-quantizer trajectories.

### Pre-Registered Stage 3A Power & Gate (`L3.1`, `L3.2`, `Amendment 3A.1`)
- **Pre-data P-A power statement (`51,175` rows, `25` clusters; Phase 1 paired half-width `~0.0045` nat):**
  - If Stage 2B's P-B KL headroom shares transfer proportionally to untouched P-A (`0.01658` nat K4 headroom), expected P-A drops are:
    - `K3.45-B1B2` (`50.2%`): `~−0.0083` nat (**well-powered / detectable**),
    - `K3.35-B2` (`36.0%`): `~−0.0060` nat (**borderline**),
    - `K3.25-iso-hot` (`16.3%`): `~−0.0027` nat (**likely underpowered on P-A alone** — a non-detection on P-A therefore does not establish `"no transfer"`, and its `B_close = −0.172` Stage 2B watch item is tracked separately).
- **Winner's curse note on `D_margin_heldout` (`n = 36`):** Because `K3.45-B1B2` was selected after observing `D_margin_build` (`+0.173` nat), expect a smaller held-out `B_close` point estimate.
- **`L3.1` (Untouched P-A & P-B KL generalization & non-inferiority):**
  - Non-inferiority: candidate P-A KL does not regress by `> +0.003` nats and P-B KL does not regress by `> +0.002` nats vs BOTH `k325-rebuilt` and `k325-native`.
  - Generalization: candidate achieves **statistically detected untouched P-A KL reduction** (`ΔKL_PA < 0`, paired 25-cluster 95% CI excluding `0`) vs `k325-rebuilt`.
- **`L3.2` (Untouched `D_margin_heldout` `n = 36` + Independent-Quantizer `D_margin_neutral` `n = 24`):**
  - Report paired bootstrap 95% CIs (`10,000` resamples, seed `20261123`) for `B_close` vs BOTH `k325-rebuilt` and `k325-native` on both `D_margin_heldout` (`n = 36`) and `D_margin_neutral` (`n = 24`), plus the neutral-author `k4-bf16core − k325-rebuilt` validity check.
- **Stage 3A → Stage 3B Advancement Rule:**
  - Advance into Stage 3B: **`k325-native`** (deployment baseline) + **`k4-bf16core`** (upper anchor) + **`K3.45-B1B2`** (if it passes `L3.1`) + **`K3.35-B2`** and/or **`K3.25-iso-hot`** if they pass `L3.1` (capped at 3–4 total Stage 3B arms = `~9–12` wall-h).

---

## 2. Stage 3B — 4K + 127K Context-Probe Task Confirmation (`~9–12` wall-h, `serve` mode, gated on Stage 3A)

- **Suite:** Full 60-item measured Context-Probe (`20 join`, `20 ledger`, `20 trace`) at **both `4K` and `128K` (`127K` content)**, **`4` passes (`single-rep0..rep3`)**, concurrency `8`, **`max_tokens = 32,768`** (strictly below CuteAFD's `65,536` RepeatGuard shingle cap), temperature `1.0`, `top_p = 0.95`.
- **Measured runtime on CuteAFD `v6.0.0`:** `~47 min / rep` → `~3.1 wall-h / arm` for 4 passes (`~9.3–12.4 wall-h` for 3–4 arms).
- **Primary 127K endpoint & anchor-gap prerequisite (`Amendment 3A.1`):**
  - **Primary 127K endpoint:** Overall 127K accuracy (`mean_score` across `60 items × 4 passes = 240` replies; paired 95% CI half-width `~±3.5–4.0 pp`), with **unfinished-reply rate (`(cap + repetition) / 240`)** as co-primary diagnostic.
  - **Anchor-gap prerequisite:** Prior `−5.73 pp` separation came from the CuteAFD `v5.0.0` `k325-fp8c` graft; `k4-bf16core − k325-native` has not yet been measured on CuteAFD `v6.0.0` with BF16/F32 core. Therefore, **first evaluate `G_anchor = score_127k(k4-bf16core) − score_127k(k325-native)`**. If `G_anchor`'s 95% CI includes `0` at 4 passes, report `"127K anchor gap not reproduced on CuteAFD v6.0.0 / BF16-core at 4 passes (MDE ±3.5–4.0 pp)"` with **no candidate task ranking** (extending to 8 passes only if point `G_anchor ≥ +2.5 pp`).
- **`L3.3` (Numeric Phase 3 → Phase 4 Gate, when `G_anchor` is detected — tightened by pre-data `Amendment 3B.1` before `G_anchor` check):**
  - Evaluated via [`tools/analyze_phase3b_v2.py`](../tools/analyze_phase3b_v2.py) (original `analyze_phase3b.py` left untouched to preserve `stage3b/FROZEN.json` integrity during live runs):
    1. `L3.1` passed,
    2. 4K accuracy shows no detected regression (`Δ4K ≥ −1.5 pp` or 95% CI includes `0`),
    3. Candidate recovers **`≥ 50%`** of `G_anchor` on 127K overall accuracy (**reported with paired item-cluster bootstrap 95% CI**), **OR** recovers **`≥ 50%`** of `k4-bf16core`'s 127K unfinished-reply reduction **ONLY IF `k4-bf16core`'s own paired unfinished-rate reduction vs `k325-native` is itself detected (95% CI excluding `0`)** (preventing tiny integer-count ratios like `1/2` from triggering advancement; also reported with bootstrap 95% CI), without regressing finished-only accuracy.
  - **Arm-level vs item-level `B_close` note (`Amendment 3B.1`):** Because per-item `B_close` correlates weakly across trajectory authors (`r ≈ 0.23`), the end-of-thinking link to Stage 3B unfinished replies is evaluated at the **arm level**; any per-item `B_close` correlation is labeled exploratory.
  - **Optional post-3B authorship check (`Amendment 3B.1`):** `probe_neutral/manifest.json` already contains `36` untouched neutral held-out items (`35 exlr8-k4 + 1 k325-fp8c`); scoring them after Stage 3B (`~1.5–2 wall-h`) cleanly separates item-set vs self-authorship on `B1`'s held-out `B_close` lift without affecting any Stage 3B gate.
