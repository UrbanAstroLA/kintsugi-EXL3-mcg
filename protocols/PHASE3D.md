# Phase 3 Stage 3D Protocol — 7-Band Loop-Exit Scan (`B0..B6`, Step 3A)

**Status:** **FROZEN before first GPU request (`2026-11-25`)**
**Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, score mode `/v1/bench/probe`)
**Scope:** Matched-library control `k325-rebuilt` plus all 7 upward single-band flips (`B0..B6 → 100% K4`, `+3.7969 GiB` each, `3.3488 bpw`) on the frozen `72` 128K dump prefixes. Step 3B (one 8-pass task confirmation arm) remains gated on this scan and separate go-ahead.

---

## 0. Motivation & Exact Matched-Library Control

Stage 3C (`[E3.8]–[E3.9]`) established that:
1. `M_class(exit_10)` cleanly separates `k4-bf16core` from `k325-native` (`−0.4272` nats, `5.5σ`, `k4`-authored `n = 11` sign check `−0.1461 < 0`, median pool coverage `≥ 0.99997`).
2. Across `12` passes (`720` replies), `k345-b1b2` (`L09–L20` K4) is at `87.17%` overall / `45/720` (`6.25%`) unfinished (`25%` pooled recovery, `35%` in `reps 8–11`), confirming `B1–B2` alone is insufficient for Phase 4.

Because all 7 single-band flips (`B0..B6 → 100% K4`) are stitched from `K3-v1` + `K4-Core-K6` + `K3.25-v1` core, Stage 3D scores the **matched stitched control `k325-rebuilt`** alongside `B0..B6` on the exact same `72` dump prefixes (paired against already-scored Stage 3C-A `k325-native`, `k345-b1b2`, and `k4-bf16core`). This prevents the library calibration offset (`k325-rebuilt − k325-native`) from accumulating `7×` in the band-sum additivity check.

---

## 1. Frozen Prefix Subset & Scoring Contract

- **Prefix set (`probe_dumps72/manifest.json`):** Exact `72` dump prefixes from Stage 3C-A's frozen `probe_loop/manifest.json` (`k325-native: 30`, `k345-b1b2: 28`, `k4-bf16core: 11`, `exlr8-k4: 3`; `47 join`, `25 ledger`; cuts `enter: 72`, `exit_5: 72`, `exit_10: 72`, `exit_20: 61`). The `40` finished controls are dropped because `k4-bf16core` lowered `M_class` on `join` controls as well (`−0.36` vs `−0.52`), so controls do not sharpen the band ranking.
- **Scoring:** Unmodified `tools/run_loop_probe.py` (`cold = true`, `score_path = "prefill"`, `top_k = 10`, identical `want` sets; bitwise prefill determinism on CuteAFD `v6.0.0` previously verified at `0.000000` across `24/24` 128K prefixes in `[E1.1]`).
- **Arms (`8` total):**
  1. `k325-rebuilt` (matched stitched control, `0 B` delta vs native, `3.2442 bpw`)
  2. `b0-up-k4` (`L03–L08 → 100% K4`)
  3. `b1-up-k4` (`L09–L14 → 100% K4`)
  4. `b2-up-k4` (`L15–L20 → 100% K4`)
  5. `b3-up-k4` (`L21–L26 → 100% K4`)
  6. `b4-up-k4` (`L27–L32 → 100% K4`)
  7. `b5-up-k4` (`L33–L38 → 100% K4`)
  8. `b6-up-k4` (`L39–L44 → 100% K4`)

---

## 2. Pre-Registered Statistical Rules (`L3D.1`, `tools/analyze_phase3d.py`)

All contrasts are paired by prefix, item-cluster bootstrapped (`10,000` resamples, seed `20261123`) on primary metric **`M_class`** (`exit_10` primary; `enter`, `exit_5`, `exit_20` secondary), measured against matched control **`k325-rebuilt`** (reporting `k325-rebuilt − k325-native` library offset explicitly):

1. **Resolution statement:**
   - If K4's `~−0.43` nat reduction were distributed evenly across 7 bands, each band would carry `~−0.06` nats, below single-band detection (`SE ~ 0.07–0.08` nats). Only a concentrated band (`≳ 0.15` nats) can individually exclude `0`.
   - Therefore, `"no single band detected"` is the expected reading under a distributed effect and reads as **"distributed"**, not **"absent"**.
2. **Pre-registered additivity checks:**
   - **(a) Pair additivity:** `(b1-up − rebuilt) + (b2-up − rebuilt)` vs measured `(k345-b1b2 − rebuilt)` at `enter`, `exit_5`, `exit_10`, and `exit_20`.
   - **(b) Whole-model 7-band sum:** `S_7 = Σ_{i=0..6} (b{i}-up − rebuilt)` vs measured `Δ_K4 = (k4-bf16core − rebuilt)` at `exit_10` (and secondary cuts):
     - **`S_7 / Δ_K4 ∈ [0.75, 1.25]` (`±25%`):** additive — single-band localization is meaningful.
     - **`S_7 / Δ_K4 < 0.50`:** strongly interacting / non-additive — do not hunt further single bands; fallback to the complement task arm (`B0 + B3–B6`) or close out.
     - **`S_7 / Δ_K4 ∈ [0.50, 0.75)` or `> 1.25`:** partial / super-additive — interpret band ranking cautiously.
3. **Author-stratum & family rank stability:**
   - Rank `B0..B6` on all `72` dump prefixes (`exit_10`), and also report the ranking within each author stratum (`k325-native` `n=30`, `k345-b1b2` `n=28`, `k4-bf16core` `n=11`, `exlr8-k4` `n=3` descriptive) and within each family (`join` `n=47`, `ledger` `n=25`). Flag explicitly if the `#1` band shifts across author strata.
   - Report Spearman rank correlation `ρ` between Stage 3D `M_class(exit_10)` band sensitivity and Stage 2A `≤ 14.4K` P-B KL band sensitivity (`B2, B1, B3, B0, B4, B6, B5`).
4. **Pre-registered Step 3B candidate selection rule (written before the scan):**
   - Advance a candidate to **Step 3B** (one 8-pass task arm, separate go-ahead) ONLY IF `S_7 / Δ_K4 ≥ 0.50` **and** there exists a **new** band set (distinct from `B1–B2` alone) of **`≤ 3` bands** (`≤ +11.39 GiB`, `≤ 3.558 bpw`; preferring `≤ 2` bands `+7.59 GiB` if it meets the bar) whose summed single-flip matched-library shares reach **`≥ 80%`** at `exit_10` and `≥ 60%` on both `join` and `ledger` (also reporting `enter` and `exit_20` shares).
   - *Interpretation caveat:* Because `M_class` overshot `k345-b1b2`'s live task unfinished recovery by `~2×` and does not observe the upstream decision to enter a recitation on `join`, any Step 3A scan winner is strictly a **hypothesis for an 8-pass Step 3B task arm**, not a confirmed task result.
   - *Step 3B reading bands (if launched):* Against 8-pass anchors (`35/480` native, `15/480` K4), pre-register `≤ 20/480` (K4-like), `≥ 30/480` (native-like), `21–29/480` (unresolved), while tracking finished-only accuracy alongside unfinished counts.
