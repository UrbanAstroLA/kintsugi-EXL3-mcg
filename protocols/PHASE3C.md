# Phase 3 Stage 3C Protocol — Loop Entry/Exit Margin Probe & `K3.45-B1B2` Mixed-Regime Replication

**Status:** **FROZEN before first GPU request (`2026-11-25`)**
**Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core)
**Scope:** Steps 1–2 (Stage 3C-A score-mode loop probe + Stage 3C-B mixed-regime replication). Step 3 is conditional and requires separate go-ahead.

---

## 0. Motivation

Phase 3 Stage 3B (`[E3.5]–[E3.7]`) established two facts and left one open contradiction:
- **Established:** The 128K anchor gap (`k4-bf16core − k325-native`) is detected under the exact two-stage inverse-normal test (`Z_acc = +2.3301`, `Z_unf = +2.8054`; pooled `+3.70 pp`, unfinished `35/480 → 15/480`). `100%` of the gap sits in rare mid-thinking runaway record-dumping loops (`join` + `ledger`; `0/240` unfinished at `4K` in every arm; finished-only accuracy equal).
- **Unresolved:** `K3.45-B1B2` failed `L3.3` under the frozen 8-pass rule (`86.90%`, `31/480` unf), but its two halves disagreed sharply (`23/240` unf in mixed `4K+128K` `rep 0–3` vs `8/240` unf in `128K`-only `rep 4–7`, Fisher exact `p = 0.008`; anchors `p = 0.73 / 0.60`).

Stage 3C resolves both questions before committing to any further band-confirmation task arm:
1. **Stage 3C-A (score mode):** Move the teacher-forced decision-margin instrument from `</think>` (`B_close`) to the **exact loop entry and exit decisions** at 128K context (`M_class`), test whether it separates `k4-bf16core` from `k325-native` including on K4-authored dumps, and **commit `k345-b1b2`'s predicted recovery share (excluding `k345`-authored dumps, with 95% CI) BEFORE Step 2 starts**.
2. **Stage 3C-B (serve mode):** Replicate `k345-b1b2` in the mixed `4K+128K` regime (`reps 8–11`, fresh server, `240` replies at `128K`), settling `k345-b1b2`'s ground truth and prospectively scoring Stage 3C-A's prediction.

---

## 1. Stage 3C-A — Loop Entry/Exit Margin Probe (CuteAFD `v6.0.0` Score Mode)

### 1.1 Frozen Prefix Manifest (`tools/build_loop_probe.py`)
Built deterministically from saved 128K trajectories across four author arms (`manifest.json` SHA-256 `42bc1f92a3cf8d759176ab9e5fe5fc44f5a702bd9ba85756e24fc91c5cb871b2`, mean `131,085` tokens/prefix):
- **Dump set (`72` prefixes — 100% of all 128K unfinished replies containing a `≥ 15`-line run matching `^(courier C-|parcel P-|hub H-|\d{5}\s+[A-Z0-9-]{3})`):**
  - `k325-native`: **`30`** | `k345-b1b2`: **`28`** | `k4-bf16core`: **`11`** | `exlr8-k4` (neutral quantizer): **`3`**
  - By family: `47 join`, `25 ledger`.
  - Cuts available: `enter` (`72/72`), `exit_5` (`72/72`), `exit_10` (`72/72`), `exit_20` (`61/72`).
- **Positive control set (`40` prefixes — deterministically stratified `10` per author, `20 join` + `20 ledger`, drawn from `774` finished 128K replies that quote `≥ 3` consecutive pure record lines mid-thinking and then exit the run, preferring exited runs of length `11–60` lines so `exit_10` is defined):**
  - `k325-native`: `10` | `k345-b1b2`: `10` | `k4-bf16core`: `10` | `exlr8-k4`: `10`
  - Cuts available: `enter` (`40/40`), `exit_5` (`40/40`), `exit_10` (`39/40`), `exit_20` (`9/40`), `leave` (`32/40`).
- **Scoring:** All `112` prefixes scored on all `3` arms (`k325-native`, `k345-b1b2`, `k4-bf16core`).

### 1.2 Quantities Measured at Every Cut (`tools/run_loop_probe.py`)
Each prefix is scored in a single cold prefill pass via `/v1/bench/probe` (`score_path = "prefill"`, `top_k = 10`, plus `want` containing the trajectory's next token ID, `join` record-start token IDs `{66384 'cour', 95115 'parcel', 26596 'hub', 19280 'Hub', 27649 'Parcel'}`, and the top empirically observed exit tokens `{73022 '```', 1112 '...', 21340 'Maybe', 2753 'No', 9450 'Yes', 12888 'Could', 8609 'Any', 4416 'So', 14181 'Wait', 198 '\n', 271 '\n\n', 154842 '</think>'}`):
- **`M_class(k)` (PRIMARY — symmetric class log-odds over `pool = top_10 ∪ want`):**
  - `logsumexp(lp of record-start tokens in pool) − logsumexp(lp of non-record tokens in pool)`, where `record-start tokens` are `{66384, 95115, 26596, 19280, 27649}` for `join` and pure ASCII digit tokens (`s.isdigit()`) present in `pool` for `ledger` (noting that on `ledger` the record side is a lower bound over listed digit tokens). Higher means a stickier loop.
- **Secondary diagnostics (reported alongside `M_class` at every cut):**
  - **`M_stick(k)`:** `logsumexp(lp(record_start)) − max(lp(non_record))` (max on exit side).
  - **`M_raw(k)`:** `lp(trajectory_next_token) − max(lp(other token in top_5))` (single-token formula matching `B_close`).
  - **`M_nl(k)`:** `lp('\n' [198]) − lp('\n\n' [271])` at `pos − 1` immediately preceding line `k+1`.
- **Pool coverage diagnostic:** total probability mass in `pool` (`exp(logsumexp(lp over pool))`) reported for every arm and cut; flag explicitly if median coverage is `< 0.95` at `exit_10`.

### 1.3 Primary Statistic, Single Gate & Recorded Prediction (`tools/analyze_phase3c.py --stage 3ca`)
- **Primary statistic (one only, frozen):**
  - **`M_class` at `exit_10`** on all `72` dump prefixes (all authors pooled), contrast **`k4-bf16core − k325-native`**, paired by prefix, item-cluster bootstrap (`10,000` resamples, seed `20261123`).
- **Single gate (`L3C.1`, with authorship robustness check):**
  - Passes iff:
    1. Pooled `k4-bf16core − k325-native` 95% CI on `M_class(exit_10)` is strictly below `0` (**`K4` less sticky**), **AND**
    2. The point estimate on **`k4-bf16core`-authored dumps (`n = 11`)** is also below `0` (where self-authorship bias runs *against* `K4`; sign check only given `n = 11`).
  - Do **not** gate on `k345-b1b2` (its task ground truth is contested until Stage 3C-B).
- **Secondary reporting:**
  - `exit_5`, `exit_20`, `enter`; `M_stick`, `M_raw`, `M_nl`; pool coverage per arm/cut; `join` vs `ledger` split of `M_class(exit_10)`; every author stratum (`k325-native` `n=30`, `k345-b1b2` `n=28`, `k4-bf16core` `n=11`, `exlr8-k4` `n=3` descriptive); `dump − control` difference per arm.
- **Recorded prospective prediction (mandatory BEFORE Stage 3C-B begins):**
  - **Primary prediction:** `k345-b1b2` recovery share `(k345 − k325) / (k4 − k325)` with 95% item-cluster bootstrap CI on dump prefixes **not authored by `k345-b1b2`** (`k325-native` + `k4-bf16core` + `exlr8-k4`, **`n = 44`**), avoiding downward self-authorship bias on `k345`-authored dumps.
  - **Alongside it:** all-author recovery share (`n = 72`), explicitly labeled as affected by `k345` self-authorship.
  - Write both to `STAGE3CA_PREDICTION.json` and commit to the repository **before starting Stage 3C-B**, with `primary_prediction_non_k345_n44` named as primary.

---

## 2. Stage 3C-B — `K3.45-B1B2` Mixed-Regime Replication (CuteAFD `v6.0.0` Serve Mode)

- **Arm & regime:** `k345-b1b2` (`L09–L20` K4, stitched snapshot `3333333333333333333333333333333300003450`), fresh CuteAFD `v6.0.0` server launch, **`reps 8–11`**, `4K + 128K` mixed regime (identical item ordering, concurrency `8`, `max_tokens = 32,768`, `temperature = 1.0`, `top_p = 0.95`, `top_k = -1` as Stage 1 `reps 0–3`).
- **Seed collision audit & fix (`tools/run_tasks_3c.py`):**
  - Pre-freeze inspection caught that `probelib_dc.seed_for` asserts `0 <= rep < 10` (and `rep = 10` would collide with `SEED_BASE["turns"] = 910_000_000`).
  - Leaving `probelib_dc.py` and `run_tasks_dc.py` untouched, `tools/run_tasks_3c.py` uses an unused deterministic seed block **`SEED_BASE_3C = 950_000_000`** (`reps 8–11` → `958,025,600 .. 961,542,656`) and `SALT_3C = "CONTEXT-PROBE-2026-11-10:3c"`. Dry-run verified **`0` collisions** across all `960` Stage 3B seeds (`reps 0–7`, `900M–907M`) and `480` Stage 3C-B seeds (`reps 8–11`, `958M–961M`).
- **Frozen reading on `128K` unfinished count out of `240` (`L3C.2`):**

| `128K` Unfinished Count (`reps 8–11`, `n = 240`) | Frozen Reading |
|---|---|
| **`≤ 12`** | `reps 0–3` were a fluctuation |
| **`13–17`** | unresolved |
| **`≥ 18`** | the mixed regime interacts with `k345-b1b2` |

- **Additional pre-registered analyses (`tools/analyze_phase3c.py --stage 3cb`):**
  1. Report the pooled 12-rep `k345-b1b2` figure descriptively only; **`L3.3` = FAIL is not reopened**.
  2. **Prospective prediction score:** compare `k345-b1b2`'s replication unfinished recovery share against the anchors' 8-pass rates (`35/480 = 7.29%` vs `15/480 = 3.125%`) to Stage 3C-A's recorded primary (`n = 44` non-`k345`-authored) recovery-share 95% CI.
  3. Apply the frozen deterministic tail classifier (`record_list` / `line_repeat` / `computation`) and stage-split pure-dump onset forensics (`reps 0–3`, `reps 4–7`, `reps 8–11`).

---

## 3. Conditional Step 3 (Not Yet Authorized)

- **If Stage 3C-A passes `L3C.1` AND its Step 1 prediction is consistent with Stage 3C-B:** propose a 7-band single-flip scan (`B0–B6`) using the validated loop-exit probe, followed by one 8-pass task arm on the localized band.
- **If Stage 3C-A fails `L3C.1` (or contradicts Stage 3C-B):** fallback to a task-level band arm (e.g., complement `B0 + B3–B6` or `B3–B6`), interpreted strictly through frozen contrasts (detected vs not detected against each anchor, noting `~±10/480` unfinished uncertainty).
- **Deferred:** The `exlr8-k4` `n = 36` held-out `B_close` check remains queued for the methodology write-up.
