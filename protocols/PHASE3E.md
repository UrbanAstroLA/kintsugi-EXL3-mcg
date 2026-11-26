# Phase 3 Stage 3E Protocol — Matched-Library Control (`k325-rebuilt`) 8-Pass 127K Task Confirmation

**Status:** **FROZEN before first GPU request (`2026-11-25`)**
**Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, serve mode, `max_tokens = 32,768`, concurrency `8`)
**Scope:** Run the matched stitched control **`k325-rebuilt`** (`K3-v1` + `K4-Core-K6` experts on `K3.25-v1` BF16/F32 core, exact `K3.25-v1` `9,072`-projection K4 map, `3.2442 bpw`) on the exact same `4K + 128K` Context-Probe task suite and paired seeds (`reps 0–3` `4K+128K` mixed + `reps 4–7` `128K`-only, `480` 128K replies + `240` 4K replies) as `k325-native`, `k4-bf16core`, and `k345-b1b2`.

---

## 0. Motivation (Stage 3D Early Stop, `[E3.10]`)

In Stage 3D arm #0, scoring the matched stitched control **`k325-rebuilt`** on the frozen `72` 128K dump prefixes revealed that **`81%`** of Stage 3C-A's `−0.4272` nat `k4 − native` `M_class(exit_10)` reduction was a **library offset** (`k325-rebuilt − k325-native = −0.3466` nats, `95% CI [−0.4817, −0.2182]`), present across `k325-native`-authored (`−0.394`), `k345-b1b2`-authored (`−0.379`), and `exlr8-k4`-authored (`−0.479`) prefixes despite functionally identical `config.json`, identical dense core tensors (`0` SHA mismatches), and `0.0006`-nat P-B KL parity (`[E1.3]`).

Against the matched control `k325-rebuilt`, `k4-bf16core` had no detected effect at the primary `exit_10` cut (`−0.0805`, `CI [−0.1987, +0.0454]`; detected only at `exit_20`: `−0.1365 [−0.2790, −0.0248]`), and `k345-b1b2` was `+0.0790 [−0.0318, +0.2001]` stickier (`+0.2265 [+0.1061, +0.3029]` on `ledger`). Stage 3D's 7-band probe scan was therefore stopped early during `b0-up-k4` (preserving `k325-rebuilt`'s receipt `[E3.10]`), because dividing 7 single-band effects by an undetected denominator (`−0.081`) cannot localize a candidate.

This raises one decisive live-task question: **Does the `35/480 → 15/480` 128K runaway-loop gap belong to `K4` precision, or to native `K3.25-v1`'s specific fit?**

---

## 1. Experimental Contract

- **Checkpoint:** `k325-rebuilt` (`derived/GLM-5.3-Flash-EXL3-Kintsugi`, revision `1111111111111111111111111111111100003250`, already verified `0` SHA mismatches in Phase 0 `[E0.5]`).
- **Driver & pairing:** Unmodified Stage 3B task driver (`run_tasks_dc.py`, `stage3b/FROZEN.json` untouched), `run-tag = p3b`:
  - `reps 0–3`: `4K + 128K` mixed (`60` items × `2` lengths × `4` passes)
  - `reps 4–7`: `128K`-only (`60` items × `4` passes)
  - Exact 1-to-1 `(item, length, rep, seed, nonce)` pairing across `reps 0–7` (`900M..907M`) with `k325-native`, `k4-bf16core`, and `k345-b1b2`.

---

## 2. Pre-Registered Reading Bands & Analyses (`L3E.1`, `tools/analyze_phase3e.py`)

### 2.1 Frozen Reading on `k325-rebuilt` `128K` Unfinished Count (`n = 480`)

| `k325-rebuilt` `128K` Unfinished (`n = 480`) | Frozen Reading |
|---|---|
| **`≥ 30 / 480` (`≥ 6.25%`)** | **`k325-rebuilt ≈ k325-native` (`35/480`):** K4's live 128K task advantage (`35 → 15/480`) is genuinely K4-specific; `M_class`'s `−0.347` nat offset was a quantization-library probe artifact (any further band search must be task-guided, e.g., complement arm). |
| **`≤ 20 / 480` (`≤ 4.17%`)** | **`k325-rebuilt ≈ k4-bf16core` (`15/480`):** native `K3.25-v1`'s elevated runaway rate is a property of that specific checkpoint/fit rather than K3.25 bit allocation, and `k345-b1b2` regressed relative to its own matched base. |
| **`21–29 / 480`** | **unresolved / intermediate** between native and K4. |

### 2.2 Paired 8-Pass Contrasts & Diagnostics (`60`-Item Cluster Bootstrap, `10,000` Resamples, Seed `20261123`)
1. **`k4-bf16core − k325-rebuilt`:** matched-library K4 effect on 128K overall accuracy, unfinished rate, and both-finished accuracy.
2. **`k325-rebuilt − k325-native`:** live-task library calibration offset (`4K` and `128K`).
3. **`k345-b1b2 − k325-rebuilt`:** matched-library effect of upgrading `B1–B2 (L09–L20)` to K4 (`reps 0–7` paired).
4. **Stage-split check (`reps 0–3` mixed vs `reps 4–7` `128K`-only):** Fisher exact test on `k325-rebuilt` halves + finished-only accuracy across both halves (checking whether stitched checkpoints show any `~1 pp` finished-only shift relative to `k325-native`).
5. **Frozen failure-mode diagnostics:** deterministic tail classifier (`record_list` / `line_repeat` / `computation`), `join` entering-vs-leaving decomposition (`≥ 15`-line run detector across all `160` `join` replies), and finished completion-token quantiles.
