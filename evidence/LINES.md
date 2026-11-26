# Pre-Registered Decision Lines

Following Kintsugi Step 5 (`R1–R6`): every line is written with a date before the run it judges. A wrong line is amended only for later runs, with the reason recorded; missed lines stay failed in the record.

---

## Phase 0 — Stitcher & Loader Contract (registered `2026-11-23`; **ALL PASS**)

- **L0.1 (Writer fidelity) — PASS (`[E0.5]`):** Rebuilding `K3.25-v1` from its own shards via `tools/stitch_exl3.py` produced `0` header mismatches across all `150,226` tensors and `0` SHA-256 mismatches across `2,215` checked tensors (`109.6 s`).
- **L0.2 (Byte budget) — PASS (`[E0.6]`):** `K3.25-rebuilt` matches `k325-selfcheck`'s exact tensor byte total (`146,178,965,496 B`, `0 B` delta; `−0.0137%` vs `K3.25-v1`'s `index.metadata.total_size`, well within `0.1%`), with `0` header/SHA mismatches (`178.4 s`). Incremental band flips (`test-b0-k4`, `test-b0-k3`) hardlinked `38/44` shards and rewrote `6` layers in `35.9 s` and `26.9 s` with `0` mismatches.
- **L0.3 (Planner admission) — PASS (`[E0.7]`):** CuteAFD `v6.0.0` CPU planner (`cuteafd plan --layout --rtx 2 --spark-ranks 0`, zero GPU) returned `READY to serve` with `CheckpointPrecision` BF16/F32 core (`fp8-prefill: Unchanged`) and `glmf:exl3-k34` on all four checkpoints: `k325-native` (`6,259,200` KV tok), `k325-rebuilt` (`6,259,200` KV tok, layout-identical), `test-b0-k4` (`5,592,064` KV tok), and `test-b0-k3` (`6,481,920` KV tok).

---

## Phase 1 — Determinism, Library Offset & Two-Component Margin Probe Validation (CuteAFD `v6.0.0`, frozen `2026-11-23` in `protocols/PHASE1.md`; **ALL PASS**)

- **L1.1 (Step 0b determinism) — PASS (`[E1.1]`):** Two separate CuteAFD `v6.0.0` launches of `K3.25-v1` showed exact `0.000000` max per-item margin spread (`24/24` identical `B_close` hashes) on `D_margin_build` and `0.000000` P-B `gen` KL difference.
- **L1.2 (Two-component probe validity) — PASS on Component B / Component A retired (`[E1.2]`):** Component B (`B_close`, `</think>` token `154842`) ranked `K4-bf16core` above `K3.25-rebuilt` by `+0.2773` nats (`95% CI [+0.0197, +0.5381]`, excluding `0`, `17/24` items higher; `+0.7311` nats / `23/24` higher vs `K3-bf16core`) with `6/24` fragile (`< 1.0` nat) items (`2/24` negative) on `K3.25-rebuilt`. Component A (`A_min`, final JSON values) saturated (`+12.41–12.65` nats, `0/24` fragile, `Δ = −0.0122`, CI includes `0`) and is retired from Phase 2 decision criteria (logged diagnostically only).
- **L1.3 (Library calibration check) — PASS (`[E1.3]`):** `|KL_PB(K3.25-rebuilt) − KL_PB(K3.25-v1)| = 0.000645` nats (`−0.000645`, `95% CI [−0.002536, +0.001145]`, `4.6×` inside `≤ 0.003`); P-A diff = `+0.000275` nats (`95% CI [−0.002806, +0.003298]`).

---

## Phase 2 — Layer-Band Damage Map & Sub-Band Localization (CuteAFD `v6.0.0`, frozen `2026-11-23` in `protocols/PHASE2.md`)

*Pre-data Amendment 2A.1 (`2026-11-23`, recorded before any Stage 2A flip completed):* Flat-null simulation against Phase 1's measured `0.784`-nat equal-bitrate `B_close` jitter (`k325-rebuilt − k325-native`) showed original `L2.1` passes `93%–100%` of the time under a completely flat null (noisy `max/min` ratio near zero + only 4 items in `[0, 1)`). Original `L2.1` is kept on record below and superseded as the governing Gate P2 by the strictly tighter `L2.1-v2`:

- **L2.1-v2 (Active governing Gate P2 — non-flat structural sensitivity) — PASS (`[E2.1]–[E2.3]`):**
  1. **Primary P-B `gen` KL spread (`15,104` rows, `53` clusters) — PASS:** Direct paired `KL_PB(B2-up) − KL_PB(B5-up) = −0.003620` nats (`95% CI [−0.005241, −0.002078]`, `4.5σ`, excludes `0`), and upward `max / min` ratio = **`4.309×`** (`≥ 1.5×`) with `min` (`B5 = −0.001093`, CI `[−0.001766, −0.000393]`) strictly positive and excluding `0`.
  2. **Cross-direction consistency — PASS:** Spearman `ρ_up_down_KL = +0.6786` (`≥ 0.60`), with `B1` and `B2` ranking `#1–#2` in both directions.
  3. **Confirmatory `B_close` + null calibration — CONFIRMATORY (pending Stage 2B two-shuffle jitter scale):** `ρ(up B_close, −ΔKL_PB) = +0.5357`; all positive upward CIs include `0` and 7-band sum is non-additive (`0.42×`). Two within-band shuffle controls (`null-shuf-b2`, `null-shuf-b5`) run first in Stage 2B (`protocols/PHASE2B.md`, `Amendment 2B.1`).
- **L2.1 (Original, archived verbatim; superseded by `L2.1-v2`) — PASS (`[E2.1]–[E2.3]`):** Upward P-B KL ratio `4.309× ≥ 2.0×` and `6/7` downward bands produced `0` new negative flips (`join-106` crossed from `+0.15` nat on `B2` downward, within Phase 1 jitter).
- **L2.2 (Stop rule) — NOT TRIGGERED (Stage 2A PASS; advancing to Stage 2B).**
- **L2.3 (Stage 2B sub-band efficiency — frozen `2026-11-23` in `protocols/PHASE2B.md`, pre-data `Amendment 2B.1`) — COMPLETE (`[E2.5]–[E2.9]`):**
  - Two-shuffle `B_close` jitter scale: pooled per-item `SD = 0.2745` nat, `S_max = 0.1547` nat; within-layer Hessian pick value on P-B KL not detected (`−0.000181`, `−0.000332`).
  - Selected for Phase 3 under `Amendment 2B.1` Rule 5:
    1. **Primary deployment candidate:** `K3.45-B1B2` (`b1b2-up-k4`, `+7.59 GiB`, `−0.006558` P-B KL = `50.2%` K4 headroom, `1.39×` `b2-up-k4` total KL).
    2. **Efficiency candidate:** `K3.35-B2` (`b2-up-k4`, `+3.80 GiB`, `−0.004713` P-B KL = `36.0%` K4 headroom, `1.44×` `b1b2-up-k4` KL/GiB; neither hot-K3 arm beat it at matched bytes).
    3. **Secondary iso-bitrate candidate:** `K3.25-iso-hot` (`iso-b0b5b6-to-hot-b1b2`, `0 B` delta, `−0.002130` P-B KL, CI `[−0.004104, −0.000154]`, no detected negative `B_close` regression).

---

## Phase 3 — Targeted Repair & Held-Out Confirmation (CuteAFD `v6.0.0`, frozen `2026-11-23` in `protocols/PHASE3.md`)

*Amendment to draft L3.2 (2026-11-23, pre-Phase-1):* Original draft required iso-bitrate superiority with 95% CI excluding `0` on 4-pass 127K tasks (`±3.5–4.0 pp` half-width), which would require recovering ~70% of the entire `+0.75 bpw` K4 jump at `+0.00 bpw`. Revised before any GPU run to: (a) make the targeted band-upgrade arm (`K3.35–K3.45-Kintsugi`, 1–2 sink bands at 100% K4) the primary deployment candidate and `K3.25-Kintsugi-iso` the secondary iso-bitrate test; and (b) separate the high-power continuous gate (`L3.1–L3.2`, Stage 3A) from the 127K task confirmation (`L3.3`, Stage 3B, with stated power).

- **L3.1 (Untouched P-A & P-B KL non-inferiority & generalization — Stage 3A) — ALL THREE CANDIDATES PASS (`[E3.1]–[E3.2]`):**
  - `K3.45-B1B2` (`+7.59 GiB`): P-A KL `0.038228` (`−0.008759` vs rebuilt `CI [−0.011693, −0.006086]`, `52.8%` of K4; `−0.008483` vs native `CI [−0.012545, −0.005131]`, `51.2%`).
  - `K3.25-iso-hot` (`0 B`): P-A KL `0.043064` (`−0.003922` vs rebuilt `CI [−0.007087, −0.001175]`, `23.7%`; `−0.003647` vs native `CI [−0.007168, −0.000749]`, `22.0%` — not separated from `K3.35-B2` on P-A, ranked below it on P-B).
  - `K3.35-B2` (`+3.80 GiB`): P-A KL `0.044508` (`−0.002479` vs rebuilt `CI [−0.004236, −0.000892]`, `14.9%`; `−0.002203` vs native `[−0.004490, +0.000264]`).
- **L3.2 (Selection-Held-Out `D_margin_heldout` `n = 36` & Neutral-Author `D_margin_neutral` `n = 24` — Stage 3A, `Amendment 3A.1`) — PASS ON K4-AUTHORED HELD-OUT / SUPPORTIVE ON NEUTRAL (`[E3.3]–[E3.4]`):**
  - `K3.45-B1B2` passes cleanly on selection-held-out `D_margin_heldout` (`35/36` K4-authored) against BOTH controls (`+0.4780` vs rebuilt `CI [+0.2423, +0.7665]`, `80.2%` `[44%, 128%]` of K4; `+0.5607` vs native `CI [+0.2359, +0.9095]`; direct `k4 − k345` not detected).
  - Neutral-author `D_margin_neutral` (`23/24` `exlr8-k4`, `1/24` K3.25 fallback): pre-registered `k4-bf16core − k325-rebuilt = +0.2517` (`CI [−0.0276, +0.5603]`, point estimate consistent with Phase 1 `+0.2773`, not detected at `n = 24`; detected vs native `+0.2705` `[+0.0643, +0.4680]`; `exlr8`-only `n = 23` unchanged). `K3.35-B2` replicated vs matched control (`+0.1764` `[+0.0378, +0.3257]`); `K3.25-iso-hot` cleared its Stage 2B watch item (`+0.0474 / +0.0661`).
- **L3.3 (127K task confirmation, `max_tokens = 32,768` — Stage 3B, 8-pass `Amendment 3B.2`) — ANCHOR GAP DETECTED (`[E3.5]`), `K3.45-B1B2` FAILS `L3.3` (`[E3.6]`):**
  - *Exact inverse-normal 2-stage anchor test (`k4-bf16core − k325-native`, `480` replies/arm):* **`Z_acc_combined = +2.3301 ≥ 1.96` (DETECTED)** (`+3.75 pp` Stage 1, `+3.65 pp` Stage 2; pooled `+3.70 pp`, `CI [+0.55, +7.06]`) and **`Z_unf_combined = +2.8054 ≥ 1.96` (DETECTED)** (`−4.17 pp` both stages; `35/480 → 15/480`, pooled `CI [−7.08, −1.46]`).
  - *`K3.45-B1B2` (`+7.59 GiB`, `L09–L20` K4):* **FAIL (`L3_3_pass = false`, all 8 reps retained untrimmed).** `4K` non-inferior (`98.23%`, `0` unf), `128K` pooled **`86.90%`** (`31/480 = 6.46%` unf): vs `k325-native` **`−0.26 pp`** (`CI [−3.20, +2.76]`, share `−7.0%` `[−263%, +63%]`), unfinished **`−0.83 pp`** (`CI [−3.96, +2.08]`, share `20.0%` `[−100%, +79%]`), finished-only `92.90%` vs `94.02%` tripped the `−1.0 pp` point guardrail (paired both-finished `−1.07 pp` `[−2.36, +0.16]`, not detected); vs `k4-bf16core` `−3.96 pp` (`[−6.74, −1.12]`), unfinished `+3.33 pp` (`[+0.62, +6.04]`). **However, the two halves disagree for `k345-b1b2` alone** (`23/240` unf / `84.06%` in `rep 0–3` vs `8/240` unf / `89.74%` in `rep 4–7`, Fisher `p = 0.008`; anchors `p = 0.73 / 0.60`); cause not identified; whether `B1–B2` recovers the 128K runaway rate is unresolved. Phase 4 (`vLLM v0.9.2`) remains blocked pending Stage 3C.
  - *`K3.25-iso-hot`:* Dropped from Stage 3B task runs pre-data under `Amendment 3B.2` (`~0.9 pp` expected lift unresolvable at affordable pass counts; verified `0 B` P-A/P-B KL gain stands in `[E2.9]`, `[E3.1]`).
- **L3C.1 (Stage 3C-A Loop Entry/Exit Margin Probe — PASS `[E3.8]`, `protocols/PHASE3C.md`):**
  - Frozen manifest (`112` prefixes: `72` dump + `40` stratified controls across `4` authors; `manifest.json` SHA-256 `42bc1f92...`; median pool coverage `0.99997` at `exit_10`).
  - Primary `M_class(exit_10)` (`n = 72` dumps, paired item-cluster bootstrap): **`k4-bf16core − k325-native = −0.4272` nats (`CI [−0.5685, −0.2673]`, `5.5σ`)** and `k4`-authored dumps (`n = 11`) = **`−0.1461 < 0`** → **PASS**.
  - Prospective `k345-b1b2` prediction committed before Stage 3C-B (`results/phase3/STAGE3CA_PREDICTION.json`): primary non-`k345`-authored (`n = 44`) recovery share **`65.0%` (`CI [25.7%, 98.6%]`)**; all-author (`n = 72`, self-authorship-affected) **`62.7%` (`CI [29.7%, 87.8%]`)**.
- **L3C.2 (Stage 3C-B `K3.45-B1B2` Mixed-Regime Replication — COMPLETE `[E3.9]`, `reps 8–11`, `4K+128K` mixed, fresh server, `SEED_BASE_3C = 950M`):**
  - `128K` unfinished count: **`14/240` (`5.83%`; `r8: 5, r9: 3, r10: 3, r11: 3`)**, `128K` accuracy `87.71%` (`fin_only = 93.14%`). Frozen reading: **`unresolved (13–17/240)`** (`L3.3` = FAIL not reopened; pooled 12-rep descriptive `87.17%`, `45/720 = 6.25%` unf).
  - Step 1 prediction consistency: replication unfinished recovery share (`35.0%` vs 8-pass anchors `35/480` vs `15/480`) fell inside Stage 3C-A's recorded primary (`n = 44`) 95% CI (`[25.7%, 98.6%]`, point `65.0%`). Conditional Step 3 awaits separate authorization.
- **L3D.1 (Stage 3D / Step 3A — 7-Band Loop-Exit Scan, FROZEN `2026-11-25`, `protocols/PHASE3D.md`):**
  - Score matched stitched control `k325-rebuilt` + all 7 upward band flips (`B0..B6 → 100% K4`, `+3.80 GiB` each) on the `72` frozen dump prefixes via unmodified `tools/run_loop_probe.py`.
  - Pre-registered additivity & selection rules (`tools/analyze_phase3d.py`):
    - Resolution floor: single-band `SE ~ 0.07–0.08` nats (`~−0.06`/band if uniform → flat scan reads as distributed, not absent).
    - Additivity: check `(B1 + B2)` vs `(k345-b1b2 − rebuilt)` and `S_7 = sum(B0..B6)` vs `(k4 − rebuilt)` (`±25%` = additive; `< 50%` = non-additive/distributed → complement fallback).
    - Author-stratum rank check across all 4 authors; Spearman `ρ` vs Stage 2A `≤ 14.4K` P-B KL.
    - Step 3B candidate rule: advance a new `≤ 3`-band set (distinct from `B1–B2`, preferring `≤ 2` bands) only if `S_7 / Δ_K4 ≥ 0.50`, summed `exit_10` share `≥ 80%`, and both `join` and `ledger` shares `≥ 60%` (separate go-ahead for Step 3B).

---

## Phase 4 — Cross-Engine Carry-Over (`vLLM v0.9.2`, draft — freeze in `protocols/PHASE4.md` before first GPU request)

- **L4.1 (Task carry-over):** Held-out 127K Context-Probe improvement / unfinished-reply reduction carries over on `vLLM v0.9.2` (`candidate` vs native `K3.25-v1`, 4 passes, `max_tokens = 32,768`).
- **L4.2 (Tool-calling non-inferiority):** `tool-eval-bench` (88 scenarios, 3 repeats, seeds 1001–1003) mean score `≥ 159 / 176` (baseline `161.0`).
- **L4.3 (Speed & memory):** C1 code decode (1K–128K) `≥ 95%` of baseline (`145–150 tok/s`) and admitted KV pool `≥ 3.5M` tokens (`≥ 4.5M` for iso-bitrate).
