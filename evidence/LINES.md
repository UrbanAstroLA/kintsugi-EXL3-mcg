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
  3. **Confirmatory `B_close` + null calibration — ALIGNED (null controls queued in Stage 2B):** `ρ(up B_close, −ΔKL_PB) = +0.5357`; `B1` (`+0.2045` nat, `73.8%` headroom) and `B2` (`+0.1041` nat, `37.5%`) rank `#1–#2`. Two within-band shuffle null controls (`null-shuf-b2`, `null-shuf-b5`) run first in Stage 2B (`protocols/PHASE2B.md`).
- **L2.1 (Original, archived verbatim; superseded by `L2.1-v2`) — PASS (`[E2.1]–[E2.3]`):** Upward P-B KL ratio `4.309× ≥ 2.0×` and `6/7` downward bands produced `0` new negative flips (`B2` was sole band with `1` new negative flip, `join-106`).
- **L2.2 (Stop rule) — NOT TRIGGERED (Stage 2A PASS; advancing to Stage 2B).**
- **L2.3 (Stage 2B sub-band efficiency — frozen `2026-11-23` in `protocols/PHASE2B.md`):** On top sink bands `B1–B2 (L09–L20)` and cheapest source pool `B0+B5+B6`, run `10` flips: `2` within-band shuffle null controls (`null-shuf-b2`, `null-shuf-b5`), `3` `B2` role splits (`gate`/`up`/`down`), `2` routing-targeted hot-K3 upgrades in `B1+B2` at `+3,888 MiB` (`3.3488 bpw`, pure routing mass vs `mass × h_rel`), `2` iso-bitrate swaps (`B0+B5+B6 → B2` and `B0+B5+B6 → hot B1+B2`, `0 B` delta), and `1` combined `B1+B2 → 100% K4` (`+7.59 GiB`, `3.4477 bpw`). Advance the best-efficiency targeted upgrade (`K3.35` / `K3.45`) and best iso-bitrate candidate (`K3.25-Kintsugi-iso`) that beat the shuffle null spread cleanly without P-B KL regression into Phase 3.

---

## Phase 3 — Targeted Repair & Held-Out Confirmation (CuteAFD `v6.0.0`, draft — freeze in `protocols/PHASE3.md` before first GPU request)

*Amendment to draft L3.2 (2026-11-23, pre-Phase-1):* Original draft required iso-bitrate superiority with 95% CI excluding `0` on 4-pass 127K tasks (`±3.5–4.0 pp` half-width), which would require recovering ~70% of the entire `+0.75 bpw` K4 jump at `+0.00 bpw`. Revised before any GPU run to: (a) make the targeted band-upgrade arm (`K3.35–K3.45-Kintsugi`, 1–2 sink bands at 100% K4) the primary deployment candidate and `K3.25-Kintsugi-iso` the secondary iso-bitrate test; and (b) separate the high-power continuous gate (`L3.2`) from the 127K task confirmation (`L3.3`, with stated power).

- **L3.1 (Static non-inferiority):** On untouched P-A and P-B windows, the candidate does not regress P-A KL by more than `+0.003` nats or P-B KL by more than `+0.002` nats vs `K3.25-rebuilt` and `K3.25-v1`.
- **L3.2 (Held-out continuous margin improvement):** On untouched `D_margin_heldout` (validated components from Phase 1), the candidate improves over both `K3.25-rebuilt` and `K3.25-v1` with paired bootstrap 95% CI excluding `0`.
- **L3.3 (127K task confirmation, `max_tokens = 32,768`):** On the 60-item Context-Probe (`127K`, 4 passes; MDE `~±3.5–4.0 pp`), report `candidate − K3.25-v1`, `candidate − K3.25-rebuilt`, `candidate − K4-bf16core`, unfinished counts (`cap` / `repetition`), and exact GiB cost. Advance to Phase 4 if `L3.1` and `L3.2` pass and 127K task score / unfinished counts move toward `K4-bf16core` without regression (extend to 8 passes if borderline within `±2.5 pp`).

---

## Phase 4 — Cross-Engine Carry-Over (`vLLM v0.9.2`, draft — freeze in `protocols/PHASE4.md` before first GPU request)

- **L4.1 (Task carry-over):** Held-out 127K Context-Probe improvement / unfinished-reply reduction carries over on `vLLM v0.9.2` (`candidate` vs native `K3.25-v1`, 4 passes, `max_tokens = 32,768`).
- **L4.2 (Tool-calling non-inferiority):** `tool-eval-bench` (88 scenarios, 3 repeats, seeds 1001–1003) mean score `≥ 159 / 176` (baseline `161.0`).
- **L4.3 (Speed & memory):** C1 code decode (1K–128K) `≥ 95%` of baseline (`145–150 tok/s`) and admitted KV pool `≥ 3.5M` tokens (`≥ 4.5M` for iso-bitrate).
