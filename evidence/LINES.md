# Pre-Registered Decision Lines

Following Kintsugi Step 5 (`R1–R6`): every line is written with a date before the run it judges. A wrong line is amended only for later runs, with the reason recorded; missed lines stay failed in the record.

---

## Phase 0 — Stitcher & Loader Contract (registered `2026-11-23`; **ALL PASS**)

- **L0.1 (Writer fidelity) — PASS (`[E0.5]`):** Rebuilding `K3.25-v1` from its own shards via `tools/stitch_exl3.py` produced `0` header mismatches across all `150,226` tensors and `0` SHA-256 mismatches across `2,215` checked tensors (`109.6 s`).
- **L0.2 (Byte budget) — PASS (`[E0.6]`):** `K3.25-rebuilt` matches `k325-selfcheck`'s exact tensor byte total (`146,178,965,496 B`, `0 B` delta; `−0.0137%` vs `K3.25-v1`'s `index.metadata.total_size`, well within `0.1%`), with `0` header/SHA mismatches (`178.4 s`). Incremental band flips (`test-b0-k4`, `test-b0-k3`) hardlinked `38/44` shards and rewrote `6` layers in `35.9 s` and `26.9 s` with `0` mismatches.
- **L0.3 (Planner admission) — PASS (`[E0.7]`):** CuteAFD `v6.0.0` CPU planner (`cuteafd plan --layout --rtx 2 --spark-ranks 0`, zero GPU) returned `READY to serve` with `CheckpointPrecision` BF16/F32 core (`fp8-prefill: Unchanged`) and `glmf:exl3-k34` on all four checkpoints: `k325-native` (`6,259,200` KV tok), `k325-rebuilt` (`6,259,200` KV tok, layout-identical), `test-b0-k4` (`5,592,064` KV tok), and `test-b0-k3` (`6,481,920` KV tok).

---

## Phase 1 — Determinism, Library Offset & Two-Component Margin Probe Validation (CuteAFD `v6.0.0`, draft — freeze in `protocols/PHASE1.md` before first GPU request)

- **L1.1 (Step 0b determinism):** Two separate CuteAFD `v6.0.0` launches of `K3.25-v1` show max per-item margin spread `< 0.05` nat (target `0.0000`) on `D_margin_build` and `0.0` P-B KL difference.
- **L1.2 (Two-component probe validity):** Each retained component of `D_margin_build` (Component A: answer-token margins; Component B: `</think>` / termination margins on finished `K4` trajectories from items where `K3.25` capped) ranks `K4-bf16core` above `K3.25-rebuilt` with paired bootstrap 95% CI excluding `0`, with at least 5 fragile (`m_i < 1.0` nat) or failing (`m_i < 0`) items on `K3.25-rebuilt`. Any component failing L1.2 is dropped before Phase 2.
- **L1.3 (Library calibration check):** `|KL_PB(K3.25-rebuilt) − KL_PB(K3.25-v1)| ≤ 0.003` nats. If exceeded, record the library offset and diagnose before Phase 2.

---

## Phase 2 — Layer-Band Damage Map (CuteAFD `v6.0.0`, draft — freeze in `protocols/PHASE2.md` before first GPU request)

- **L2.1 (Non-flat sensitivity):** Across the 7 six-layer bands (`L03–L44`), the ratio between highest-sensitivity sink band and lowest-sensitivity safe source band is `≥ 2.0×` in either $\Delta m/\text{MiB}$ or $\Delta\text{KL}/\text{MiB}$, and at least one source band produces `0` new negative-margin flips (`hit list = 0`).
- **L2.2 (Stop rule):** If L2.1 fails (sensitivity is flat across bands within noise), stop the allocation search and publish a negative result.
- **Stage 2B draft note (`[E0.8]`):** Following Stage 2A, Stage 2B tests whether routing-targeted sub-band upgrades (top-routed remaining K3 experts in top sink bands, plus iso-bitrate cold-to-hot swaps) recover a high fraction of full-band margin gain at lower byte cost (exact ranking rules and thresholds to be frozen in `protocols/PHASE2.md` before Phase 2 GPU runs).

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
