# Pre-Registered Decision Lines

Following Kintsugi Step 5 (`R1–R6`): every line is written with a date before the run it judges. A wrong line is amended only for later runs, with the reason recorded; missed lines stay failed in the record.

---

## Phase 0 — Stitcher & Loader Contract (registered `2026-11-23`)

- **L0.1 (Writer fidelity):** Rebuilding `K3.25-v1` from its own shards via `tools/stitch_exl3.py` produces bit-identical tensors on 100% of checked projections and identical `tensor_storage` metadata.
- **L0.2 (Byte budget):** `K3.25-rebuilt` (stitched from `K3-v1` + `K4-Core-K6` + `K3.25-v1` BF16 core using `K3.25-v1`'s exact `9,072`-projection map) matches `K3.25-v1`'s total tensor byte count within `0.1%`.
- **L0.3 (Planner admission):** CuteAFD `v6.0.0` CPU planner returns `READY to serve` with BF16 core on: (a) native `K3.25-v1`, (b) `K3.25-rebuilt`, (c) a test variant with one 6-layer band at `100%` K4, and (d) a test variant with one 6-layer band at `0%` K4.

---

## Phase 1 — Determinism, Library Offset & Probe Validation (CuteAFD `v6.0.0`, draft lines — freeze in `protocols/PHASE1.md` before first GPU request)

- **L1.1 (Step 0b determinism):** Two separate CuteAFD `v6.0.0` launches of `K3.25-v1` show max per-item margin spread `< 0.05` nat (target `0.0000`) on `D_margin_build` and `0.0` P-B KL difference.
- **L1.2 (Probe validity):** `D_margin_build` ranks `K4-bf16core` above `K3.25-rebuilt` with paired bootstrap 95% CI excluding `0`, and at least 5 items in `D_margin_build` are fragile (`m_i < 1.0` nat) or failing (`m_i < 0`) on `K3.25-rebuilt`.
- **L1.3 (Library calibration check):** `|KL_PB(K3.25-rebuilt) − KL_PB(K3.25-v1)| ≤ 0.003` nats. If exceeded, record the library offset and diagnose before Phase 2.

---

## Phase 2 — Layer-Band Damage Map (CuteAFD `v6.0.0`, draft lines — freeze in `protocols/PHASE2.md` before first GPU request)

- **L2.1 (Non-flat sensitivity):** Across the 7 six-layer bands (`L03–L44`), the ratio between highest-sensitivity sink band and lowest-sensitivity safe source band is `≥ 2.0×` in either $\Delta m/\text{MiB}$ or $\Delta\text{KL}/\text{MiB}$, and at least one source band produces `0` new negative-margin flips (`hit list = 0`).
- **L2.2 (Stop rule):** If L2.1 fails (sensitivity is flat across bands within noise), stop the allocation search and publish a negative result.

---

## Phase 3 — Iso-Bitrate Repair & Held-Out Confirmation (CuteAFD `v6.0.0`, draft lines — freeze in `protocols/PHASE3.md` before first GPU request)

- **L3.1 (Static non-inferiority):** On untouched P-A and P-B windows, `K3.25-Kintsugi-iso` does not regress P-A KL by more than `+0.003` nats or P-B KL by more than `+0.002` nats vs `K3.25-rebuilt` (and vs `K3.25-v1`).
- **L3.2 (Held-out task superiority — ship line):** On the held-out 60-item Context-Probe (`127K`, 4 passes, paired item bootstrap), `K3.25-Kintsugi-iso` improves over *both* `K3.25-rebuilt` and `K3.25-v1` with 95% CI excluding `0`.

---

## Phase 4 — Cross-Engine Carry-Over (`vLLM v0.9.2`, draft lines — freeze in `protocols/PHASE4.md` before first GPU request)

- **L4.1 (Task carry-over):** Held-out 127K Context-Probe improvement holds on `vLLM v0.9.2` (`K3.25-Kintsugi-iso − K3.25-v1`, 4 passes, 95% CI excluding `0`).
- **L4.2 (Tool-calling non-inferiority):** `tool-eval-bench` (88 scenarios, 3 repeats, seeds 1001–1003) mean score `≥ 159 / 176` (baseline `161.0`).
- **L4.3 (Speed & memory):** C1 code decode (1K–128K) `≥ 95%` of baseline (`145–150 tok/s`) and admitted KV pool `≥ 4.5M` tokens (for iso-bitrate).
