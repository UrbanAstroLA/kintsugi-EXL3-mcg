# Protocol: Phase 1 — Determinism, Library Calibration & Two-Component Probe Validation (`2026-11-23-kintsugi-phase1`)

**Status:** Frozen before first GPU request (`FROZEN.json` records SHA-256 of this protocol, frozen probe manifest, drivers, and orchestrator prior to launch).  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, image `ghcr.io/tpurtell/cuteafd-coordinator:v6.0.0@sha256:f2bd29c76f8810e078eff0e1aed390427908fae944932cc6ca7dfb98071fa415`), 2× RTX PRO 6000 Blackwell Max-Q (300 W), local experts, score form (`ENABLE_BENCH=on`, `PREFIX_CACHE_ENTRIES=0`, `prefill_scoring=streamed`).  
**Between-phase serving policy:** Standing vLLM is stopped before launch 1 and **not** restored at Phase 1 exit (GPUs left idle and clear for Phase 2).

---

## 1. Questions & Pre-Registered Lines (`Gate P1`)

1. **L1.1 (Step 0b determinism):** Two separate CuteAFD `v6.0.0` score launches of native `K3.25-v1` (`r0` and `r1`) must show max per-item margin spread `< 0.05` nat (target `0.0000`) on `D_margin_build` (both Component A and Component B) and `0.0` P-B KL difference (`[0.0, 0.0]`, bit-equal rows).
2. **L1.2 (Two-component probe validity):** On `D_margin_build` (`n = 24` items at ~133K tokens: `8 join`, `8 ledger`, `8 trace`; stratified to hold half of historical K3.25 127K cap-hit items, strictly disjoint from the `n = 36` `D_margin_heldout` split untouched until Phase 3):
   - **Component A (Answer-value margins):** exact teacher-forced margin $m_t = \log p(y_t) - \max_{y \neq y_t}\log p(y)$ across answer-value tokens on the final `ANSWER: {...}` line (item summary: `min_margin` and `mean_margin`).
   - **Component B (Termination / `</think>` margins):** exact margin at `</think>` (`154842`) plus the 8 immediately preceding reasoning tokens along verified finished K4 trajectories (including items where K3.25 historically exhausted the 32,768-token cap).
   - **Validation rule (Kintsugi Step 2):** A component is retained for Phase 2 only if it ranks `k4-bf16core` above `k325-rebuilt` with 10,000-draw paired item bootstrap 95% CI excluding `0`, AND `k325-rebuilt` exhibits at least 5 fragile (`m < 1.0` nat) or negative (`m < 0`) items on that component.
3. **L1.3 (Library calibration offset):** `|KL_PB(k325-rebuilt) − KL_PB(k325-native)| ≤ 0.003` nats. Because `K3-v1`, `K3.25-v1`, and `K4-Core-K6` came from separate quantization runs (`[E0.2]`), this directly measures any calibration-pool offset before Phase 2.

---

## 2. Arms (5 CuteAFD `v6.0.0` Score Launches)

| Step | Arm | Checkpoint | Captures |
|---|---|---|---|
| 1 | `k325-native` (`r0`) | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@0490d2f7` | Smoke + `D_margin_build` + P-B + P-A |
| 2 | `k325-native` (`r1`) | Same (fresh container restart) | Smoke + `D_margin_build` + P-B |
| 3 | `k325-rebuilt` (`r0`) | Stitched (`K3-v1` + `K4-Core-K6` + `K3.25-v1` core, exact 9,072 K4 map) | Smoke + `D_margin_build` + P-B + P-A |
| 4 | `k4-bf16core` (`r0`) | Stitched (all `37,152` K4 from `K4-Core-K6` + `K3.25-v1` core) | Smoke + `D_margin_build` + P-B + P-A |
| 5 | `k3-bf16core` (`r0`) | Stitched (all `37,152` K3 from `K3-v1` + `K3.25-v1` core) | Smoke + `D_margin_build` + P-B + P-A |

- **Inline margin scoring:** `/v1/bench/probe` with `cold: true`, `score_path: "prefill"`, `top_k: 5`, `score_positions`, and `want` (zero logit dumps).
- **Static KL scoring:** P-B (`cuteafd bench fidelity run --tier full --score-path prefill`, `glm5_flash-v2_20261005_bf16root@a7e7d1b4`) and P-A (`glmf-teacher-kl.py --path prefill`, 25 × 2K teacher panel), scored with `score_pb.py`, `validate_pb.py`, and `score_pa.py`; raw FP32 dump directories deleted immediately after CPU scoring and validation.

---

## 3. Budget & Gates

- **Estimated GPU time:** `~1.5–1.8 GPU-h` (~15–20 min per launch across 5 launches).
- **Estimated wall time:** `~2.0 h` (including CPU builds of `k4-bf16core` and `k3-bf16core`, ~5 min).
- **Kill / Pivot rule:** If `L1.1` fails (non-deterministic prefill margins), stop and diagnose engine determinism. If neither Component A nor Component B passes `L1.2`, revise the probe before spending Phase 2 GPU hours. If `L1.3` exceeds `0.003` nats, inspect `k325-rebuilt − k325-native` before interpreting Phase 2 flips.
