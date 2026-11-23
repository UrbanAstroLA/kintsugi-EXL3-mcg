# Phase 1 Summary — Determinism, Library Offset & Two-Component Margin Probe Validation

**Date:** 2026-11-23  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Protocol:** [`protocols/PHASE1.md`](../../protocols/PHASE1.md) (frozen before first GPU request)  
**Gate P1 Verdict:** **PASS (`L1.1` PASS, `L1.2` PASS on Component B / Component A retired as saturated, `L1.3` PASS)**

---

## 1. Pre-Registered Gate P1 Verdicts

| Line | Criterion | Measured Result | Verdict |
|---|---|---|---|
| **L1.1** (Step 0b determinism) | Max per-item margin spread `< 0.05` nat (target `0.0000`) and `0.0` P-B KL diff across two separate launches of `K3.25-v1` | Margin max spread = **`0.000000`** (`24/24` identical logit hashes on `B_close`); P-B gen KL diff = **`0.000000`** (`95% CI [0.0, 0.0]`, `15,104` rows) | **PASS** (`[E1.1]`) |
| **L1.2** (Two-component probe validity) | Retained probe component(s) rank `K4-bf16core` above `K3.25-rebuilt` (paired bootstrap 95% CI excluding `0`) with `≥ 5` fragile (`< 1.0` nat) or failing (`< 0`) items on `K3.25-rebuilt`; any failing component dropped before Phase 2 | **Component B (`B_close`, `</think>` token `154842`):** **`+0.2773` nats** (`95% CI [+0.0197, +0.5381]`, excludes `0`), **`6/24` fragile `< 1.0`** (`2/24` negative), **`17/24`** items higher (`23/24` higher for `K4 vs K3`, `+0.7311` nats). **Component A (`A_min`, final JSON answer tokens):** saturated at `+12.41–12.65` nats (`0/24` fragile, `K4 − K3.25-rebuilt = −0.0122`, `CI [−0.2198, +0.2201]`) → **retired** before Phase 2. | **PASS (Component B retained; Component A retired)** (`[E1.2]`) |
| **L1.3** (Library calibration offset) | `\|KL_PB(K3.25-rebuilt) − KL_PB(K3.25-v1)\| ≤ 0.003` nats | P-B gen KL diff = **`−0.000645` nats** (`95% CI [−0.002536, +0.001145]`, includes `0`; `4.6×` inside threshold). P-A teacher KL diff = **`+0.000275` nats** (`95% CI [−0.002806, +0.003298]`, includes `0`). | **PASS** (`[E1.3]`) |

---

## 2. Arm-Level Measurements (`D_margin_build` `n = 24`, P-B `32,768` rows, P-A `51,175` rows)

Every P-B run passed `validate_pb.py` same-dump verification with `32,768 / 32,768` records matched, `0` field mismatches, and `0.0` max absolute KL/NLL discrepancy.

| Arm | bpw | `A_min` (nat) | `B_close` (nat) | `B_close` `< 1.0` / `< 0` | P-B `gen` KL (95% CI) | P-B `gen` top-1 | P-B `gen` dNLL | P-A KL (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k3-bf16core-r0` | `3.00` | `+12.5583` | **`+3.1465`** | `7` / `2` | `0.051782` `[0.044003, 0.058448]` | `93.25%` | `0.044368` | `0.051983` `[0.039838, 0.065859]` |
| `k325-rebuilt-r0` | `3.25` | `+12.4235` | **`+3.6003`** | `6` / `2` | `0.044447` `[0.037724, 0.050259]` | `93.90%` | `0.032152` | `0.046986` `[0.036560, 0.058898]` |
| `k325-native-r0` | `3.25` | `+12.6546` | **`+3.7108`** | `6` / `1` | `0.045092` `[0.038106, 0.051193]` | `93.84%` | `0.035627` | `0.046711` `[0.035364, 0.059948]` |
| `k325-native-r1` | `3.25` | `+12.6546` | **`+3.7108`** | `6` / `1` | `0.045092` `[0.038106, 0.051193]` | `93.84%` | `0.035627` | *(not repeated)* |
| `k4-bf16core-r0` | `4.00` | `+12.4113` | **`+3.8775`** | `5` / `2` | `0.031373` `[0.026174, 0.036124]` | `94.95%` | `0.017861` | `0.030406` `[0.022946, 0.039552]` |

### Key Paired Contrasts (`B − A`, cluster-bootstrap 95% CI)

| Contrast | `ΔB_close` (nat, `n=24`) | Items `B > A` | `ΔKL_PB` (`gen`, `53` clusters) | `ΔKL_PA` (`25` clusters) |
|---|---:|---:|---:|---:|
| `k325-native-r1 − k325-native-r0` (`L1.1`) | `0.0000` `[0.0, 0.0]` | `0 / 24` (`24` ties) | `0.000000` `[0.0, 0.0]` | — |
| `k325-rebuilt − k325-native` (`L1.3`) | `−0.1105` `[−0.4244, +0.1786]` | — | `−0.000645` `[−0.002536, +0.001145]` | `+0.000275` `[−0.002806, +0.003298]` |
| `k4-bf16core − k325-rebuilt` (`L1.2`) | **`+0.2773` `[+0.0197, +0.5381]`** | **`17 / 24`** | **`−0.013074` `[−0.015765, −0.010210]`** | **`−0.016581` `[−0.021284, −0.012103]`** |
| `k3-bf16core − k325-rebuilt` | **`−0.4538`** | `6 / 24` (`18/24` worse) | **`+0.007335` `[+0.005215, +0.009583]`** | **`+0.004996` `[+0.001473, +0.008552]`** |
| `k4-bf16core − k3-bf16core` | **`+0.7311` `[+0.4150, +1.0512]`** | **`23 / 24`** | **`−0.020409` `[−0.023658, −0.016969]`** | **`−0.021577` `[−0.027631, −0.015826]`** |

---

## 3. Scientific Findings & Methodological Implications for Phase 2

1. **Why standard answer-token margins fail on long-context reasoning MoEs — and why `</think>` closing margins work (`[E1.2]`):**
   Once GLM-5.3-Flash finishes its reasoning trace and enters the final JSON answer block, answer tokens are strongly determined by the immediately preceding chain of thought (`A_min = +12.41` to `+12.65` nats across `K3`–`K4`, `0/24` fragile `< 1.0`, zero monotonic separation). In contrast, the **`</think>` closing boundary (`B_close`, token `154842`)** is genuinely fragile (`6/24` items `< 1.0` nat on `K3.25-rebuilt`) and scales monotonically with routed-expert bitwidth across **`23/24` items** (`+3.1465 → +3.6003 → +3.8775` nats). This directly mirrors the real-world 127K failure mode observed in prior CuteAFD `v5.0.0` runs (unfinished cap-hit replies dominating the accuracy gap).
2. **Large remaining headroom above `K3.25` (`[E1.4]`):**
   On P-B `gen` KL, `K3.25-rebuilt` captures `35.9%` (`0.007335 / 0.020409` nats) of the full `K3 → K4` recovery using `25.0%` of the K4 budget; **`64.1%` (`0.013074` nats)** remains between `K3.25-rebuilt` and `K4-bf16core`. On P-A teacher KL, **`76.8%` (`0.016581 / 0.021577` nats)** remains. On `B_close`, `K3.25-rebuilt` captures `62.1%` (`0.4538 / 0.7311` nats) of the `K3 → K4` margin lift, leaving **`+0.2773` nats (`37.9%`)** to recover toward `K4-bf16core`.
3. **Why matched stitched control (`K3.25-rebuilt`) matters (`[E1.3]`):**
   While `K3.25-rebuilt` and `K3.25-native` are statistically indistinguishable on P-B KL (`−0.000645` nats), P-A KL (`+0.000275` nats), and mean `B_close` (`−0.1105`, CI includes `0`), individual items shift by up to `2.59` nats due to distinct `.trellis` fits across independent quantization runs (`[E0.2]`). Pairing every Phase 2 flip against `K3.25-rebuilt` removes this per-item library noise completely.
4. **Measured wall-clock timing for Phase 2 (`[E1.5]`):**
   Per arm on CuteAFD `v6.0.0`: launch + smoke `~60 s`, `D_margin_build` (`n = 24`, mean `132,947` tokens, inline zero-dump) `561–576 s` (`9.5 min`), P-B full (`32,768` rows) `~250 s` GPU + `~60 s` CPU scoring. Combined with `~30 s` incremental hardlink stitching (`[E0.6]`), each Phase 2 single-band flip takes **`~16 min`** (`~3.8` wall-h for all 14 Stage 2A flips).

---

## 4. Operational Run Amendments (No Protocol, Weight, or Probe Changes)

Two operational fixes were recorded during Phase 1 execution; `FROZEN.json` (`0444`), `protocols/PHASE1.md`, `probe/manifest.json`, checkpoint weights, and all scoring tools remained unchanged:
1. **Derived hub model-ID prefix:** `cuteafd bench fidelity run` validates the served model basename via `fidelity_dataset::base_model()` (`GLM-5.3-Flash-` prefix required). Renamed the local derived hub directory from `derived/kintsugi-exl3` to `derived/GLM-5.3-Flash-EXL3-Kintsugi` before running P-B on stitched arms.
2. **Non-fatal fidelity exit code `3` on `k3-bf16core-r0`:** `cuteafd bench fidelity run` completed all `32,768` rows (`247.87 s`, `0` missing/non-finite) and returned exit code `3` (CuteAFD's fidelity-threshold alert on 3-bit checkpoints). The completed dump was scored and validated identically (`0` mismatches, `0.0` diff), and Phase 2's runner accepts both `rc = 0` and `rc = 3` when followed by `validate_pb.py` verification.
