# Phase 2 Stage 2B Summary — Shuffle Controls, Role Splits, Routing-Targeted Hot-K3 & Iso-Bitrate Swaps

**Date:** 2026-11-23  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Protocol:** [`protocols/PHASE2B.md`](../../protocols/PHASE2B.md) (frozen with pre-data `Amendment 2B.1` before Flip 1 completed)  
**Matched control:** Phase 1 `k325-rebuilt-r0` (`B_close = +3.6003` nats, P-B `gen` KL = `0.044447` nats)  
**Scope:** P-B fidelity (`≤ 14.4K` tokens, `15,104` generated rows across `53` clusters) + `D_margin_build` (`n = 24` items, `127K`); transfer to untouched P-A, `D_margin_heldout` (`n = 36`), neutral-author trajectories, and `4K–127K` tasks is tested in Phase 3.

---

## 1. Complete 10-Flip Stage 2B Matrix (Matched Against `k325-rebuilt-r0`)

*(All 10 flips passed `stitch_exl3.py verify` with `0` header/SHA mismatches and `validate_pb.py` same-dump verification with `32,768/32,768` records, `0` mismatches, and `0.0` max KL/NLL diff.)*

| Arm | Category | ΔGiB | bpw | `ΔKL_PB` (`gen`, 95% CI) | KL Headroom Share | `−ΔKL / GiB` | `ΔB_close` (`n=24`, 95% CI) | `B_close` Detected (`2B.1`)? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `null-shuf-b2` | Shuffle (`B2`) | `0.00` | `3.2442` | `−0.000181` `[−0.001763, +0.001303]` | `+1.4%` | — | `−0.1547` `[−0.2894, −0.0213]` | *(sets `S_max = 0.1547`)* |
| `null-shuf-b5` | Shuffle (`B5`) | `0.00` | `3.2442` | `−0.000332` `[−0.001128, +0.000492]` | `+2.5%` | — | `+0.0039` `[−0.0652, +0.0798]` | *(two-shuffle control)* |
| `b2-role-gate` | Role (`B2`) | `+1.45` | `3.2842` | `−0.002340` `[−0.003556, −0.001123]` | `+17.9%` | `0.001614` | `−0.0171` `[−0.1884, +0.1944]` | Not detected |
| `b2-role-up` | Role (`B2`) | `+1.29` | `3.2798` | `−0.002419` `[−0.004065, −0.000820]` | `+18.5%` | `0.001872` | `−0.0592` `[−0.1714, +0.0597]` | Not detected |
| `b2-role-down` | Role (`B2`) | `+1.05` | `3.2733` | `−0.001328` `[−0.002612, −0.000067]` | `+10.2%` | `0.001259` | `+0.0416` `[−0.1031, +0.2103]` | Not detected |
| `b1b2-hot-route-3888` | Hot-K3 (`B1+B2`) | `+3.80` | `3.3488` | `−0.004482` `[−0.006018, −0.002939]` | `+34.3%` | `0.001180` | `+0.0418` `[−0.1558, +0.2222]` | Not detected |
| `b1b2-hot-routed-err-3888` | Hot-K3 (`B1+B2`) | `+3.80` | `3.3488` | `−0.004270` `[−0.005653, −0.002878]` | `+32.7%` | `0.001125` | `−0.1509` `[−0.2987, −0.0100]` | Not detected (`≤ S_max`) |
| `iso-b0b5b6-to-b2` | Iso-bitrate (`0 B`) | `0.00` | `3.2442` | `−0.001411` `[−0.003195, +0.000519]` | `+10.8%` | — | `−0.0136` `[−0.1809, +0.1577]` | Not detected |
| **`iso-b0b5b6-to-hot-b1b2`** | **Iso-bitrate (`0 B`)** | **`0.00`** | **`3.2442`** | **`−0.002130` `[−0.004104, −0.000154]`** | **`+16.3%`** | — | `−0.1720` `[−0.4410, +0.0884]` | Not detected (CI incl. `0`) |
| **`b1b2-up-k4`** | **Full `B1+B2`** | **`+7.59`** | **`3.4535`** | **`−0.006558` `[−0.008230, −0.004803]`** | **`+50.2%`** | `0.000864` | `+0.1731` `[−0.0572, +0.4197]` | Not detected (CI incl. `0`) |
| *Ref: `b2-up-k4` (2A)* | *Full `B2` (2A)* | *`+3.80`* | *`3.3488`* | *`−0.004713` `[−0.006613, −0.002964]`* | *`+36.0%`* | *`0.001241`* | *`+0.1041` `[−0.0480, +0.2668]`* | *Not detected* |

### Direct Head-to-Head P-B `gen` KL Contrasts (`53` clusters, `B − A`)

| Contrast (`B − A`) | `KL(A)` | `KL(B)` | `ΔKL_PB (B − A)` | Paired 95% CI | Reading |
|---|---:|---:|---:|---:|---|
| `b1b2-hot-route-3888 − b2-up-k4` | `0.039734` | `0.039965` | `+0.000231` | `[−0.001117, +0.001593]` | Not detected (matched `+3.80 GiB`) |
| `b1b2-hot-routed-err-3888 − b2-up-k4` | `0.039734` | `0.040177` | `+0.000443` | `[−0.001091, +0.002095]` | Not detected (matched `+3.80 GiB`) |
| `iso-b0b5b6-to-hot-b1b2 − iso-b0b5b6-to-b2` | `0.043036` | `0.042317` | `−0.000719` | `[−0.002330, +0.001035]` | Not detected (`0 B` matched) |
| **`b1b2-up-k4 − b2-up-k4`** | `0.039734` | **`0.037889`** | **`−0.001845`** | **`[−0.003453, −0.000250]`** | **Detected improvement (`B1` adds real KL gain over `B2`)** |

---

## 2. Key Empirical Findings (`[E2.5]–[E2.9]`)

1. **Within-layer Hessian picks vs random picks (`[E2.5]`):**
   Randomly permuting `K3.25-v1`'s K4 selections inside `B2` (`938` swapped projections) or `B5` (`942` swapped projections) produced **no detected P-B KL degradation** (`−0.000181`, CI `[−0.001763, +0.001303]` in `B2`; `−0.000332`, CI `[−0.001128, +0.000492]` in `B5`). On `B_close`, the **two shuffles** yield a pooled per-item SD of **`0.2745` nat** (`SE_24 = 0.0560`) and **`S_max = 0.1547` nat** (`null-shuf-b2 = −0.1547`, CI `[−0.2894, −0.0213]`), confirming that single-band `B_close` shifts of `~±0.15` nat occur at zero extra bytes.
2. **All three projection roles in `B2` contribute (`[E2.6]`):**
   Every individual role upgrade inside `B2` significantly reduced P-B KL: `up_proj` (`−0.002419` / `0.001872` per GiB), `gate_proj` (`−0.002340` / `0.001614` per GiB), and `down_proj` (`−0.001328` / `0.001259` per GiB, smaller because `K3.25-v1` already gave `down_proj` `50%` K4 vs `19%` for `gate_proj` and `31%` for `up_proj`).
3. **Depth concentration dominates within-layer calibration routing at `+3.80 GiB` (`[E2.7]`):**
   At exact matched `+3,888 MiB` (`3.3488 bpw`), neither calibration-routing-targeted arm (`b1b2-hot-route-3888`: `−0.004482`; `b1b2-hot-routed-err-3888`: `−0.004270`) beat simpler full-band `b2-up-k4` (`−0.004713`; paired differences `+0.000231` and `+0.000443`, both CIs spanning zero). Combined with `[E2.5]`, **cross-depth band allocation (`B1–B2` vs `B5–B6`) carries far more P-B KL leverage than within-layer calibration ranking**.
4. **`K3.45-B1B2` (`b1b2-up-k4`) captures `50.2%` of the K3.25→K4 P-B KL gap in `27.7%` of the bytes (`[E2.8]`):**
   Upgrading `B1 + B2 (L09–L20)` to `100%` K4 (`+7.59 GiB`, `3.4535 bpw`) reduced P-B `gen` KL from `0.044447` to **`0.037889`** (`ΔKL_PB = −0.006558`, `CI [−0.008230, −0.004803]`, `7.5σ`), significantly outperforming `b2-up-k4` (`−0.001845`, `CI [−0.003453, −0.000250]`, `1.39×` total KL reduction).
5. **Iso-bitrate (`0 B`) P-B KL gain is detected (`[E2.9]`):**
   Moving `3,888` K4 projections out of `B0 + B5 + B6` (`0%` K4) into the top `3,888` hot-K3 projections across `B1 + B2` (`iso-b0b5b6-to-hot-b1b2`, exact `0 B` delta / `3.2442 bpw`) reduced P-B `gen` KL by **`−0.002130` nats (`95% CI [−0.004104, −0.000154]`, `16.3%` of K4 headroom)** with no detected negative `B_close` regression (`−0.1720`, CI `[−0.4410, +0.0884]` includes `0`).

---

## 3. Pre-Registered `L2.3` (`Amendment 2B.1`) Phase 3 Candidate Selection

Applying `Amendment 2B.1` Rule 5 mechanically without post-hoc adjustment:
1. **`best_3888` (`+3.80 GiB`):** **`K3.35-B2` (`b2-up-k4`)** — numerically `#1` at `+3,888 MiB` (`−0.004713`); neither hot-K3 arm beat it with CI excluding `0`.
2. **`K3.35` vs `K3.45` advancement:**
   - Total KL ratio `b1b2-up-k4 / b2-up-k4 = 1.392× ≥ 1.25×` → **`K3.45-B1B2` (`b1b2-up-k4`, `+7.59 GiB`) advances as PRIMARY deployment candidate**.
   - Efficiency ratio `(b2-up-k4 KL/GiB) / (b1b2-up-k4 KL/GiB) = 1.437× ≥ 1.25×` → **`K3.35-B2` (`b2-up-k4`, `+3.80 GiB`) ALSO advances as efficiency candidate**.
3. **Secondary iso-bitrate candidate (`0 B`):** **`K3.25-iso-hot` (`iso-b0b5b6-to-hot-b1b2`, `0 B` delta)** — sole iso-bitrate arm whose P-B KL 95% CI vs `k325-rebuilt` is strictly `< 0` (`[−0.004104, −0.000154]`) with no detected negative `B_close` regression.
