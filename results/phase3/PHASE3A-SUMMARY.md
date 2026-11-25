# Phase 3 Stage 3A Summary — Untouched P-A, Held-Out `B_close` & Independent-Quantizer Neutral-Author Check

**Date:** 2026-11-24  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Protocol:** [`protocols/PHASE3.md`](../../protocols/PHASE3.md) (frozen with pre-data `Amendment 3A.1`, swapping `probe_neutral/` atomically to published independent-quantizer `exlr8-k4` — `23/24` `exlr8-k4` + `1/24` `k325-fp8c` fallback — before any neutral request ran)  
**Stage 3A Verdict:** **ALL THREE CANDIDATES PASS `L3.1` (untouched P-A KL); `K3.45-B1B2` PASSES `L3.2` ON K4-AUTHORED `D_margin_heldout` (`+0.478` vs rebuilt, `+0.561` vs native); NEUTRAL `exlr8-k4` CHECK SHOWS CONSISTENT POINT ESTIMATE (`+0.252` vs rebuilt, not detected at `n = 24`; detected vs native `+0.271`)**

---

## 1. Untouched P-A Teacher Panel (`51,175` rows, `25` clusters — NEVER Used in Phase 2)

| Arm | ΔGiB | bpw | P-A KL (95% CI) | P-A NLL | `ΔKL_PA` vs `k325-rebuilt` (95% CI) | K4 Headroom | `ΔKL_PA` vs `k325-native` (95% CI) | `L3.1` Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `k325-native` | `0.00` | `3.2442` | `0.046711` `[0.035364, 0.059948]` | `1.072767` | — | — | — | Published baseline |
| `k325-rebuilt` | `0.00` | `3.2442` | `0.046986` `[0.036560, 0.058898]` | `1.081445` | — | `0.0%` | `+0.000275` `[−0.002806, +0.003298]` | Matched control |
| `k335-b2` | `+3.80` | `3.3488` | `0.044508` `[0.034139, 0.056499]` | `1.078540` | `−0.002479` `[−0.004236, −0.000892]` | `14.9%` | `−0.002203` `[−0.004490, +0.000264]` | **PASS** (borderline vs native) |
| **`k325-iso-hot`** | **`0.00`** | **`3.2442`** | **`0.043064` `[0.033388, 0.054085]`** | `1.076602` | **`−0.003922` `[−0.007087, −0.001175]`** | **`23.7%`** | **`−0.003647` `[−0.007168, −0.000749]`** | **PASS (`0 B` extra!)** |
| **`k345-b1b2`** | **`+7.59`** | **`3.4535`** | **`0.038228` `[0.029526, 0.048121]`** | `1.075230` | **`−0.008759` `[−0.011693, −0.006086]`** | **`52.8%`** | **`−0.008483` `[−0.012545, −0.005131]`** | **PASS (`6.1σ`)** |
| `k4-bf16core` | `+27.42` | `4.0000` | `0.030406` `[0.022946, 0.039552]` | `1.067576` | `−0.016581` `[−0.021284, −0.012103]` | `100.0%` | `−0.016305` `[−0.021391, −0.011533]` | Upper bound |

- Direct paired **`k345-b1b2 − k335-b2`**: **`−0.006280` nats (`95% CI [−0.009111, −0.003876]`, `37.9%` of K4 headroom)** — adding `B1 (L09–L14)` adds substantial untouched P-A KL recovery on top of `B2`, and `k345-b1b2` stays cleanly ahead of `k325-iso-hot` (`−0.004836`, `CI [−0.007227, −0.001826]`).
- **Zero-byte generalization (`[E3.2]`):** `k325-iso-hot` (`0 B` extra) cleared `L3.1` against **both** controls (`−0.003922` vs rebuilt, `−0.003647` vs native, both CIs excluding `0`, with zero selection leakage because hot rankings come from the offline K3 error ledger, not P-A). Directly compared against `+3.80 GiB` `k335-b2`, `k325-iso-hot` is **not separated on P-A** (`−0.001444`, `CI [−0.004488, +0.001081]` includes `0`) and ranked below `k335-b2` on P-B (`0.0423` vs `0.0397`).

---

## 2. Held-Out `D_margin_heldout` (`n = 36`, `35/36` K4-Authored) & Neutral-Author `D_margin_neutral` (`n = 24`, `23/24` `exlr8-k4`)

### Selection-Held-Out `D_margin_heldout` (`n = 36`, NEVER Used in Phases 1–2; `35/36` Authored by `K4-Core-K6` Experts)

*(Provenance note: untouched for candidate selection, but `35/36` trajectories were authored by `K4-Core-K6` experts — `27 v6-bf16core`, `4 v6-bf16core-bridge`, `4 v6-hqa`, `1 fp8c`.)*

| Arm | `B_close` Mean | `ΔB_close` vs `k325-rebuilt` (95% CI) | Items `>` Rebuilt | `ΔB_close` vs `k325-native` (95% CI) | Share of K4 Lift vs Rebuilt (95% CI) |
|---|---:|---:|---:|---:|---:|
| `k325-native` | `+3.1712` | `−0.0828` `[−0.3502, +0.1638]` | `17 / 36` | — | — |
| `k325-rebuilt` | `+3.2540` | — | — | `+0.0828` `[−0.1637, +0.3507]` | `0.0%` |
| `k325-iso-hot` (`0 B`) | `+3.3082` | `+0.0542` `[−0.1604, +0.2470]` | `22 / 36` | `+0.1369` `[−0.2233, +0.4898]` | `+9.1%` |
| `k335-b2` (`+3.80 GiB`) | `+3.4150` | **`+0.1610` `[+0.0048, +0.3264]`** | `22 / 36` | `+0.2438` `[−0.0446, +0.5585]` | `+27.0%` |
| **`k345-b1b2` (`+7.59 GiB`)** | **`+3.7320`** | **`+0.4780` `[+0.2423, +0.7665]`** | **`27 / 36`** | **`+0.5607` `[+0.2359, +0.9095]`** | **`80.2%` `[44%, 128%]`** |
| `k4-bf16core` (`+27.42 GiB`) | `+3.8499` | **`+0.5959` `[+0.3830, +0.8029]`** | **`29 / 36`** | **`+0.6787` `[+0.3980, +0.9905]`** | `100.0%` |

### Neutral-Author `D_margin_neutral` (`n = 24`: `23/24` Independent-Quantizer `exlr8-k4`; `1/24` K3.25-Authored Fallback `ledger-118`)

| Arm | `B_close` Mean | `ΔB_close` vs `k325-rebuilt` *(pre-registered)* (95% CI) | `ΔB_close` vs `k325-native` (95% CI) | Items `>` Rebuilt / Native |
|---|---:|---:|---:|---:|
| `k325-native` | `+3.5356` | `−0.0188` `[−0.3036, +0.3142]` | — | `8/24` / — |
| `k325-rebuilt` | `+3.5544` | — | `+0.0188` `[−0.3139, +0.3036]` | — / `16/24` |
| `k325-iso-hot` (`0 B`) | `+3.6018` | `+0.0474` `[−0.1292, +0.2485]` | `+0.0661` `[−0.1424, +0.2977]` | `9/24` / `12/24` |
| **`k335-b2` (`+3.80 GiB`)** | **`+3.7308`** | **`+0.1764` `[+0.0378, +0.3257]`** | `+0.1951` `[−0.1020, +0.4723]` | `15/24` / `16/24` |
| `k345-b1b2` (`+7.59 GiB`) | `+3.7412` | `+0.1868` `[−0.0256, +0.4348]` | `+0.2056` `[−0.0052, +0.4209]` | `12/24` / `15/24` |
| `k4-bf16core` (`+27.42 GiB`) | `+3.8061` | `+0.2517` `[−0.0276, +0.5603]` *(not detected at `n=24`)* | **`+0.2705` `[+0.0643, +0.4680]`** | `12/24` / `17/24` |

### Key `B_close` Diagnostics (`[E3.3]`, `[E3.4]`)
1. **Pre-registered neutral validity check (`k4-bf16core − k325-rebuilt` on the same 24 build item IDs):**
   Point estimate is consistent with Phase 1 on independent trajectories (**`+0.2517` `[−0.0276, +0.5603]`** on `exlr8-k4` vs **`+0.2773` `[+0.0197, +0.5381]`** on K4-authored Phase 1), arguing against a large self-authorship inflation, though the pre-registered matched-control contrast is **not detected at `n = 24`** (detected vs native: `+0.2705` `[+0.0643, +0.4680]`). Excluding the single K3.25-authored fallback (`ledger-118`, `exlr8-k4`-only `n = 23`) leaves the result unchanged: `k4 − rebuilt = +0.2296` `[−0.0583, +0.5475]`, `k4 − native = +0.2661` `[+0.0490, +0.4740]`.
2. **`B2` vs `B1` authorship split (`D_margin_heldout` vs `D_margin_neutral`):**
   - `B2` alone (`k335-b2 − rebuilt`) replicates across authorship: **`+0.1610` `[+0.0048, +0.3264]`** (K4-authored `n=36`) and **`+0.1764` `[+0.0378, +0.3257]`** (`exlr8-k4` `n=24`, sole candidate detected vs matched control on neutral items).
   - What `B1` adds (`k345-b1b2 − k335-b2`) is large on K4-authored `D_margin_heldout` (**`+0.3170` `[+0.1285, +0.5356]`**, bringing `k345` to `80.2%` `[44%, 128%]` of K4's lift, with direct `k4 − k345 = +0.1179` `[−0.1399, +0.3547]` not detected), but near zero on `exlr8-k4` `D_margin_neutral` (**`+0.0104` `[−0.1379, +0.1786]`**; `k4 − k345 = +0.0649` `[−0.0993, +0.2260]`). Because the two splits use different item sets, separating item-set from authorship requires scoring `probe_neutral`'s untouched 36-item held-out split (`35 exlr8-k4 + 1 k325-fp8c`), noted as a pre-registered optional post-3B check.
3. **Stage 2B `K3.25-iso-hot` watch item cleared:** The negative build-set point estimate (`−0.1720`) did not reappear on either `D_margin_heldout` (`+0.0542` vs rebuilt, `+0.1369` vs native) or `D_margin_neutral` (`+0.0474` / `+0.0661`).

---

## 3. Stage 3B Arm Selection (`4` Arms, Anchor-Gap Early-Stop Gate)

Per `protocols/PHASE3.md` (`Amendment 3A.1` & `Amendment 3B.1`, max 4 Stage 3B arms on 60-item `4K + 127K` Context-Probe × 4 passes, `max_tokens = 32,768`, CuteAFD `v6.0.0` BF16/F32 core):
1. **`k325-native`** (published deployment baseline)
2. **`k4-bf16core`** (upper anchor) → **Automated `G_anchor` prerequisite check after Arm 2**; stops cleanly if `G_anchor` 95% CI includes `0` at 4 passes.
3. **`k345-b1b2`** (`+7.59 GiB`, `3.4535 bpw` — **Primary deployment candidate**: `51.2%–52.8%` untouched P-A KL, `80.2%` `[44%, 128%]` K4-authored held-out `B_close`)
4. **`k325-iso-hot`** (`0 B` delta, `3.2442 bpw` — **Zero-byte iso-bitrate finalist**: `22.0%–23.7%` untouched P-A KL detected against both controls at zero extra bytes)
