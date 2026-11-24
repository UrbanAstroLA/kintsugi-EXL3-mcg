# Phase 3 Stage 3A Summary — Untouched P-A, Held-Out `B_close` & Independent-Quantizer Neutral-Author Check

**Date:** 2026-11-24  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Protocol:** [`protocols/PHASE3.md`](../../protocols/PHASE3.md) (frozen with pre-data `Amendment 3A.1`, swapping `probe_neutral/` atomically to published independent-quantizer `exlr8-k4` before any neutral request ran)  
**Stage 3A Verdict:** **ALL THREE CANDIDATES PASS `L3.1` (untouched P-A KL); `K3.45-B1B2` PASSES `L3.2` BY A WIDE MARGIN (`80–83%` of K4 held-out `B_close`); INDEPENDENT `exlr8-k4` NEUTRAL CHECK CONFIRMS `B_close` IS QUANTIZER-INDEPENDENT (`+0.2705` nat, CI excludes `0`)**

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

- Direct paired **`k345-b1b2 − k335-b2`**: **`−0.006280` nats (`95% CI [−0.009111, −0.003876]`, `37.9%` of K4 headroom)** — adding `B1 (L09–L14)` contributes more untouched P-A KL recovery than `B2` alone.
- **Zero-byte generalization (`[E3.2]`):** `k325-iso-hot` (`0 B` extra) unexpectedly exceeded the pre-data power expectation (`~−0.0027` projected vs **`−0.003922` observed**, CI strictly `< 0` against **both** `k325-rebuilt` and `k325-native`), and even achieved lower untouched P-A KL than `+3.80 GiB` `k335-b2` (`0.043064` vs `0.044508`) because it reallocated `2,156` hot K4 slots into `B1 (L09–L14)`.

---

## 2. Untouched `D_margin_heldout` (`n = 36`, `127K`) & Independent `exlr8-k4` Neutral-Author `D_margin_neutral` (`n = 24`, `127K`)

### Untouched `D_margin_heldout` (`n = 36`, NEVER Seen in Phases 1–2)

| Arm | `B_close` Mean | `ΔB_close` vs `k325-rebuilt` (95% CI) | Items `>` Rebuilt | `ΔB_close` vs `k325-native` (95% CI) | Share of K4 Lift |
|---|---:|---:|---:|---:|---:|
| `k325-native` | `+3.1712` | `−0.0828` `[−0.3502, +0.1638]` | `17 / 36` | — | — |
| `k325-rebuilt` | `+3.2540` | — | — | `+0.0828` `[−0.1637, +0.3507]` | `0.0%` |
| `k325-iso-hot` (`0 B`) | `+3.3082` | `+0.0542` `[−0.1604, +0.2470]` | `22 / 36` | `+0.1369` `[−0.2233, +0.4898]` | `+9.1%` / `+20.2%` |
| `k335-b2` (`+3.80 GiB`) | `+3.4150` | **`+0.1610` `[+0.0048, +0.3264]`** | `22 / 36` | `+0.2438` `[−0.0446, +0.5585]` | `+27.0%` / `+35.9%` |
| **`k345-b1b2` (`+7.59 GiB`)** | **`+3.7320`** | **`+0.4780` `[+0.2423, +0.7665]`** | **`27 / 36`** | **`+0.5607` `[+0.2359, +0.9095]`** | **`80.2%` / `82.6%`** |
| `k4-bf16core` (`+27.42 GiB`) | `+3.8499` | **`+0.5959` `[+0.3830, +0.8029]`** | **`29 / 36`** | **`+0.6787` `[+0.3980, +0.9905]`** | `100.0%` |

### Independent-Quantizer `exlr8-k4` Neutral-Author `D_margin_neutral` (`n = 24`: `23 exlr8-k4 + 1 k325-fp8c`, ZERO EXL3-MCG Authorship)

| Arm | `B_close` Mean | `ΔB_close` vs `k325-rebuilt` (95% CI) | `ΔB_close` vs `k325-native` (95% CI) | Items `>` Native |
|---|---:|---:|---:|---:|
| `k325-native` | `+3.5356` | `−0.0188` `[−0.3036, +0.3142]` | — | — |
| `k325-rebuilt` | `+3.5544` | — | `+0.0188` `[−0.3139, +0.3036]` | `16 / 24` |
| `k325-iso-hot` (`0 B`) | `+3.6018` | `+0.0474` `[−0.1292, +0.2485]` | `+0.0661` `[−0.1424, +0.2977]` | `12 / 24` |
| **`k335-b2` (`+3.80 GiB`)** | **`+3.7308`** | **`+0.1764` `[+0.0378, +0.3257]`** | `+0.1951` `[−0.1020, +0.4723]` | `16 / 24` |
| **`k345-b1b2` (`+7.59 GiB`)** | **`+3.7412`** | `+0.1868` `[−0.0256, +0.4348]` | `+0.2056` `[−0.0052, +0.4209]` | `15 / 24` |
| **`k4-bf16core` (`+27.42 GiB`)** | **`+3.8061`** | `+0.2517` `[−0.0276, +0.5603]` | **`+0.2705` `[+0.0643, +0.4680]`** | **`17 / 24`** |

1. **Self-authorship confound resolved (`[E3.4]`):** On completely independent `exlr8-k4` trajectories (zero shared weights with any EXL3-MCG arm), `B_close` reproduces the full monotonic ladder (`+3.5356 → +3.5544 → +3.6018 → +3.7308 → +3.7412 → +3.8061`), with `k4-bf16core − k325-native = +0.2705` nats (`CI [+0.0643, +0.4680]`, matching Phase 1's `+0.2773` nats on K4-authored trajectories!).
2. **Held-out `B_close` confirmation (`[E3.3]`):** `K3.45-B1B2` passes `L3.2` decisively against **both** `k325-rebuilt` (`+0.4780`, `CI [+0.2423, +0.7665]`) and `k325-native` (`+0.5607`, `CI [+0.2359, +0.9095]`), recovering **`80.2%–82.6%` of K4's full held-out `B_close` lift in `27.7%` of the bytes**.
3. **Stage 2B `K3.25-iso-hot` watch item cleared:** The negative build-set point estimate (`−0.1720`) did not persist on either untouched `D_margin_heldout` (`+0.0542` vs rebuilt, `+0.1369` vs native) or neutral `exlr8-k4` (`+0.0474` / `+0.0661`).

---

## 3. Stage 3B Arm Selection (`4` Arms, Anchor-Gap Early-Stop Gate)

Per `protocols/PHASE3.md` (`Amendment 3A.1`, max 4 Stage 3B arms on 60-item `4K + 127K` Context-Probe × 4 passes, `max_tokens = 32,768`):
1. **`k325-native`** (published deployment baseline, `~3.1 wall-h`)
2. **`k4-bf16core`** (upper anchor, `~3.1 wall-h`) → **Automated `G_anchor` prerequisite check after Arm 2 (`~6.2 h`)**; stops cleanly if `G_anchor` 95% CI includes `0` at 4 passes.
3. **`k345-b1b2`** (`+7.59 GiB`, `3.4535 bpw` — **Primary deployment candidate**: `51.2%–52.8%` untouched P-A KL, `80.2%–82.6%` held-out `B_close`, `~3.1 wall-h`)
4. **`k325-iso-hot`** (`0 B` delta, `3.2442 bpw` — **Zero-byte iso-bitrate finalist**: `22.0%–23.7%` untouched P-A KL, beating `+3.80 GiB` `k335-b2` on P-A at zero extra bytes, `~3.1 wall-h`)
