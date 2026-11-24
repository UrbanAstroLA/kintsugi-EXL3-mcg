# Phase 2 Stage 2A Summary — 7-Band Damage Map & Gate P2 Verdict

**Date:** 2026-11-23  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Protocol:** [`protocols/PHASE2.md`](../../protocols/PHASE2.md) (frozen with pre-data `Amendment 2A.1` / `L2.1-v2` before Flip 1 completed)  
**Matched control:** Phase 1 `k325-rebuilt-r0` (`B_close = +3.6003` nats, P-B `gen` KL = `0.044447` nats)  
**Gate P2 (`L2.1-v2`) Verdict:** **DECISIVE PASS (`4.31×` upward P-B KL spread; direct paired `B2 − B5 = −0.003620` nats, `4.5σ`; `ρ_up_down_KL = +0.6786`)**

---

## 1. Pre-Registered Gate P2 (`L2.1-v2`) Verdict

**Scope:** The P-B fidelity panel covers prompts up to `14,403` tokens (`15,104` generated rows across `53` clusters). Stage 2A therefore ranks depth bands by **short-context distribution fidelity (`≤ 14.4K` tokens)**; transfer to `64K–127K` tasks is tested in Phase 3.

| Criterion (`L2.1-v2`) | Threshold | Measured Result | Verdict |
|---|---|---|---|
| **(1a) Direct paired `top − bottom` upward P-B `gen` KL** (`53` clusters) | 95% CI excludes `0` | **`B2 − B5 = −0.003620` nats (`95% CI [−0.005241, −0.002078]`, `4.5σ`)** | **PASS** (`[E2.1]`) |
| **(1a) Upward `max / min` P-B `gen` KL sensitivity** | `≥ 1.5×`, `min > 0` (CI excludes `0`) | **`4.309×`** (`B2 = −0.004713` vs `B5 = −0.001093`, `B5` CI `[−0.001766, −0.000393]` excludes `0`) | **PASS** (`[E2.1]`) |
| **(1b) Cross-direction structural consistency (`ρ_up_down_KL`)** | `≥ 0.60` | **`+0.6786`** (Spearman across `B0..B6`) | **PASS** (`[E2.2]`) |
| **Confirmatory `B_close` (`n = 24`)** | Reported (pending Stage 2B shuffle jitter scale) | `ρ(up B_close, −ΔKL_PB) = +0.5357`; `ρ(up, down B_close) = +0.4643`; all upward positive CIs include `0` and 7-band sum is non-additive (`0.42×`) — treated as confirmatory pending Stage 2B's two-shuffle jitter scale | **Confirmatory (pending null)** (`[E2.3]`) |

*(All 14 single-band flips passed `stitch_exl3.py verify` with `0` header/SHA mismatches and `validate_pb.py` same-dump verification with `32,768/32,768` records, `0` mismatches, and `0.0` max KL/NLL discrepancy. Archived original `L2.1` also passed.)*

---

## 2. Full 7-Band Damage Map (`B0..B6`, Matched Against `k325-rebuilt-r0`)

### Upward Flips (`band → 100% K4`, `+3,888 MiB` / `+3.7969 GiB` each, `3.3488 bpw`)

| Band | Layers | `ΔKL_PB` (`gen`, 95% CI) | KL Headroom Share | `ΔB_close` (nat, 95% CI) | `B_close` Headroom Share | Items `>` Base |
|---|---|---:|---:|---:|---:|---:|
| **`B2`** | **`L15–L20`** | **`−0.004713` `[−0.006613, −0.002964]`** | **`36.0%`** (#1) | **`+0.1041` `[−0.0480, +0.2668]`** | **`37.5%`** (#2) | `14 / 24` |
| **`B1`** | **`L09–L14`** | **`−0.002583` `[−0.004071, −0.001123]`** | **`19.8%`** (#2) | **`+0.2045` `[−0.0060, +0.4231]`** | **`73.8%`** (#1) | `17 / 24` |
| `B3` | `L21–L26` | `−0.002217` `[−0.003279, −0.001177]` | `17.0%` (#3) | `+0.0313` `[−0.1202, +0.2042]` | `11.3%` (#4) | `12 / 24` |
| `B0` | `L03–L08` | `−0.001688` `[−0.003104, −0.000239]` | `12.9%` (#4) | `−0.2520` `[−0.4971, −0.0410]` | `−90.9%` (#7) | `8 / 24` |
| `B4` | `L27–L32` | `−0.001544` `[−0.002438, −0.000625]` | `11.8%` (#5) | `−0.0455` `[−0.1514, +0.0587]` | `−16.4%` (#6) | `10 / 24` |
| `B6` | `L39–L44` | `−0.001150` `[−0.001528, −0.000765]` | `8.8%` (#6) | `+0.0532` `[−0.0262, +0.1321]` | `19.2%` (#3) | `14 / 24` |
| `B5` | `L33–L38` | `−0.001093` `[−0.001766, −0.000393]` | `8.4%` (#7) | `+0.0222` `[−0.0373, +0.0804]` | `8.0%` (#5) | `16 / 24` |
| *Sum of 7 bands* | *`L03–L44`* | *`−0.014988` (`1.15×` whole-model `−0.013074`)* | *`114.6%`* | *`+0.1178` (`0.42×` whole-model `+0.2773`)* | *`42.5%`* | — |

### Downward Flips (`band → 0% K4`, `−1,296 MiB` / `−1.2656 GiB` each, `3.2093 bpw`)

| Band | Layers | `ΔKL_PB` (`gen`, 95% CI) | KL Damage / GiB | `ΔB_close` (nat, 95% CI) | New Neg / Frag |
|---|---|---:|---:|---:|---:|
| **`B2`** | **`L15–L20`** | **`+0.002546` `[+0.001333, +0.003817]`** | **`+0.002011`** (#1 worst) | `−0.1344` `[−0.2782, +0.0093]` | **`1` (`join-106`)** / `1` |
| **`B1`** | **`L09–L14`** | **`+0.001772` `[+0.000433, +0.003279]`** | **`+0.001400`** (#2 worst) | **`−0.1734` `[−0.3504, +0.0003]`** | `0` / `1` |
| `B3` | `L21–L26` | `+0.001176` `[+0.000338, +0.002095]` | `+0.000929` (#3) | `−0.1436` `[−0.2609, −0.0361]` | `0` / `1` |
| `B6` | `L39–L44` | `+0.000739` `[+0.000357, +0.001163]` | `+0.000584` (#4) | **`+0.0067` `[−0.0627, +0.0816]`** | `0` / `0` |
| `B5` | `L33–L38` | `+0.000680` `[−0.000193, +0.001624]` | `+0.000537` (#5) | `−0.0324` `[−0.1335, +0.0760]` | `0` / `1` |
| `B4` | `L27–L32` | `+0.000090` `[−0.000645, +0.000952]` | `+0.000071` (#6) | `−0.1334` `[−0.2574, −0.0116]` | `0` / `0` |
| `B0` | `L03–L08` | **`−0.000132` `[−0.001672, +0.001375]`** | **`−0.000104`** (#7 cheapest) | `−0.1077` `[−0.3384, +0.1125]` | `0` / `1` |
| *Sum of 7 bands* | *`L03–L44`* | *`+0.006871` (`0.94×` whole-model `+0.007335`)* | — | *`−0.7182`* | — |

---

## 3. Comparison Against Offline Local K3 Reconstruction Error (`[E2.4]`)

Comparing the empirical Stage 2A P-B sensitivity map (`≤ 14.4K` tokens) against the offline K3 reconstruction error ledger (`K3-v1/ds4rt-exl3-error-ledger.jsonl`, `3,888` remaining K3 projections per band) shows that **per-band local K3 reconstruction error did not predict per-band end-to-end P-B KL sensitivity** across the seven bands (`Spearman ρ = −0.464` for `mean_h_rel`, `−0.607` for `sum(g² × h_rel)`, `n = 7`, neither statistically significant across all 7 bands, though the two endpoints `B2` and `B5` reverse):

| Band | Layers | Offline `mean_h_rel` (K3-v1) | Offline `sum(g² × h_rel)` | Offline `mean_h_num` | **Empirical `−ΔKL_PB` (Upward)** | **Empirical Rank vs Offline Rank** |
|---|---|---:|---:|---:|---:|---|
| **`B2`** | **`L15–L20`** | `0.007109` (#6) | `0.060058` (#7) | `6,292` (#4) | **`0.004713` (#1)** | #1 empirical vs #6–7 offline |
| **`B1`** | **`L09–L14`** | `0.007241` (#5) | `0.067187` (#4) | `5,028` (#7) | **`0.002583` (#2)** | #2 empirical vs #4–7 offline |
| `B3` | `L21–L26` | `0.007417` (#4) | `0.065461` (#5) | `6,234` (#5) | `0.002217` (#3) | #3 empirical vs #4–5 offline |
| `B0` | `L03–L08` | `0.007569` (#3) | `0.071821` (#3) | `5,232` (#6) | `0.001688` (#4) | #4 empirical vs #3 offline |
| `B4` | `L27–L32` | `0.008691` (#2) | `0.073540` (#2) | `11,745` (#3) | `0.001544` (#5) | #5 empirical vs #2 offline |
| `B6` | `L39–L44` | `0.006901` (#7) | `0.061799` (#6) | `21,694` (#1) | `0.001150` (#6) | #6 empirical vs #1 `h_num` |
| **`B5`** | **`L33–L38`** | **`0.008995` (#1)** | **`0.075106` (#1)** | `15,799` (#2) | **`0.001093` (#7)** | #7 empirical vs #1 offline |

1. **Early-mid MoE P-B KL concentration (`B1–B2`, `L09–L20`):** On the P-B panel (`≤ 14.4K` tokens), `12 / 42` MoE layers (`28.6%`) account for **`55.8%`** (`19.8% + 36.0%`) of the total K3.25→K4 P-B KL headroom, with nearly additive single-band sums (`1.15×` upward, `0.94×` downward).
2. **Late-layer P-B insensitivity (`B5–B6`, `L33–L44`):** Despite high local reconstruction errors in the offline ledger, `B5` and `B6` contribute `8.4%–8.8%` of P-B KL headroom (`4.31×` less than `B2`) and small downward KL cost (`+0.000680`, `+0.000739` nats).
3. **`B_close` status (confirmatory, pending Stage 2B null scale):** Every upward `ΔB_close` 95% CI includes `0` except `B0` (negative, `−0.2520`, CI `[−0.4971, −0.0410]`), and two downward CIs (`B3`, `B4`) also exclude `0` out of 14 band tests; moreover, single-band `B_close` sums to only `0.42×` of whole-model lift (`+0.1178` vs `+0.2773`). Per-band `B_close` shifts (including `B0` upward and the single `join-106` crossing from `+0.15` nat on `B2` downward) are therefore not interpreted as established findings until calibrated against Stage 2B's two-shuffle jitter scale. Whether reallocating bits from `B0 + B5 + B6` into `B1–B2` outperforms uniform `K3.25` is directly tested in Stage 2B and Phase 3.

---

## 4. Next Step — Stage 2B (`protocols/PHASE2B.md`, `10` flips, ETA `~2.7` wall-h)

Stage 2A advances directly to **Stage 2B** (frozen before first Stage 2B GPU request):
- **2 within-band iso-bitrate shuffle null controls** (`null-shuf-b2`, `null-shuf-b5`, per `Amendment 2A.1`),
- **3 role splits on `#1` sink `B2`** (`gate_proj`, `up_proj`, `down_proj`),
- **2 routing-targeted hot-K3 upgrades across `B1+B2`** at exact 1-band byte budget (`+3,888 MiB`, `3.3488 bpw`) ranked by `expert_gate_squared_mass_fraction` vs `mass × hessian_rel_err`,
- **2 iso-bitrate (`0 B` delta, `3.2500 bpw`) swaps** (`B0+B5+B6 → B2` and `B0+B5+B6 → hot B1+B2`), and
- **1 combined `B1+B2 → 100% K4` check** (`+7.59 GiB`, `3.4477 bpw`, additivity check for primary `K3.45-Kintsugi`).
