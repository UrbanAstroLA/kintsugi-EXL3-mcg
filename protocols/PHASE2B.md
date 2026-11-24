# Phase 2 Stage 2B Protocol — Null Controls, Role Splits, Routing-Targeted Hot-K3 & Iso-Bitrate Swaps

**Status:** FROZEN before first Stage 2B GPU request (`2026-11-23`)  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Matched control baseline:** Phase 1 `k325-rebuilt-r0` (`B_close = +3.6003` nats, P-B `gen` KL = `0.044447` nats)  
**Stage 2A inputs (`[E2.1]–[E2.4]`):**
- Top sink bands: **`B2 (L15–L20)`** (`ΔKL_PB = −0.004713` nats / `36.0%` headroom, `ΔB_close = +0.1041` nats) and **`B1 (L09–L14)`** (`ΔKL_PB = −0.002583` / `19.8%`, `ΔB_close = +0.2045` / `73.8%`).
- Cheapest 3-band source pool (`−3,888 MiB` combined): **`B0 (L03–L08)`** (`ΔKL_PB = −0.000132`), **`B5 (L33–L38)`** (`+0.000680`), and **`B6 (L39–L44)`** (`+0.000739`), total downward KL cost `+0.001287` nats (`0` new negative `B_close` flips across all three).
- Offline routing & K3 reconstruction error source: `K3-v1/ds4rt-exl3-error-ledger.jsonl` (`37,152` routed projections; `route_evidence.expert_gate_squared_mass_fraction` and `quantizer_metrics.hessian_weighted_relative_error`).

---

## 1. Stage 2B Flip Matrix (`10` flips, ETA `~2.7` wall-h)

All 10 allocation maps are generated deterministically by [`tools/make_stage2b_maps.py`](../tools/make_stage2b_maps.py) before launch and stitched incrementally via `tools/stitch_exl3.py --reuse-from k325-rebuilt`. Every flip runs the identical frozen `D_margin_build` (`n = 24`) inline probe + full `32,768`-row P-B fidelity evaluation (`validate_pb.py` `0` mismatches required) + rolling cleanup of temporary shards and dumps.

| # | Arm | Category | Allocation Rule (from `K3.25-rebuilt`, `9,072` K4 base) | K4 Count | ΔMiB (ΔGiB) | bpw |
|---|---|---|---|---:|---:|---:|
| 1 | `null-shuf-b2` | Null control (`2A.1`) | Inside `B2 (L15–L20)`, randomly permute which projections are K4 preserving exact per-`(layer, role)` K4 counts (`seed=20261123`) | `9,072` | `0` (`0.00`) | `3.2442` |
| 2 | `null-shuf-b5` | Null control (`2A.1`) | Inside `B5 (L33–L38)`, randomly permute which projections are K4 preserving exact per-`(layer, role)` K4 counts (`seed=20261124`) | `9,072` | `0` (`0.00`) | `3.2442` |
| 3 | `b2-role-gate` | Role split (`B2`) | Upgrade all remaining K3 `gate_proj` in `B2 (L15–L20)` to K4 | `10,557` | `+1,485` (`+1.45`) | `3.2842` |
| 4 | `b2-role-up` | Role split (`B2`) | Upgrade all remaining K3 `up_proj` in `B2 (L15–L20)` to K4 | `10,395` | `+1,323` (`+1.29`) | `3.2798` |
| 5 | `b2-role-down` | Role split (`B2`) | Upgrade all remaining K3 `down_proj` in `B2 (L15–L20)` to K4 | `10,152` | `+1,080` (`+1.05`) | `3.2733` |
| 6 | `b1b2-hot-route-3888` | Routing-targeted (`+3.80 GiB`) | Upgrade top `3,888` remaining K3 projections across `B1+B2 (L09–L20)` ranked by `expert_gate_squared_mass_fraction` | `12,960` | `+3,888` (`+3.80`) | `3.3488` |
| 7 | `b1b2-hot-routed-err-3888` | Routing × error (`+3.80 GiB`) | Upgrade top `3,888` remaining K3 projections across `B1+B2 (L09–L20)` ranked by `expert_gate_squared_mass_fraction × hessian_weighted_relative_error` | `12,960` | `+3,888` (`+3.80`) | `3.3488` |
| 8 | `iso-b0b5b6-to-b2` | Iso-bitrate band swap (`0 B`) | Drop `B0 + B5 + B6` to `0%` K4 (`−3,888`); upgrade `B2` to `100%` K4 (`+3,888`) | `9,072` | `0` (`0.00`) | `3.2442` |
| 9 | `iso-b0b5b6-to-hot-b1b2` | Iso-bitrate hot swap (`0 B`) | Drop `B0 + B5 + B6` to `0%` K4 (`−3,888`); upgrade top `3,888` remaining K3 projections in `B1+B2` (`mass × h_rel`) | `9,072` | `0` (`0.00`) | `3.2442` |
| 10 | `b1b2-up-k4` | Primary deployment check (`+7.59 GiB`) | Upgrade all of `B1 + B2 (L09–L20)` to `100%` K4 (tests additivity of `B1` + `B2` for `K3.45-Kintsugi`) | `16,848` | `+7,776` (`+7.59`) | `3.4535` |

---

## 2. Stage 2B Questions & Phase 3 Selection Rules (`L2.3`)

1. **Null jitter & within-layer Hessian value (`null-shuf-b2`, `null-shuf-b5`):**
   - Directly measures single-band zero-bitrate `B_close` jitter (`|ΔB_close|` envelope) and shows how much `K3.25-v1`'s within-layer Hessian picks outperform random K4 picks inside a high-sensitivity band (`B2`) vs a low-sensitivity band (`B5`).
2. **Role dominance inside `B2` (`gate` / `up` / `down`):**
   - Identifies whether `B2`'s `36.0%` KL spike concentrates in a single projection role (`ΔKL_PB / GiB` and `ΔB_close / GiB`).
3. **Head-to-head routing test at exact `+3,888 MiB` (`3.3488 bpw`):**
   - Compares `b1b2-hot-route-3888` and `b1b2-hot-routed-err-3888` directly against Stage 2A's full-band `b2-up-k4` (`ΔKL_PB = −0.004713`, `ΔB_close = +0.1041`) and `b1-up-k4` (`ΔKL_PB = −0.002583`, `ΔB_close = +0.2045`) at **identical byte cost (`+3.7969 GiB`)**.
4. **Iso-bitrate feasibility (`0 B` delta, `3.2500 bpw`):**
   - Tests whether `iso-b0b5b6-to-b2` or `iso-b0b5b6-to-hot-b1b2` achieves statistically significant P-B KL reduction (95% CI excluding `0`) and positive `B_close` movement at **zero extra bytes**.
5. **Phase 3 candidate selection (`L2.3`):**
   - **Primary deployment candidate (`K3.35–K3.45`):** Advance `b1b2-up-k4` (`K3.45`, `+7.59 GiB`) AND/OR the best `+3.80 GiB` (`K3.35`) arm (`b2-up-k4` vs `b1b2-hot-route-3888` vs `b1b2-hot-routed-err-3888`) if the `+3.80 GiB` winner achieves substantially higher recovery per GiB.
   - **Secondary iso-bitrate candidate (`K3.25-iso`):** Advance whichever of `iso-b0b5b6-to-b2` and `iso-b0b5b6-to-hot-b1b2` achieves stronger verified P-B KL reduction and better `B_close` without regression (`ΔKL_PB < 0` with CI excluding `0`).
   - **Neutral-author `B_close` check:** Run on the selected Phase 3 candidates + `k325-rebuilt` + `k4-bf16core` at Phase 3 entry before final held-out task confirmation.
