# Phase 2 Stage 2B Protocol — Null Controls, Role Splits, Routing-Targeted Hot-K3 & Iso-Bitrate Swaps

**Status:** FROZEN before first Stage 2B GPU request (`2026-11-23`)  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Matched control baseline:** Phase 1 `k325-rebuilt-r0` (`B_close = +3.6003` nats, P-B `gen` KL = `0.044447` nats)  
**Stage 2A inputs (`[E2.1]–[E2.4]`):**
- Top sink bands: **`B2 (L15–L20)`** (`ΔKL_PB = −0.004713` nats / `36.0%` headroom, `ΔB_close = +0.1041` nats) and **`B1 (L09–L14)`** (`ΔKL_PB = −0.002583` / `19.8%`, `ΔB_close = +0.2045` / `73.8%`).
- Cheapest 3-band source pool (`−3,888 MiB` combined): **`B0 (L03–L08)`** (`ΔKL_PB = −0.000132`), **`B5 (L33–L38)`** (`+0.000680`), and **`B6 (L39–L44)`** (`+0.000739`), total downward KL cost `+0.001287` nats (`0` new negative `B_close` flips across all three).
- Offline routing & K3 reconstruction error source: `K3-v1/ds4rt-exl3-error-ledger.jsonl` (`37,152` routed projections; `route_evidence.expert_gate_squared_mass_fraction` and `quantizer_metrics.hessian_weighted_relative_error`).

---

## 1. Stage 2B Flip Matrix (`10` flips)

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

## 2. Stage 2B Questions & Pre-Data Selection Rules (`L2.3`, Amendment 2B.1)

> **Pre-data Amendment 2B.1 (`2026-11-23`, recorded before Flip 1 of Stage 2B completed):** Locks the quantitative interpretation of the two within-band shuffles, the `B_close` detection threshold, the iso-bitrate selection criterion, and the `K3.35` vs `K3.45` advancement rule before any Stage 2B result is observed.

1. **Shuffle interpretation (`null-shuf-b2`, `null-shuf-b5` — two shuffles, not a pure null):**
   - Permuting which projections are K4 inside `B2` or `B5` removes `K3.25-v1`'s within-layer Hessian selection at matched bytes (`0 B` delta).
   - **P-B `gen` KL:** Read each shuffle's `ΔKL_PB` (with its cluster-bootstrap 95% CI) as the empirical **pick value** of `K3.25-v1`'s within-layer Hessian selection vs random selection in `B2` and `B5`.
   - **`B_close` jitter scale:** Compute (a) the pooled per-item `ΔB_close` SD across the two shuffles and (b) `S_max = max(|ΔB_close(null-shuf-b2)|, |ΔB_close(null-shuf-b5)|)`. Always qualify this envelope as derived from **two shuffles** (a rough jitter scale, not a full null distribution).
2. **`B_close` detection rule:**
   - A Stage 2B arm's `B_close` change counts as a **detected signal** only if **BOTH** (a) its paired bootstrap 95% CI vs `k325-rebuilt` excludes `0` **AND** (b) `|ΔB_close| > S_max`. Shifts failing either clause read as within the two-shuffle jitter scale (`"not detected"`).
3. **Role dominance inside `B2` (`gate` / `up` / `down`):**
   - Identifies whether `B2`'s `36.0%` P-B KL spike concentrates in a single projection role (`−ΔKL_PB / GiB`, paired 95% CIs).
4. **Head-to-head routing test at exact `+3,888 MiB` (`3.3488 bpw`):**
   - Compares `b1b2-hot-route-3888` and `b1b2-hot-routed-err-3888` directly against Stage 2A's full-band `b2-up-k4` (`ΔKL_PB = −0.004713`) via direct paired `53`-cluster P-B KL contrasts (`pair-hot-*-vs-b2-up.json`) at identical byte cost (`+3.7969 GiB`).
5. **Pre-registered Phase 3 candidate selection rules (`L2.3`):**
   - **Primary deployment candidate (`K3.35` vs `K3.45`):**
     - Let `best_3888` be the `+3.80 GiB` arm (`b2-up-k4`, `b1b2-hot-route-3888`, or `b1b2-hot-routed-err-3888`) with the largest P-B `gen` KL reduction (`−ΔKL_PB`, preferring a hot-K3 arm over `b2-up-k4` only if its direct paired P-B KL 95% CI vs `b2-up-k4` excludes `0`, else keeping simpler full-band `b2-up-k4`).
     - Advance **`b1b2-up-k4` (`K3.45`, `+7.59 GiB`)** as primary deployment candidate **AND** also advance `best_3888` (`K3.35`, `+3.80 GiB`) as an efficiency candidate if `best_3888` achieves **`≥ 1.25×`** the P-B KL recovery per GiB of `b1b2-up-k4` (equivalently `≥ 62.5%` of `b1b2-up-k4`'s total KL reduction at half the byte cost). If `b1b2-up-k4` recovers `< 1.25×` `best_3888`'s total KL reduction (sub-additive), advance `best_3888` alone as primary.
   - **Secondary iso-bitrate candidate (`K3.25-iso`, `0 B` delta):**
     - Advance whichever of `iso-b0b5b6-to-b2` and `iso-b0b5b6-to-hot-b1b2` achieves the largest P-B `gen` KL reduction (breaking ties in favor of simpler `iso-b0b5b6-to-b2` if direct paired `pair-iso-hot-vs-iso-b2.json` 95% CI includes `0`), provided its paired P-B `gen` KL 95% CI vs `k325-rebuilt` is strictly `< 0` AND it shows **no detected negative `B_close` regression** under Rule 2 above (not requiring a coin-flip positive `B_close` point estimate).
   - **Neutral-author `B_close` check:** Run on the selected Phase 3 candidates + `k325-rebuilt` + `k4-bf16core` at Phase 3 entry before final held-out `64K–127K` task confirmation.
