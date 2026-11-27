# Final Synthesis — Kintsugi Mixed-Precision EXL3-MCG on GLM-5.3-Flash (`K3.25-v1`)

**Status:** Study closed (`2026-11-26`) | **Engine measured:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, `max_tokens = 32,768` on graded tasks) | **Phase 4 (`vLLM v0.9.2`) status:** Held closed (no sparse band repair cleared `L3.3`)

---

## 1. Executive Conclusion

We tested whether the [Kintsugi method](https://github.com/professorpalmer/kintsugi-method) (Palmer, 2026) — empirical, task-gated mixed-precision bit allocation — can repair **`wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1`** (`3.2442 bpw`, unquantized BF16/F32 dense core + `37,152` MCG EXL3 routed expert projections) toward full `K4` (`4.0 bpw`) at low byte cost via targeted band upgrades or iso-bitrate reallocation.

**Central finding:** **No sparse band patch exists on `K3.25-v1` that recovers full `K4`'s `128K` task stability.**

1. **Short-context KL and teacher-forced margins localized sharply — and generalized on held-out continuous metrics — but failed to transfer to live `128K` tasks.**
   - On `≤ 14.4K` P-B `gen` KL, sensitivity spanned **`4.31×`** (`4.5σ`, `ρ_up_down = +0.68`), concentrating `55.8%` of KL headroom in **`B1–B2 (L09–L20)`** while offline K3 Hessian error failed to rank depth (`ρ = −0.464`, `[E2.1]–[E2.4]`).
   - Upgrading `B1–B2` to `100%` K4 (`K3.45-B1B2`, `+7.59 GiB`, `27.7%` of K4's extra bytes) captured **`50.2%`** of P-B KL (`[E2.8]`), **`52.8%`** of untouched P-A KL (`[E3.1]`), and **`80.2%`** (`[44%, 128%]`) of K4-authored held-out `B_close` (`</think>` margin, `[E3.3]`). Even at **`0 B`** extra cost, `K3.25-iso-hot` recovered **`23.7%`** of untouched P-A KL (`[E3.2]`).
   - Yet on live `128K` graded tasks (`480` paired replies/arm), **`K3.45-B1B2` delivered zero detected recovery** over its matched stitched base `k325-rebuilt` (`−0.10 pp [−3.10, +2.86]`, unfinished `+0.21 pp [−2.71, +3.12]`; pooled 12-pass rate `45/720 = 6.25%` vs `30/480 = 6.25%` on `k325-rebuilt`, `[E3.6]`, `[E3.9]`, `[E3.11]`).
2. **Full `K4`'s `128K` task advantage is real, 100% concentrated in runaway thinking loops, and distributed across depth.**
   - Because `K3.25-v1` retains all `1,618` dense core tensors in unquantized BF16/F32, short-context benchmarks (`tool-eval-bench` `161.0` vs `160.3 / 176`, GPQA `86.9%` vs `85.7%`, `4K` Context-Probe `98.12%` vs `98.70%`) already have zero `K3.25 → K4` deficit.
   - At `128K`, full `k4-bf16core` beat both `k325-native` (`+3.70 pp [+0.55, +7.06]`, `Z_acc = +2.33`, `Z_unf = +2.81`; unfinished `35/480 → 15/480`, `[E3.5]`) and matched stitched `k325-rebuilt` (`+3.85 pp [+0.70, +6.93]`, `30/480 → 15/480`, `[E3.11]`). On `join + ledger` (`97.3%` of all unfinished replies), `k4-bf16core − k325-rebuilt` was `+4.84 pp [+0.39, +9.22]` and unfinished `−4.69 pp [−9.06, −0.31]` (`29/320 → 14/320`).
   - Finished-only accuracy of single-run `k325-native` (`94.02%`) already matched single-run `k4-bf16core` (`93.79%`); **`100%`** of the anchor gap lived in runaway thinking loops (`content_chars = 0`, predominantly `record_list` / `line_repeat`, `[E3.7]`).
   - Testing the **`5/7` complement graft `k377-not-b1b2`** (`B0 + B3–B6` at `100%` K4, `30/42` MoE layers, `+18.98 GiB`, `69.2%` of K4's extra bytes) yielded **`22/320` unfinished** (`11+11` halves, `p = 1.00`) — only **`~47%–57%`** of K4's gap (`86.33%` vs `83.55% → 88.40%`, `[E3.13]`). Diagnostically, `k377-not-b1b2` matched full K4 on `join` (`9/160` vs `8/160` unf; loop entry `13/160` identical to K4's `13/160`), while leaving `ledger` unfinished near baseline (`13/160` vs `6/160` K4, `[E3.14]`). Neither `B1–B2` nor its `5/7` complement stabilized `ledger` alone.
3. **Cross-library intra-layer stitching incurs a `−1.44 pp` finished-sequence penalty invisible to short-context KL.**
   - Matching `K3.25-v1`'s exact bit-map by stitching `K3-v1` + `K4-Core-K6` (`k325-rebuilt`, `42/42` mixed-within-layer) produced `0.0006`-nat P-B KL parity (`[E1.3]`) and identical `4K` accuracy (`98.12%`), yet suffered a statistically detected **`−1.44 pp` (`95% CI [−2.76, −0.14]`)** both-finished `128K` accuracy drop (`92.81%` vs `94.02%` single-run `k325-native`, `[E3.12]`).
   - `k345-b1b2` (`30/42` mixed layers) sat at the same level (`92.90%` over 8p; `92.98%` over 12p), whereas `k377-not-b1b2` (only `12/42` mixed layers) returned to **`92.70%`** on `join + ledger` (matching single-run `k325-native` `92.70%` and `k4-bf16core` `92.44%`, `[E3.14]`). Plausible untested mechanism: sequential calibration fits each layer and projection against activations from already-quantized upstream layers in the same run (`K3-v1` on BF16 core vs `K4-Core-K6` on K6 core), so mixing them inside a layer breaks residual error cancellation on long chains.

---

## 2. Full Stage-by-Stage Gate Ledger (`L0.1–L3F.1`)

| Phase / Stage | Governing Line | Outcome | Core Receipt |
|---|---|---|---|
| **Phase 0** | `L0.1–L0.3` | **PASS** | `0` header/SHA mismatches (`150,226` tensors); `4/4` CuteAFD `v6.0.0` planner `READY` (`[E0.5]–[E0.7]`) |
| **Phase 1** | `L1.1–L1.3` | **PASS** | Exact `0.0` restart spread; `B_close` `+0.277` nat K4 vs K3.25 (`+0.731` vs K3); `\|ΔKL_PB\| = 0.0006` (`[E1.1]–[E1.4]`) |
| **Phase 2A** | `L2.1-v2` | **PASS** | `4.31×` P-B KL spread (`4.5σ`), `ρ_up_down = +0.68`; `B1–B2` carry `55.8%` of headroom; offline Hessian `ρ = −0.46` (`[E2.1]–[E2.4]`) |
| **Phase 2B** | `L2.3` | **PASS** | `K3.45-B1B2` `50.2%` P-B KL; `K3.35-B2` `36.0%`; `K3.25-iso-hot` `16.3%` at `0 B`; shuffles `~0` (`[E2.5]–[E2.9]`) |
| **Phase 3A** | `L3.1–L3.2` | **PASS** | `K3.45-B1B2` `52.8%` untouched P-A KL & `80.2%` K4-authored held-out `B_close`; `K3.25-iso-hot` `23.7%` P-A KL at `0 B` (`[E3.1]–[E3.4]`) |
| **Phase 3B** | `L3.3` | **FAIL (`K3.45-B1B2`)** | Anchor gap detected (`Z_acc = 2.33`, `Z_unf = 2.81`, `+3.70 pp`, `35→15/480`); `K3.45-B1B2` `86.90%` (`31/480` unf), halves `23` vs `8` (`p = 0.008`, `[E3.5]–[E3.7]`) |
| **Phase 3C** | `L3C.1–L3C.2` | **3C-A PASS / 3C-B `14/240`** | `M_class(exit_10)` `−0.427` nats (`5.5σ`); `K3.45-B1B2` fresh mixed replication `14/240` unf (pooled 12p `45/720 = 6.25%`, `[E3.8]–[E3.9]`) |
| **Phase 3D** | `L3D.1` | **Stopped Early** | Arm #0 `k325-rebuilt − native = −0.347` nats (`81%` of `K4 − native`), leaving `K4 − rebuilt = −0.081` (not detected at `exit_10`, `[E3.10]`) |
| **Phase 3E** | `L3E.1` | **Complete (`30/480`)** | `k325-rebuilt` `87.01%` (`30/480`, `p=0.57`) ≈ native; `K4 − rebuilt = +3.85 pp [+0.70, +6.93]`; `k345 − rebuilt = −0.10 pp`; `−1.44 pp` cross-library finished penalty (`[E3.11]–[E3.12]`) |
| **Phase 3F** | `L3F.1` | **Complete (`22/320`)** | Complement `k377-not-b1b2` (`+18.98 GiB`, `30/42` layers K4): `86.33%`, `22/320` unf (`11+11`, `p=1.00`), intermediate (`~50%` recovery); matches K4 on `join` (`9` vs `8`), misses `ledger` (`13` vs `6`, `[E3.13]–[E3.14]`) |
| **Phase 4** | `L4.1–L4.3` | **Not Run (Closed)** | No sparse `K3.25` repair justified cross-engine deployment on `vLLM v0.9.2` |

---

## 3. Practical & Methodological Takeaways

1. **Always include a matched stitched control (`k325-rebuilt`) in weight-graft experiments:** Without Stage 3D/3E, both `M_class`'s `−0.347`-nat library shift and `k345-b1b2`'s `−1.1 pp` finished-only deficit would have been misattributed to the target band.
2. **Never deploy cross-run intra-layer grafts to production:** Use weight stitching only for rapid screening; any winning mixed-precision map should be re-quantized in a single coherent sequential pass with matching dense-core precision.
3. **Measure anchor gaps on the target failure regime before searching:** Protecting the dense core in BF16/F32 already eliminates short-context `K3.25 → K4` regressions; sparse Kintsugi bit reallocation is better targeted at lower base bitrates (e.g., `K3.0 → K3.25`, where `B_close` headroom is `2.6×` larger) or single-run coherent intermediate quants (`K3.50`) calibrated with long-context trajectories.
4. **Separate reasoning quality from loop stability:** Because native `K3.25-v1` already equals `K4` on finished `128K` replies (`94.02%` vs `93.79%`), lightweight zero-byte inference-side loop guardrails (thinking-budget close or online repetition/dump interruption inside `<think>`) offer a more direct path to recovering the `+3.7 pp` gap than spending `+19–27 GiB` on expert bits.
