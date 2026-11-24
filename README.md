# kintsugi-EXL3-mcg

Applying the [Kintsugi method](https://github.com/professorpalmer/kintsugi-method) (Palmer, 2026) — task-gated, empirical mixed-precision bit allocation — to **EXL3 MCG** routed experts on **GLM-5.3-Flash** (`K3.25` base quant, 2× RTX PRO 6000 Blackwell Max-Q, 300 W).

> **Goal:** Take a practical, high-headroom quantization (`K3.25-v1`, 4.71M KV tokens on vLLM v0.9.2) and repair its task-failing layers/bands so it closes the gap toward higher-precision K4 quants at minimal extra byte cost — testing both **targeted band upgrades (`K3.35–K3.45`)** and **iso-bitrate reallocation (`K3.25`)**.
>
> **Engine scope:** Primary search and localization run on **CuteAFD v6.0.0** (pinned by digest `sha256:f2bd29c7...`; chosen for bitwise prefill determinism, ~3× faster restarts than vLLM v0.9.2, native inline `/v1/bench/probe` margin scoring, and unquantized BF16/F32 dense-core execution on plain checkpoints). Held-out cross-engine confirmation tests carry-over onto **tpurtell's vLLM fork v0.9.2** (`sha256:ea0430aa`, `TP=2`, `EP=1`). No result is claimed for any engine version other than the one measured.

---

## Motivation

Prior measurements in [cuteafd-eval](https://github.com/UrbanAstroLA/cuteafd-eval) and [local-inference-evals](https://github.com/UrbanAstroLA/local-inference-evals) establish two facts that motivate this study:

1. **Precision loss sits in the 3.25-bit routed EXL3 experts, not the dense core.** All `1,618` non-expert tensors in `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` are already unquantized (`1,327` BF16 + `291` F32). Upgrading routed experts from 3.25-bit to 4-bit moves teacher-panel KL by `−0.0195` to `−0.0212` nats (`0.0458 → 0.0246` on vLLM v0.9.2; `0.0476 → 0.0281` on CuteAFD v3–v5). On CuteAFD `v5.0.0`, grafting the same K3.25 expert bytes onto the K6/`fp8-compact` core (`k325-fp8c`) still trailed K4 experts (`fp8c`) by `−5.73 pp` (`[−9.79, −1.98]`) at 127K (`0.8703` vs `0.9276`, with `16` vs `4` unfinished replies — `14` cap hits + `2` loop aborts vs `4` cap hits; [cuteafd-eval `FC-k325-fp8c-vs-fp8c-tasks`](https://urbanastrola.github.io/cuteafd-eval/long-tasks.html)). Native `K3.25-v1` (with its BF16/F32 core) has not yet been measured on the 60-item task set on CuteAFD.
2. **`K3.25-v1` uses a uniform per-layer K4 quota.** Inspecting `K3.25-v1/quantize_config.json` shows every MoE layer `L03–L44` received the exact same 25.0% K4 share (`gate = 40–41`, `up = 67–68`, `down = 108`, total `215–217 / 864` per layer; `L45` MTP received `0%`), selected by a within-layer Hessian-weighted reconstruction proxy with zero cross-layer reallocation (`[E0.1]`).

A complete `K ∈ {3, 4}` MCG EXL3 fit library (`mcg_multiplier = 3417055213`, identical tensor shapes and dtypes, exact `1.000 MiB` delta per K3→K4 projection) already exists in published checkpoints (`K3-v1` and `K4-Core-K6`), and Phase 0 verified that stitched checkpoints pass CuteAFD `v6.0.0`'s CPU planner cleanly (`[E0.5]–[E0.7]`).

---

## Method & Repository Layout

- **Pre-registered lines ([`evidence/LINES.md`](evidence/LINES.md)):** Every pass/fail threshold is written before the run it judges. Missed lines remain recorded as misses.
- **Evidence ledger ([`evidence/EVIDENCE.md`](evidence/EVIDENCE.md)):** Every reported number cites a row `[E#.#]`. Failed, neutral, and diagnostic runs are retained.
- **Neutral reporting:** `"not detected"` never means `"no effect"`; intervals, power bounds, and byte costs are always stated; held-out confirmation uses items and seeds not used during allocation selection.

| Path | Purpose |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Staged, decision-driven experimental plan (Phases 0–4) with ETAs, power statements, and go/no-go gates |
| [`docs/BASELINES.md`](docs/BASELINES.md) | Published baseline numbers from `cuteafd-eval` and `local-inference-evals` |
| [`docs/PHASE0-INVENTORY.md`](docs/PHASE0-INVENTORY.md) | Zero-GPU checkpoint audit, byte-identity checks, and loader contract verification |
| [`results/phase0/PHASE0-SUMMARY.md`](results/phase0/PHASE0-SUMMARY.md) | Verified Phase 0 stitcher (`L0.1–L0.2`) and CuteAFD `v6.0.0` CPU planner (`L0.3`) receipts |
| [`results/phase1/PHASE1-SUMMARY.md`](results/phase1/PHASE1-SUMMARY.md) | Verified Phase 1 determinism (`L1.1`), `B_close` margin probe (`L1.2`), and calibration offset (`L1.3`) receipts |
| [`results/phase2/PHASE2A-SUMMARY.md`](results/phase2/PHASE2A-SUMMARY.md) | Verified Phase 2 Stage 2A 7-band P-B damage map (`L2.1-v2` PASS: `4.31×` spread, `B1–B2` concentration on `≤ 14.4K` P-B panel) |
| [`results/phase2/PHASE2B-SUMMARY.md`](results/phase2/PHASE2B-SUMMARY.md) | Verified Phase 2 Stage 2B 10-flip localization (`K3.45-B1B2` `50.2%` KL headroom, `K3.35-B2` `36.0%`, `K3.25-iso-hot` `16.3%` at `0 B`) |
| [`results/phase3/PHASE3A-SUMMARY.md`](results/phase3/PHASE3A-SUMMARY.md) | Verified Phase 3 Stage 3A held-out continuous gate (`L3.1` PASS on all 3; `K3.45-B1B2` `52.8%` P-A KL; `K3.25-iso-hot` `23.7%` P-A KL at `0 B`; `exlr8-k4` neutral check) |
| [`protocols/PHASE1.md`](protocols/PHASE1.md) | Frozen Phase 1 protocol |
| [`protocols/PHASE2.md`](protocols/PHASE2.md) | Frozen Phase 2 Stage 2A layer-band scan protocol (with pre-data `Amendment 2A.1`) |
| [`protocols/PHASE2B.md`](protocols/PHASE2B.md) | Frozen Phase 2 Stage 2B protocol (with pre-data `Amendment 2B.1`) |
| [`protocols/PHASE3.md`](protocols/PHASE3.md) | Frozen Phase 3 Stage 3A (held-out continuous + neutral-author gate) & Stage 3B (`4K–127K` task confirmation) protocol |
| [`evidence/LINES.md`](evidence/LINES.md) | Pre-registered decision lines per phase |
| [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md) | Numbered evidence rows traced to artifacts |
| [`tools/stitch_exl3.py`](tools/stitch_exl3.py) | Zero-dependency per-layer EXL3-MCG checkpoint stitcher and SHA-256 verifier |

---

## Current Status

| Phase | Status | Engine | GPU Time | Wall ETA | Key Outcome |
|---|---|---|---|---|---|
| **Phase 0** — Stitcher & planner verification | **PASS (`L0.1–L0.3`)** | CPU / CuteAFD `v6.0.0` planner | `0 h` | Done (`~6 min`) | `0` header/SHA mismatches; 4/4 `READY to serve`; `27–36 s` per incremental 6-layer band flip |
| **Phase 1** — Determinism, library offset & two-component margin probe | **PASS (`L1.1–L1.3`)** | CuteAFD `v6.0.0` | `~1.7 h` | Done (`~2.0 h`) | Exact `0.0` restart spread; `B_close` (`</think>`) `+0.277` nat K4 vs K3.25 (`+0.731` vs K3, `23/24`); `A_min` retired; `\|ΔKL_PB\| = 0.0006` |
| **Phase 2A** — Coarse 7-band damage map (`14` flips) | **PASS (`L2.1-v2`)** | CuteAFD `v6.0.0` | `~3.8 h` | Done (`3.8 h`) | **`4.31×` P-B KL spread (`4.5σ`)**, `ρ_up_down_KL = +0.68`; **`B1–B2 (L09–L20)` carry `55.8%` of P-B KL headroom** (`≤ 14.4K`; `B_close` confirmatory pending 2B shuffles) |
| **Phase 2B** — Null controls, role splits, routing-targeted & iso-bitrate (`10` flips) | **PASS (`L2.3`)** | CuteAFD `v6.0.0` | `~2.9 h` | Done (`2.9 h`) | `K3.45-B1B2` (`50.2%` KL headroom in `27.7%` bytes), `K3.35-B2` (`36.0%`, `1.44×` KL/GiB), `K3.25-iso-hot` (`16.3%` at `0 B`, CI `< 0`); shuffle KL `~0` |
| **Phase 3A** — Held-out continuous gate (P-A, `D_margin_heldout`, `exlr8-k4` neutral) | **PASS (`L3.1–L3.2`)** | CuteAFD `v6.0.0` | `~2.9 h` | Done (`2.9 h`) | `K3.45-B1B2`: `52.8%` P-A KL, `80.2%` `[44%, 128%]` K4-authored held-out `B_close`; `K3.25-iso-hot`: `23.7%` P-A KL at `0 B` (`CI < 0`); `exlr8-k4` neutral `+0.252` vs rebuilt (not detected at `n=24`), `+0.271` vs native (`CI > 0`) |
| **Phase 3B** — `4K + 127K` 60-item task confirmation (`8`-pass `Amendment 3B.2` extension) | **In Progress** | CuteAFD `v6.0.0` | `~6.65 h` done + `~11.4 h` ext | `~11.4 h` remaining | 4-pass anchor stop (`+3.75 pp`, CI `[−0.05, +7.71]`; unfinished `16→6/240`, CI `[−7.92, −0.42]`); extending anchors + `k345-b1b2` to 8 passes (`Z ≥ 1.96` inverse-normal); `k325-iso-hot` dropped from task runs |
| **Phase 4** — Cross-engine carry-over | Planned (gated) | vLLM fork `v0.9.2` | `~6.5–8 h` | `~6.5–8 h` | Tasks, `tool-eval-bench`, speed, and KV pool |

---

## Credits & References

- **Kintsugi method:** Cary Palmer (2026), *Kintsugi: Find and Repair Quantization Damage Where Tasks Break, Not Where Averages Look*, [github.com/professorpalmer/kintsugi-method](https://github.com/professorpalmer/kintsugi-method), DOI [10.5281/zenodo.24455575](https://doi.org/10.5281/zenodo.24455575).
- **Engines & checkpoints:** `tpurtell` (`CuteAFD`, `vLLM` GLM-5.3-Flash fork, `wrldsuksgo2mars` EXL3 checkpoints); `zai-org/GLM-5.3-Flash`; `incoai/GLM-5.3-Flash-DFlash2`.
- **Prior baselines:** [UrbanAstroLA/cuteafd-eval](https://github.com/UrbanAstroLA/cuteafd-eval) and [UrbanAstroLA/local-inference-evals](https://github.com/UrbanAstroLA/local-inference-evals).

Maintained by Michael M (`UrbanAstroLA`). Code under MIT (`LICENSE`); prose and records under CC BY 4.0.
