# kintsugi-EXL3-mcg

Applying the [Kintsugi method](https://github.com/professorpalmer/kintsugi-method) (Palmer, 2026) — task-gated, empirical mixed-precision bit allocation — to **EXL3 MCG** routed experts on **GLM-5.3-Flash** (`K3.25` bitrate budget, 2× RTX PRO 6000 Blackwell Max-Q, 300 W).

> **Engine scope:** Primary search and localization run on **CuteAFD v6.0.0** (chosen for bitwise prefill determinism, fast warm restarts, native `/v1/bench/probe` margin scoring, and true BF16 dense-core execution on plain checkpoints). Held-out cross-engine confirmation tests carry-over onto **tpurtell's vLLM fork v0.9.2** (`sha256:ea0430aa`, `TP=2`, `EP=1`). No result is claimed for any engine version other than the one measured.

---

## Motivation

Prior measurements in [cuteafd-eval](https://github.com/UrbanAstroLA/cuteafd-eval) and [local-inference-evals](https://github.com/UrbanAstroLA/local-inference-evals) establish two facts about `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1`:

1. **Precision loss sits in the 3.25-bit routed EXL3 experts, not the dense core.** All 1,618 non-expert tensors in `K3.25-v1` are already BF16. Upgrading routed experts from 3.25-bit to 4-bit moves teacher-panel KL by `−0.0195` to `−0.0212` nats (`0.0458 → 0.0246` on vLLM v0.9.2; `0.0476 → 0.0281` on CuteAFD v3–v5) and closes a `−5.73 pp` (`[−9.79, −1.98]`) 127K graded-task deficit (`0.8703` vs `0.9276`, 16 vs 4 unfinished replies; [cuteafd-eval `FC-k325-fp8c-vs-fp8c-tasks`](https://urbanastrola.github.io/cuteafd-eval/long-tasks.html)).
2. **`K3.25-v1` uses a uniform per-layer K4 quota.** Inspecting `K3.25-v1/quantize_config.json` shows every MoE layer `L03–L44` received the exact same 25.0% K4 share (`gate = 40–41`, `up = 67–68`, `down = 108`, total `215–217 / 864` per layer; `L45` MTP received `0%`), selected by a within-layer Hessian-weighted reconstruction proxy with zero cross-layer reallocation (`[E0.1]`).

Meanwhile, a complete `K ∈ {3, 4}` MCG EXL3 fit library (`mcg_multiplier = 3417055213`, identical tensor shapes and dtypes) already exists in published checkpoints (`K3-v1` and `K4-Core-K6`), and both **CuteAFD v6.0.0** and **vLLM fork v0.9.2** natively load arbitrary per-layer/per-projection `{K3, K4}` mixtures. This lets us test Kintsugi's core hypothesis directly at **exact matched 3.25 bpw** (and optionally modest headroom budgets) without re-quantizing from 643 GB BF16 weights.

---

## Method & Repository Layout

This repository follows Kintsugi's preregistration and open-seams discipline alongside `cuteafd-eval`'s evidence rules:

- **Pre-registered lines ([`evidence/LINES.md`](evidence/LINES.md)):** Every pass/fail threshold is written before the run it judges. Missed lines stay recorded as misses.
- **Evidence ledger ([`evidence/EVIDENCE.md`](evidence/EVIDENCE.md)):** Every reported number cites a row `[E#.#]`. Failed, neutral, and diagnostic runs are retained.
- **Neutral reporting:** `"not detected"` never means `"no effect"`; intervals and bounds are always stated; held-out confirmation uses items and seeds not used during allocation selection.

| Path | Purpose |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Staged, decision-driven experimental plan (Phases 0–4) with go/no-go gates and resource budgets |
| [`docs/BASELINES.md`](docs/BASELINES.md) | Reference baseline numbers from `cuteafd-eval` and `local-inference-evals` |
| [`docs/PHASE0-INVENTORY.md`](docs/PHASE0-INVENTORY.md) | Zero-GPU checkpoint audit, byte-identity checks, and CuteAFD v6.0.0 / vLLM v0.9.2 loader contracts |
| [`evidence/LINES.md`](evidence/LINES.md) | Pre-registered decision lines per phase |
| [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md) | Numbered evidence rows traced to artifacts |
| [`protocols/`](protocols/) | Frozen per-phase protocols (hashed before first request) |
| [`tools/`](tools/) | Checkpoint stitcher, launch checks, margin probe drivers, and analysis scripts |
| [`results/`](results/) | Frozen per-run summaries and manifests (no raw weight files or logit dumps) |

---

## Current Status

- **Phase 0 (CPU audit & feasibility):** **In progress** — checkpoint allocation audit, library schema check, and CuteAFD v6.0.0 / vLLM v0.9.2 manifest contract checks complete (`[E0.1]–[E0.4]`). Next: build and verify the `K3/K4` checkpoint stitcher (`K3.25-rebuilt` control) and run zero-GPU CuteAFD v6.0.0 planner checks.
- **Phase 1–4 (GPU runs):** Not yet started.

---

## Credits & References

- **Kintsugi method:** Cary Palmer (2026), *Kintsugi: Find and Repair Quantization Damage Where Tasks Break, Not Where Averages Look*, [github.com/professorpalmer/kintsugi-method](https://github.com/professorpalmer/kintsugi-method), DOI [10.5281/zenodo.24455575](https://doi.org/10.5281/zenodo.24455575).
- **Engines & checkpoints:** `tpurtell` (`CuteAFD`, `vLLM` GLM-5.3-Flash fork, `wrldsuksgo2mars` EXL3 checkpoints); `zai-org/GLM-5.3-Flash`; `incoai/GLM-5.3-Flash-DFlash2`.
- **Prior baselines:** [UrbanAstroLA/cuteafd-eval](https://github.com/UrbanAstroLA/cuteafd-eval) and [UrbanAstroLA/local-inference-evals](https://github.com/UrbanAstroLA/local-inference-evals).

Maintained by Michael M (`UrbanAstroLA`). Code under MIT (`LICENSE`); prose and records under CC BY 4.0.
