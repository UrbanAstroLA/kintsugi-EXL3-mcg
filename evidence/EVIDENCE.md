# Evidence Ledger

Every quantitative claim in this repository cites a row `[E#.#]` below. Failed, neutral, and diagnostic checks are retained.

| ID | Date | Engine / Scope | Fact / Measurement | Receipt / Source |
|---|---|---|---|---|
| **E0.1** | 2026-11-23 | CPU metadata inspection | `GLM-5.3-Flash-EXL3-K3.25-v1@0490d2f7` has `1,618` BF16 core tensors and `37,152` MCG EXL3 routed expert projections (`28,080` K3, `9,072` K4: `1,701` gate, `2,835` up, `4,536` down). K4 allocation is uniform per layer across `L03–L44` (`215–217 / 864 = 25.0%` per layer; `L45` MTP `0%`). | `docs/PHASE0-INVENTORY.md` §1 (`K3.25-v1/quantize_config.json`) |
| **E0.2** | 2026-11-23 | CPU tensor inspection | All `37,152` routed expert projections exist in `K3-v1@319d66a8` (100% K3) and `K4-Core-K6@5d3fffe1` (100% K4) with identical MCG schema (`mcg_multiplier = 3417055213`, `int16` trellis, `float16` `suh`/`svh`). Spot-checked `.trellis` SHA-256 hashes differ across `K3-v1`, `K3.25-v1`, and `K4-Core-K6`, confirming distinct quantization runs and requiring a matched stitched control (`K3.25-rebuilt`). | `docs/PHASE0-INVENTORY.md` §2 |
| **E0.3** | 2026-11-23 | Source check (`CuteAFD v6.0.0` `15148d339`) | All 14 top-level keys of `K3.25-v1/quantize_config.json` pass `exl3_contract.rs:219–227`; CuteAFD v6.0.0 supports two-tier `{K3, K4}` routed experts (`v41_exl3.rs`, `v41_exl3_residency.rs:262`) and retains BF16 core tensors in BF16. | `docs/PHASE0-INVENTORY.md` §3 |
| **E0.4** | 2026-11-23 | Source check (`vLLM fork v0.9.2`) | `patches/port-exl3-projection-mixed-glm53.py:339–349` supports arbitrary per-projection `{K3, K4}` mixtures per layer (at most two consecutive integer tiers per layer, `codebook="mcg"`, `EP=1`, `TP=2`, BF16 non-expert core) via `MODEL_DIR_OVERRIDE`. | `docs/PHASE0-INVENTORY.md` §3 |
