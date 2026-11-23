# Reference Baselines (from Published `cuteafd-eval` and `local-inference-evals`)

All numbers below trace to published receipts on the same two-GPU workstation (2× RTX PRO 6000 Blackwell Max-Q, 300 W) and remain scoped strictly to the engine version on which they were measured.

---

## 1. Target Baseline: `K3.25-v1` on tpurtell vLLM Fork `v0.9.2`

- **Checkpoint:** `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1` rev `0490d2f7` (137 GiB; unquantized BF16/F32 dense core + 37,152 MCG EXL3 routed experts: 28,080 K3 + 9,072 K4).
- **Engine:** `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:ea0430aa` (`v0.9.2`, `TP=2`, `EP=1`, `KV_CACHE_DTYPE=fp8_ds_mla`, `DFlash2` K3, `util=0.95`, thinking on).

| Metric | `K3.25-v1` (vLLM `v0.9.2`) | Comparator (`K4` on vLLM `v0.9.2`) | Source |
|---|---:|---:|---|
| P-A Teacher-Panel KL (51,175 rows, full vocab) | `0.0458` nats (Top-1 `93.50%`, NLL `1.0731`) | `0.0246` nats (`tr3-4bpw`) | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm`, `2026-10-27-k4-vllm-cuteafd` |
| P-B Fidelity-Set KL (15,104 gen rows, 0.5K–14.4K) | `0.0451–0.0457` nats (Top-1 `93.84%`, jitter `±0.0006`) | — | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm` |
| Q2 Human-Text NLL (32K–128K, 6 docs) | `1.2329` nats | — | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm` |
| `tool-eval-bench` (88 scenarios / 176 pts, 3 reps) | `160 / 163 / 160` (`91–93%`; `4.81–4.90 s/turn`) | `163 / 158 / 160` | `cuteafd-eval` `2026-11-02-tool-calling` |
| Block L Long Output (128K cap, 8 streams each) | Code `7/8` (217/218 tests), Math `8/8`, Prose `8/8`, JSON `6/8` | — | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm` |
| GPQA Diamond (`v0.9.1` 3-pass / `B1` 1-pass) | `86.9%` raw (`87.9%` stated) / `87.4%` raw (9 empty) | `85.7%` raw (`87.7%` stated) | `local-inference-evals` `FINDINGS.md` |
| Block T Speed (C1 code decode, 1K–128K) | `145.2–150.2 tok/s` (TTFT `0.25 s` @ 1K, `35.02 s` @ 128K) | — | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm` |
| DFlash2 K3 Acceptance | `1.53 / 3` (`51%`) | — | `cuteafd-eval` `2026-10-26-cuteafd-vs-vllm` |
| KV Pool & Warm Ready Time | `4,707,515` tok (`4.49×` 1M), `382 s` warm | `1,377,179` tok (`1.31×` 1M) | `local-inference-evals` / pinned recipe |

---

## 2. K3.25 vs K4 Precision Isolation (Published CuteAFD `v5.0.0`, `cuteafd-eval`)

These published runs measured the exact same K3.25 routed expert bytes on CuteAFD `v5.0.0` (either in `K3.25-Core-FP8` or grafted onto the K6/`fp8-compact` core in `k325-fp8c`; native `K3.25-v1` with its BF16/F32 core has not run the 60-item task set on CuteAFD):

| Configuration (CuteAFD `v5.0.0`) | P-B KL | P-A KL | 4K Tasks | 127K ITT (4 passes) | 127K Fin-Only | 127K Unfinished (/240) | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| `k325-fp8` (`K3.25` experts + FP8 core) | `0.0485` | `0.0476` | `0.9849` | `0.8667` | — | `16` | `2026-11-10-context-probe`, `2026-11-15-fp8-core-contrasts` |
| `k325-hqa` (`K3.25` experts + HQA core graft) | `0.0462` | `0.0472` | — | — | — | — | `2026-11-15-fp8-core-contrasts` (`FC-k325-graft-kl`) |
| `k325-fp8c` (`K3.25` experts + `fp8-compact` graft) | `0.0495` | `0.0486` | — | `0.8703` | `0.9325` | `16` (`14` cap, `2` rep) | `2026-11-15-fp8-core-contrasts` (`FC-k325-fp8c-vs-fp8c-tasks`) |
| `fp8c` (`K4` experts + `fp8-compact` core) | `0.0327` | `0.0326` | `0.9807` | `0.9276` | `0.9433` | `4` (`4` cap, `0` rep) | `2026-11-15-fp8-core-contrasts` |
| `hqa` (`K4` experts + HQA core) | `0.0302` | `0.0315` | `0.9844` | `0.9245` | — | `5` | `2026-11-10-context-probe` |

**Key paired contrasts (CuteAFD `v5.0.0`):**
- Pure expert step (`k325-fp8c − fp8c`, matched `fp8-compact` core): **`+0.0168 [+0.0134, +0.0202]`** P-B KL; **`−0.0573 [−0.0979, −0.0198]`** at 127K ITT (`16` vs `4` unfinished — roughly `4.7 pp` of the `5.73 pp` gap comes from non-completion, mostly 32,768-token cap hits).
- Pure core step on K3.25 (`k325-fp8c − k325-fp8`): **`+0.0036 [−0.0365, +0.0448]`** at 127K (not detected; identical `16` unfinished replies).

---

## 3. CuteAFD `v6.0.0` Baselines

No `K3.25` checkpoint has been measured on CuteAFD `v6.0.0` in published `cuteafd-eval` runs yet. Phase 1 establishes the `CuteAFD v6.0.0` baselines directly for `K3.25-v1`, `K3.25-rebuilt`, `K4-bf16core`, and `K3-bf16core`.
