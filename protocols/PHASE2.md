# Phase 2 Protocol — Layer-Band Damage Map (Stage 2A) & Sub-Band Localization (Stage 2B)

**Status:** FROZEN before first Phase 2 GPU request (`2026-11-23`)  
**Engine:** CuteAFD `v6.0.0` (`tag 15148d339`, `sha256:f2bd29c7...`, BF16/F32 dense core unquantized, `prefill_scoring = streamed`, `fp8-prefill = auto/Unchanged`)  
**Matched control baseline:** Phase 1 `k325-rebuilt-r0` (stitched from `K3-v1@319d66a8` + `K4-Core-K6@5d3fffe1` + `K3.25-v1@0490d2f7` core; `B_close = +3.6003` nats, `6/24` fragile `< 1.0`, `2/24` negative; P-B `gen` KL = `0.044447` nats, `[E1.2]–[E1.4]`).  
**Frozen probe:** Exact same `D_margin_build` (`n = 24` items, mean `132,947` tokens) frozen in Phase 1 (`D_margin_heldout`, `n = 36`, remains strictly untouched until Phase 3).  
**Primary task-margin metric:** **`B_close`** (`</think>` closing token `154842` margin, validated by `L1.2`, `[E1.2]`). Component A (`A_min`, retired by `L1.2`) and `B_tail_mean` are logged diagnostically only.  
**Secondary fidelity metric:** **P-B `gen` KL** (`15,104` generated rows across `53` clusters, validated via `validate_pb.py` same-dump `0`-mismatch check).

---

## 1. Stage 2A — 7 Six-Layer Band Scan (`14` single-band flips, ETA `~3.8` wall-h)

Partition MoE layers `L03–L44` into **7 symmetric six-layer bands** (`5,184` routed expert projections per band; `L45` MTP remains at `0%` K4 as in `K3.25-v1`). Every band in `K3.25-rebuilt` starts with exactly `3,888` K3 + `1,296` K4 projections (`25.0%` K4):

| Band | Layers | Base K3 / K4 | Upward Flip (`→ 100% K4`) | Downward Flip (`→ 0% K4`) |
|---|---|---:|---:|---:|
| `B0` | `L03–L08` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B1` | `L09–L14` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B2` | `L15–L20` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B3` | `L21–L26` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B4` | `L27–L32` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B5` | `L33–L38` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |
| `B6` | `L39–L44` | `3,888` / `1,296` | `+3,888 MiB` (`+3.7969 GiB`, `3.3488 bpw`) | `−1,296 MiB` (`−1.2656 GiB`, `3.2093 bpw`) |

### Per-Flip Execution Procedure (Rolling Scratch, `≤ 30 GiB` transient disk)
1. Stitch via `tools/stitch_exl3.py build --map-preset k325-native --override-band "<lo>-<hi>:<bits>" --reuse-from k325-rebuilt` (`38/44` shards hardlinked, `6` rewritten in `~30 s`), then `verify --sample-experts 100 --sample-core 20` (`0` mismatches required).
2. Launch CuteAFD `v6.0.0` (single-arm clean GPU check + smoke test `7 × 6 = 42` + `prefill_scoring = streamed`).
3. Run `tools/run_margin_probe.py --split build --top-k 5` (inline `/v1/bench/probe`, zero dumps, `~9.5 min`).
4. Run `cuteafd bench fidelity run --tier full --score-path prefill` (accepting `rc = 0` or `rc = 3` non-fatal low-bit alert), score on CPU via `score_pb.py`, verify `validate_pb.py` (`32,768/32,768` rows, `0` mismatches, `0.0` diff), compute paired contrast vs Phase 1 `pb-k325-rebuilt-r0.rows.npz`, and immediately delete raw P-B logit dumps.
5. Stop CuteAFD and immediately delete the temporary stitched flip directory before starting the next flip.

### Stage 2A Analysis & Pre-Registered Gate P2 (`L2.1`, `L2.2`)
Computed by `tools/analyze_phase2.py` against matched control `k325-rebuilt-r0`:
- For each upward flip (`+3.7969 GiB`):
  - `ΔB_close = mean(B_close(flip) − B_close(k325-rebuilt))` with paired bootstrap 95% CI (`10,000` resamples, seed `20261123`), `ΔB_close / GiB`, fragile (`< 1.0`) and negative (`< 0`) counts, and fraction of total `K4 − K3.25-rebuilt` headroom (`+0.2773` nat) recovered.
  - `ΔKL_PB = KL_PB_gen(flip) − KL_PB_gen(k325-rebuilt)` with paired cluster-bootstrap 95% CI, `−ΔKL_PB / GiB`, and fraction of total `K4 − K3.25-rebuilt` KL headroom (`0.013074` nat) recovered.
- For each downward flip (`−1.2656 GiB`):
  - `ΔB_close`, damage rate `−ΔB_close / GiB`, **hit-list count** (number of items flipping from `B_close ≥ 0` on `k325-rebuilt` to `B_close < 0` on the downward flip, plus new fragile `< 1.0` flips), and `ΔKL_PB / GiB`.
- Cross-band alignment: Spearman rank correlation $\rho$ (with bootstrap 95% CI) between upward `ΔB_close / GiB` and `−ΔKL_PB / GiB` across `B0..B6`, and between upward and downward sensitivity across `B0..B6`.
- **L2.1 (Non-flat sensitivity — PASS condition):**
  - The sensitivity ratio `max(sink) / min(safe source)` across `B0..B6` is **`≥ 2.0×`** in either `ΔB_close / GiB` or `|ΔKL_PB| / GiB` (computing both upward-only max/min and best upward sink vs cheapest safe downward source), **AND**
  - At least one downward-flipped source band has **hit list = `0`** (`0` new `B_close < 0` flips).
- **L2.2 (Stop rule):** If `L2.1` fails, stop before Stage 2B/Phase 3 and publish a clean negative result.

---

## 2. Stage 2B — Sub-Band Role & Routing-Targeted Localization (`~1.5–2.0` wall-h, gated on Stage 2A `L2.1` PASS)

Once Stage 2A identifies the top 1–2 sink bands (`S1`, `S2`) and cheapest safe source bands (`C1..Ck`), Stage 2B runs up to `6` targeted flips (frozen in `protocols/PHASE2B.md` before Stage 2B GPU launch) exploiting `[E0.8]` (`67.24×` median within-layer routing skew; `38.6%` of top-25% routed experts still at K3):

1. **Role splits on top sink band `S1` (3 flips, `~0.8` wall-h):** `gate_proj → K4`, `up_proj → K4`, `down_proj → K4` individually (`--override-band "<lo>-<hi>:4:<role>"`).
2. **Routing-targeted hot-K3 upgrade on `S1` (and `S1+S2`) (2 flips, `~0.5` wall-h):** Upgrade only the top-routed remaining K3 projections inside `S1` (ranked by `K3.25-v1/quant_log.csv` `samples` and/or `samples × err`, matching `~1/3` to `1/2` of the full-band byte cost, `~+1.27–1.90 GiB`) via `stitch_exl3.py --map-json`, testing whether hot-expert targeting captures a disproportionate share of the full-band `ΔB_close` gain per GiB.
3. **Iso-bitrate cold-to-hot swap (1 flip, `~0.3` wall-h):** Drop `1,296` K4 projections from the cheapest safe source band `C1` (`−1.2656 GiB`) and allocate all `+1,296` K4 slots to the hottest remaining K3 projections inside `S1` (or `S1+S2`) at **exact `0 B` tensor delta (`3.2500 bpw`)**.

**L2.3 (Selection rule for Phase 3):** Compare full-band sink upgrade (`S1` / `S1+S2`), best role/routing sub-band upgrade, and best iso-bitrate swap on `ΔB_close`, `ΔB_close / GiB`, and P-B `gen` KL non-inferiority (`ΔKL_PB ≤ +0.002` nats), advancing the primary deployment candidate (`K3.35–K3.45` or sub-band winner) and secondary iso-bitrate candidate (`K3.25-Kintsugi-iso` vs `K3.25-KL-iso`) into Phase 3.
