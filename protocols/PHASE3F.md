# Phase 3 Stage 3F Protocol — Complement Graft (`k377-not-b1b2`) Localized-vs-Diffuse Test (`L3F.1`)

**Status:** FROZEN before first Stage 3F GPU request (`2026-11-26`)
**Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, `max_tokens = 32,768`, concurrency `8`)
**Question:** Stage 3E established that full `k4-bf16core` beats matched stitched `k325-rebuilt` at `128K` (`+3.85 pp`, `[+0.70, +6.93]`; `+4.84 pp`, `[+0.39, +9.22]` on `join + ledger`, with unfinished rate `−4.69 pp`, `[−9.06, −0.31]`), whereas upgrading `B1–B2 (L09–L20)` to `100%` K4 (`k345-b1b2`) showed no detected live task gain (`−0.10 pp` overall, `+0.21 pp` unfinished `[E3.11]`). Does K4's `128K` runaway stability live **localized in layers outside `B1–B2`** (`B0` or `B3–B6`, disjoint from `≤ 14.4K` P-B/P-A KL), or is it **diffuse / super-additive across depth** (requiring precision across `B1–B2` together with the rest of the stack)?

---

## 1. Complement Arm (`k377-not-b1b2`)

Built deterministically via [`tools/stitch_exl3.py`](../tools/stitch_exl3.py) using `--reuse-from` on `k325-rebuilt` (hardlinking `core.safetensors`, `L09–L20`, and `L45`; rewriting `30` layer shards `L03–L08` and `L21–L44`) and verified (`0` header mismatches, `300` expert + `100` core SHA-256 checks) before server launch:

| Property | `k325-rebuilt` (matched base) | `k345-b1b2` (`B1–B2` K4) | **`k377-not-b1b2` (complement)** | `k4-bf16core` (ceiling) |
|---|---:|---:|---:|---:|
| Source library | Stitched `K3-v1 + K4-Core-K6` | Stitched `K3-v1 + K4-Core-K6` | **Stitched `K3-v1 + K4-Core-K6`** | Single-run `K4-Core-K6` |
| MoE layers at `100%` K4 | `0 / 42` | `12 / 42` (`B1–B2: L09–L20`) | **`30 / 42` (`B0: L03–L08`, `B3–B6: L21–L44`)** | `42 / 42` (`L03–L44`) |
| MoE layers at `25%` K4 (`K3.25` map) | `42 / 42` (`L03–L44`) | `30 / 42` (`B0`, `B3–B6`) | **`12 / 42` (`B1–B2: L09–L20`)** | `0 / 42` |
| `L45` (MTP) | `0%` K4 | `0%` K4 | **`0%` K4** | `100%` K4 |
| K4 / K3 routed projections | `9,072` / `28,080` | `16,848` / `20,304` | **`28,512` / `8,640`** | `37,152` / `0` |
| Effective expert `bpw` | `3.2442` | `3.4535` | **`3.7674`** | `4.0000` |
| Delta bytes vs `k325-rebuilt` | `0 B` | `+7.5938 GiB` | **`+18.9844 GiB` (`+19,440 MiB`)** | `+27.4219 GiB` |

*(Note: `k377-not-b1b2` costs `5/7` of K4's MoE upgrade bytes — this arm is a scientific localization experiment, not a sparse deployment candidate.)*

---

## 2. Targeted Task Suite (`join + ledger` `128K`, Paired `reps 0–7`)

Unmodified Stage 3B task driver (`run_tasks_dc.py`, `stage3b/FROZEN.json` untouched), restricted via existing CLI flags `--lengths 128k --families join ledger --concurrency 8 --max-tokens 32768` across `reps 0–7`:

- **Items & pairing:** All `40` `128K` items in `join` (`20`) and `ledger` (`20`) × `8` passes = **`320` paired replies**, using the exact same `reps 0–7` deterministic seeds (`900M..907M`) and nonces as `k325-native`, `k325-rebuilt`, `k345-b1b2`, and `k4-bf16core`.
- **Why `join + ledger` `128K`-only:** Across all four completed 8-pass arms (`1,920` 128K replies), `join + ledger` accounts for **`108 / 111` (`97.3%`)** of all unfinished replies (`trace` had `1, 1, 0, 1` out of `160`; `4K` had `0, 1, 0, 0` out of `240`). On `join + ledger` `128K` alone, the anchor separation against matched `k325-rebuilt` is cleaner on both primary metrics:

| Existing Arm (`join + ledger` `128K`, `n = 320`, `reps 0–7`) | Overall Acc | Fin-Only Acc | Unfinished (`r0–3 + r4–7`, halves `p`) | `join` unf | `ledger` unf | `join` enter (`≥ 15`) |
|---|---:|---:|---:|---:|---:|---:|
| **`k325-native`** | `82.85%` | `92.70%` | `34/320` (`10.62%`; `16+18`, `p = 0.86`) | `20` | `14` | `28/160` |
| **`k325-rebuilt`** (matched base) | `83.55%` | `91.88%` | **`29/320` (`9.06%`; `17+12`, `p = 0.44`)** | `13` | `16` | `22/160` |
| **`k345-b1b2`** (`B1–B2` K4) | `82.42%` | `91.26%` | `31/320` (`9.69%`; `23+8`, `p = 0.007`) | `18` | `13` | `26/160` |
| **`k4-bf16core`** (ceiling) | **`88.40%`** | `92.44%` | **`14/320` (`4.38%`; `5+9`, `p = 0.41`)** | `8` | `6` | `13/160` |

- **Existing paired `40`-item cluster bootstrap anchor contrasts (`10,000` resamples, seed `20261123`):**
  - `k4-bf16core − k325-rebuilt`: **`+4.84 pp` (`z = +2.14`, `CI [+0.39, +9.22]`, DETECTED)**; unfinished **`−4.69 pp` (`z = −2.05`, `CI [−9.06, −0.31]`, DETECTED)**; both-finished `+0.70 pp` (`[−0.35, +1.67]`).
  - `k4-bf16core − k325-native`: **`+5.55 pp` (`z = +2.39`, `CI [+1.05, +10.12]`, DETECTED)**; unfinished **`−6.25 pp` (`z = −3.03`, `CI [−10.31, −2.19]`, DETECTED)**.
  - `k345-b1b2 − k325-rebuilt`: `−1.13 pp` (`[−5.39, +3.05]`); unfinished `+0.62 pp` (`[−3.75, +5.00]`).

---

## 3. Pre-Registered Decision Line (`L3F.1`) & Secondary Diagnostics

Analyzed automatically by [`tools/analyze_phase3f.py`](../tools/analyze_phase3f.py) using `40`-item cluster bootstrap (`10,000` resamples, seed `20261123`).

### Primary reading bands on `k377-not-b1b2` `join + ledger` `128K` unfinished count (out of `320`)

| Observed `k377-not-b1b2` unfinished (`n = 320`) | Frozen `L3F.1` Reading | Interpretation & Next Action |
|---|---|---|
| **`≤ 19 / 320` (`≤ 5.94%`)** | **`k377-not-b1b2 ≈ k4-bf16core` (`14/320`)** | 128K runaway stability is **localized outside `B1–B2`** (disjoint from the `≤ 14.4K` KL map). Opens optional bisection of `B0 + B3–B6` (`B3–B4` vs `B5–B6` vs `B0`) to test whether a smaller band subset outside `B1–B2` captures K4's stability. |
| **`≥ 25 / 320` (`≥ 7.81%`)** | **`k377-not-b1b2 ≈ k325-rebuilt` (`29/320`)** | **Diffuse / super-additive across depth:** neither `B1–B2` (`31/320`) nor its `5/7` complement (`≥ 25/320`) recovers K4's stability alone — ruling out any sparse band patch on `K3.25-v1` for this behaviour. |
| **`20–24 / 320` (`6.25%–7.50%`)** | **Intermediate / partial recovery** | Consistent with gradual/proportional scaling with total K4 bit share (~`5/7` of the `15`-reply `k4 − rebuilt` gap ≈ `11` replies → `18/320`; partial recovery of `5–9` replies → `20–24/320`); inspect paired CIs and family split before deciding whether further bisection is warranted. |

### Secondary Diagnostics (logged in `STAGE3F_SUMMARY.json`)

1. **Paired contrasts (`k377-not-b1b2` vs `k325-rebuilt`, `k4-bf16core`, `k325-native`, `k345-b1b2`):** overall accuracy, unfinished rate, and both-finished accuracy.
2. **Cross-library stitching penalty check (`both-finished` vs `k325-native` & `k4-bf16core`):** tests whether raising K4 share from `25%` (`k325-rebuilt`) / `45%` (`k345-b1b2`) to `77%` (`k377-not-b1b2`, with `30/42` layers 100% single-library K4) reduces the `~1.3–1.4 pp` finished-only deficit observed in Stage 3E (`[E3.12]`).
3. **Halves stability (`reps 0–3` vs `reps 4–7`, Fisher exact `p`)**, `join` vs `ledger` family breakdown, `join` entering-vs-leaving (`≥ 15`-line run), deterministic tail classes (`record_list`, `line_repeat`, `computation`), and finished completion-token quantiles.
