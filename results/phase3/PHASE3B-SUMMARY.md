# Phase 3 Stage 3B Summary — 127K Task Confirmation (`L3.3`: FAIL on `K3.45-B1B2`)

**Status:** Complete (`2026-11-25`) | **Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, `max_tokens = 32,768`, concurrency `8`, `4.96M` KV pool) | **Wall time:** `18.36 h` (`6.65 h` Stage 1 4-pass anchors + `11.71 h` Stage 2 extension; `~36.7` card GPU-h) | **Protocol:** [`protocols/PHASE3.md`](../../protocols/PHASE3.md) (`Amendment 3A.1`, `3B.1`, `3B.2`)

---

## Executive Verdict

1. **The 128K anchor gap (`k4-bf16core` vs `k325-native`) reproduced decisively under the pre-registered two-stage inverse-normal test (`[E3.5]`):**
   - Stage 1 (`rep 0–3`, `240` replies/arm): `+3.75 pp` (`z1 = +1.8958`, `CI [−0.05, +7.71]`, triggering the pre-registered 8-pass extension); unfinished reduction `−4.17 pp` (`16 → 6/240`, `z1_unf = +2.2210`, `CI [−7.92, −0.42]`).
   - Stage 2 (`rep 4–7`, `240` replies/arm): `+3.65 pp` (`z2 = +1.3994`, `CI [−1.15, +9.11]`); unfinished reduction `−4.17 pp` (`19 → 9/240`, `z2_unf = +1.7465`, `CI [−9.17, 0.00]`).
   - **Exact inverse-normal combination (`Z = (z1 + z2)/√2 ≥ 1.96`, one-sided `α = 0.025`):** **`Z_acc_combined = +2.3301` (DETECTED)** and **`Z_unf_combined = +2.8054` (DETECTED)**.
   - **Pooled 8-pass (`480` replies/arm):** `k325-native` **`87.16%`** (`35/480 = 7.29%` unfinished) vs `k4-bf16core` **`90.86%`** (`15/480 = 3.12%` unfinished) → **`+3.70 pp` (`CI [+0.55, +7.06]`)**, unfinished **`−4.17 pp` (`CI [−7.08, −1.46]`)**.
2. **`K3.45-B1B2` (`B1–B2`, `L09–L20` upgraded to K4, `+7.59 GiB`, `3.4535 bpw`) FAILED `L3.3` decisively (`[E3.6]`):**
   - Despite capturing **`52.8%`** of untouched P-A KL headroom (`[E3.1]`), **`50.2%`** of P-B KL headroom (`[E2.8]`), and **`80.2%`** (`[44%, 128%]`) of K4-authored held-out `B_close` margin headroom (`[E3.3]`), `K3.45-B1B2` scored **`86.90%`** at 128K (`31/480 = 6.46%` unfinished):
     - vs `k325-native`: **`−0.26 pp`** (`CI [−3.20, +2.76]`, recovery share **`−7.0%`**, `CI [−263%, +63%]`); unfinished **`−0.83 pp`** (`CI [−3.96, +2.08]`, recovery share **`20.0%`**, `CI [−100%, +79%]`).
     - vs `k4-bf16core`: **`−3.96 pp`** (`CI [−6.74, −1.12]`, `z = −2.77`), unfinished **`+3.33 pp`** (`CI [+0.62, +6.04]`, `z = −2.47`) — statistically indistinguishable from `k325-native` and significantly worse than `k4-bf16core` on both accuracy and unfinished-reply rate.
     - Finished-only accuracy was `92.90%` vs `94.02%` (`k325-native`, paired both-finished `−1.07 pp`, `CI [−2.36, +0.16]`), also tripping the `−1.0 pp` non-inferiority guardrail.
   - **Gate outcome:** **`L3.3` = FAIL (`L3_3_pass = false`).** Phase 4 (`vLLM v0.9.2`) is blocked for `K3.45-B1B2`.
3. **Failure-mode mechanism & proxy boundary (`[E3.7]`):**
   - Across `1,440` 128K replies, **`100%`** of the `k4 − k325` accuracy gap comes from rare runaway thinking loops (`35/35` `k325-native` and `15/15` `k4-bf16core` unfinished replies ended inside thinking with `content_chars = 0`; finished-only accuracy is identical at `94.02%` vs `93.79%`, median finished tokens `4,918` vs `5,219`).
   - Under the frozen deterministic tail classifier, `34/35` (`k325-native`), `14/15` (`k4-bf16core`), and `29/31` (`k345-b1b2`) unfinished replies are `record_list` or `line_repeat` runaway loops in `join` and `ledger` (`trace` has `1, 1, 0` unfinished across `160` replies/arm).
   - **Empirical boundary established:** Offline local Hessian error failed to rank depth sensitivity (`ρ = −0.464`, `[E2.4]`); end-to-end KL (`≤ 14.4K`) cleanly localized middle depth (`B1–B2`, `4.31×`, `4.5σ`, transferring `50.2% → 52.8%` to untouched P-A); **yet neither `≤ 14.4K` KL nor end-of-trace `B_close` (`</think>`) localized the 128K runaway record-listing failure mode** — empirically validating the Kintsugi principle that surrogate averages and finished-trace endpoints can decouple from the specific long-horizon failure mode that breaks tasks.

---

## 1. Main 8-Pass Results Table (`CuteAFD v6.0.0`, BF16/F32 Core)

| Arm | Size Δ | `4K` (4p, `n=240`) | `128K` Stage 1 (`rep 0–3`) | `128K` Stage 2 (`rep 4–7`) | **`128K` Pooled 8-Pass (`n=480`)** | `128K` Fin-Only | `128K` Unfinished (`cap / rep`) | In-Thinking (`content=0`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`k325-native`** (`3.2442 bpw`) | `0 B` | `98.12%` (`0` unf) | `87.81%` (`16/240`) | `86.51%` (`19/240`) | **`87.16%`** | `94.02%` (`n=445`) | **`35/480` (`7.29%`; `33 / 2`)** | `35 / 35` |
| **`k4-bf16core`** (`3.9942 bpw`) | `+27.43 GiB` | `98.70%` (`0` unf) | `91.56%` (`6/240`) | `90.16%` (`9/240`) | **`90.86%`** | `93.79%` (`n=465`) | **`15/480` (`3.12%`; `12 / 3`)** | `15 / 15` |
| **`k345-b1b2`** (`3.4535 bpw`, `L09–L20` K4) | `+7.59 GiB` | `98.23%` (`0` unf) | `84.06%` (`23/240`) | `89.74%` (`8/240`) | **`86.90%`** | `92.90%` (`n=449`) | **`31/480` (`6.46%`; `29 / 2`)** | `30 / 31` |

*Integrity check:* All `2,160` graded replies returned `status = ok` with `0` runtime server warnings/errors and `0` GPU thermal throttling events. CuteAFD's EXL3 shared-read layer log confirmed exact K4 byte reads (`3,634,498,944` B/layer) on `L09–L20` and `K3.25` byte reads (`2,953,973,120 / 2,956,070,272` B/layer) on `L03–L08` and `L21–L44`.

---

## 2. Pre-Registered Statistical Tests (`Amendment 3B.2`)

### 2.1 Exact Two-Stage Inverse-Normal Anchor Test (`k4-bf16core − k325-native`, 128K)

| Endpoint | Stage 1 (`rep 0–3`) | Stage 2 (`rep 4–7`) | Combined `Z = (z1+z2)/√2` | Threshold | Verdict |
|---|---:|---:|---:|---:|---|
| **128K overall accuracy** | `+3.75 pp` (`SE 1.98`, `z1 = +1.8958`) | `+3.65 pp` (`SE 2.61`, `z2 = +1.3994`) | **`+2.3301`** | `≥ 1.96` | **DETECTED** |
| **128K unfinished reduction** | `−4.17 pp` (`SE 1.88`, `z1 = +2.2210`) | `−4.17 pp` (`SE 2.39`, `z2 = +1.7465`) | **`+2.8054`** | `≥ 1.96` | **DETECTED** |

### 2.2 Descriptive Pooled 8-Pass Contrasts (`60`-Item Cluster Bootstrap, `10,000` Resamples, Seed `20261123`)

| Contrast (`B − A`, 128K) | Accuracy `Δ` | 95% CI | Unfinished `Δ` | 95% CI |
|---|---:|---:|---:|---:|
| **`k4-bf16core − k325-native`** | **`+3.70 pp`** | `[+0.55, +7.06]` | **`−4.17 pp`** | `[−7.08, −1.46]` |
| **`k345-b1b2 − k325-native`** | **`−0.26 pp`** | `[−3.20, +2.76]` | **`−0.83 pp`** | `[−3.96, +2.08]` |
| **`k345-b1b2 − k4-bf16core`** | **`−3.96 pp`** | `[−6.74, −1.12]` | **`+3.33 pp`** | `[+0.62, +6.04]` |
| `4K` (`rep 0–3`): `k345-b1b2 − k325-native` | `+0.10 pp` | `[−0.68, +0.94]` | `0.00 pp` | `[0.00, 0.00]` |

- **`L3.3` Recovery Shares (`k345-b1b2`):**
  - Accuracy recovery share: **`−7.0%`** (`95% CI [−263.3%, +63.4%]`; target `≥ 50%`).
  - Unfinished-rate recovery share: **`+20.0%`** (`95% CI [−100.0%, +78.6%]`; target `≥ 50%`).
  - Both-finished paired contrast (`k345-b1b2 − k325-native`): `−1.07 pp` (`95% CI [−2.36, +0.16]`); `k4-bf16core − k325-native`: `−0.33 pp` (`95% CI [−1.91, +1.13]`).

---

## 3. Per-Rep Profile, Family Breakdown & Deterministic Tail Taxonomy

### 3.1 Per-Rep 128K Accuracy (`unfinished count`)

| Arm | `r0` | `r1` | `r2` | `r3` | `r4` | `r5` | `r6` | `r7` | Dispersion `χ²(7)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k325-native` | `90.4% (2)` | `82.3% (7)` | `87.5% (5)` | `91.0% (2)` | `87.7% (4)` | `84.0% (6)` | `87.9% (4)` | `86.5% (5)` | `5.00` |
| `k4-bf16core` | `91.5% (2)` | `92.9% (1)` | `90.4% (1)` | `91.5% (2)` | `91.0% (1)` | `90.0% (3)` | `90.8% (3)` | `88.8% (2)` | `2.60` |
| `k345-b1b2` | `87.9% (3)` | `87.3% (4)` | `85.0% (5)` | `76.0% (11)` | `88.5% (2)` | `88.8% (2)` | `90.2% (3)` | `91.5% (1)` | `17.77` |

*Dispersion note:* `k345-b1b2` exhibited higher rep-to-rep dispersion (`23/240` unfinished in `rep 0–3` including `11/60` in `rep 3`, vs `8/240` in `rep 4–7`; `0` server errors/warnings/throttles). All 8 pre-registered reps are retained untrimmed for every reported statistic.

### 3.2 Descriptive Family Split (`160` Replies/Family at 128K)

| Family | `k325-native` (overall / fin / unf) | `k4-bf16core` (overall / fin / unf) | `k345-b1b2` (overall / fin / unf) |
|---|---|---|---|
| **`join`** | `86.64%` / `99.02%` / `20` (`12.50%`) | `94.22%` / `99.18%` / `8` (`5.00%`) | `87.50%` / `98.59%` / `18` (`11.25%`) |
| **`ledger`** | `79.06%` / `86.64%` / `14` (`8.75%`) | `82.58%` / `85.80%` / `6` (`3.75%`) | `77.34%` / `84.18%` / `13` (`8.12%`) |
| **`trace`** | `95.78%` / `96.38%` / `1` (`0.62%`) | `95.78%` / `96.38%` / `1` (`0.62%`) | `95.86%` / `95.86%` / `0` (`0.00%`) |

### 3.3 Frozen Deterministic Tail Classification & Finished Length Quantiles

| Arm | `record_list` | `line_repeat` | `computation` | Finished Completion Tokens (`p25 / p50 / p75 / p90`) | Finished Reasoning Chars (`p25 / p50 / p75 / p90`) |
|---|---:|---:|---:|---|---|
| **`k325-native`** (`35` unf) | `29` | `5` | `1` | `3,144 / 4,918 / 7,707 / 11,920` | `7,009 / 10,223 / 17,803 / 27,926` |
| **`k4-bf16core`** (`15` unf) | `11` | `3` | `1` | `3,188 / 5,219 / 7,412 / 10,781` | `6,984 / 10,296 / 16,471 / 25,084` |
| **`k345-b1b2`** (`31` unf) | `23` | `6` | `2`* | `3,228 / 4,827 / 7,616 / 11,828` | `6,954 / 9,788 / 17,108 / 26,679` |

\*Inspecting the `2` `computation`-classified tails on `k345-b1b2` shows both were also degenerate listing/repetition loops that missed the regex (`ledger-106 r5`: bulleted 5-digit ledger listing `- 09867 ELS.`; `ledger-115 r1`: inline `045? 045? ...` repeat abort). The single non-thinking unfinished reply (`ledger-114 r3`, `225` content chars) hit the `32,768`-token cap mid-JSON right after a `line_repeat` thinking tail.

---

## 4. Scientific Interpretation & Open Hypotheses

This negative confirmation on `K3.45-B1B2` is a decisive, clean result across the entire Phase 1–3 evidence chain:

| Signal | Local Offline Hessian (`K3-v1`) | End-to-End `≤ 14.4K` KL (`P-B` / `P-A`) | End-of-Trace `B_close` (`</think>`) | **Live 128K Context-Probe (`480` replies/arm)** |
|---|---|---|---|---|
| **Does `B1–B2 (L09–L20)` recover ≥ 50% of K4?** | No (`ρ = −0.464`, missed) | **Yes** (`50.2%` P-B, **`52.8%`** untouched P-A) | **Yes** on K4-authored held-out (`80.2%`); `B1` not separated on `n=24` `exlr8-k4` neutral (`+0.010`) | **NO** (`−7.0%` acc share, `20.0%` unf share; trails K4 by `−3.96 pp`, `CI [−6.74, −1.12]`) |

Three falsifiable hypotheses explain why `k4-bf16core` cuts 128K runaway record-listing by `−4.17 pp` (`35 → 15/480`) while `K3.45-B1B2` (`L09–L20`) does not:
1. **Context-length / depth mismatch:** `B1–B2` dominates `≤ 14.4K` teacher-forced KL, whereas runaway record-listing occurs only at `128K` (`0/240` at `4K` on all 3 arms) and may be governed by a different depth band (e.g., early `B0` or deeper `B3–B6` layers that `≤ 14.4K` KL ranked lower).
2. **Wrong probe position (`</think>` vs loop onset) + authorship confounding:** `B_close` evaluates `</think>` at the end of a *finished* reasoning trace, whereas real task failures enter `record_list` loops early/mid-thinking and never reach an answer state; in addition, `B1`'s held-out `B_close` lift did not yet separate on independent `exlr8-k4` trajectories (`n=24`).
3. **Distributed rather than sparse stability:** Preventing rare `128K` runaway loops may require broader K4 coverage across multiple bands rather than a single `12/42`-layer Kintsugi patch.
