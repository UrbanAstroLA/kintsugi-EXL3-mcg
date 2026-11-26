# Phase 3 Stage 3E Summary — Matched-Library Control (`k325-rebuilt`) 8-Pass 127K Task Confirmation (`L3E.1`: `30/480` → `K4` Gain Is K4-Specific)

**Status:** Complete (`2026-11-26`) | **Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, `max_tokens = 32,768`, concurrency `8`, `0` runtime warnings/errors) | **Protocol:** [`protocols/PHASE3E.md`](../../protocols/PHASE3E.md) (frozen at `9c7ae9c`)

---

## Executive Verdict (`L3E.1`, `[E3.11]–[E3.12]`)

Running the matched stitched control **`k325-rebuilt`** (`K3-v1` + `K4-Core-K6` experts on `K3.25-v1` BF16/F32 core, exact `9,072`-projection K4 map) on all `8` paired passes (`480` 128K + `240` 4K replies, 1-to-1 on identical `reps 0–7` seeds `900M..907M`) produced four sharp, decisive conclusions:

1. **Frozen `L3E.1` reading (`30/480 ≥ 30`):** **`k325-rebuilt ≈ k325-native` (`87.01%` vs `87.16%`; `30/480 = 6.25%` vs `35/480 = 7.29%` unfinished; halves consistent `17+13`, Fisher `p = 0.57`).**
   - **`k4-bf16core − k325-rebuilt` = `+3.85 pp` (`z = +2.43`, `95% CI [+0.70, +6.93]`, DETECTED)** — proving K4's live 128K task gain (`30–35/480 → 15/480` unfinished) is **genuinely K4-specific**, not a quirk of native `K3.25-v1`.
   - Stage 3D's `−0.347` nat `k325-rebuilt − k325-native` shift on `M_class(exit_10)` (`[E3.10]`) was therefore a **quantization-library probe artifact**, not the live task mechanism.
2. **True matched-library effect of `B1–B2` (`k345-b1b2 − k325-rebuilt`) is zero across every live 128K metric:**
   - Overall 128K accuracy: **`−0.10 pp`** (`86.90%` vs `87.01%`, `CI [−3.10, +2.86]`, `z = −0.07`).
   - 128K unfinished rate: **`+0.21 pp`** (`31/480 = 6.46%` vs `30/480 = 6.25%`, `CI [−2.71, +3.12]`; `k345-b1b2` 12-pass descriptive rate is also `45/720 = 6.25%`).
   - Paired both-finished 128K accuracy: **`−0.0015 pp`** (`CI [−1.11, +1.10]`).
   - Upgrading `B1–B2 (L09–L20)` to 100% K4 (`+7.59 GiB`, `52.8%` untouched `≤ 14.4K` P-A KL, `80.2%` K4-authored `B_close`) delivered **no live 128K task recovery** over its exact matched stitched base.
3. **Detected cross-library stitching penalty on finished 128K reasoning (`[E3.12]`):**
   - Paired both-finished 128K accuracy for **`k325-rebuilt − k325-native` = `−1.44 pp` (`95% CI [−2.76, −0.14]`, DETECTED)** (`92.81%` `k325-rebuilt` and `92.90%` `k345-b1b2` vs `94.02%` single-run `K3.25-v1` and `93.79%` single-run `K4-Core-K6`).
   - This explains the consistent `~1 pp` finished-only deficit previously observed across all three 4-pass blocks of `k345-b1b2` (`92.97 / 92.83 / 93.14%`): grafting experts across two independent `true_sequential` EXL3 calibration runs (`K3-v1` + `K4-Core-K6`) is invisible on `≤ 14.4K` P-A/P-B KL (`0.0006` nats) and `4K` tasks (`98.12%` vs `98.12%`), yet costs **`−1.44 pp`** on finished `128K` reasoning replies. Any production mixed-precision EXL3-MCG allocation must therefore be **re-quantized inline (`true_sequential: true`)** rather than deployed as a cross-run tensor graft.
4. **Family nuance on `k325-rebuilt`:** `k325-rebuilt` partially reduced `join` unfinished replies (`13/160` vs `20/160` native; `join` leaving rate `9/22 = 40.9%` vs `8/28 = 28.6%`), while losing on `ledger` (`16/160` unf, `84.90%` fin-only vs `14/160`, `86.64%` native) and `trace` (`94.50%` vs `96.38%` fin-only), netting to `−0.16 pp` overall vs native.

---

## 1. Full 4-Arm 8-Pass Table (`CuteAFD v6.0.0`, BF16/F32 Core, Paired `reps 0–7`)

| Arm | Source Library | `4K` (4p, `n=240`) | `128K` `r0–3` (`unf`) | `128K` `r4–7` (`unf`) | **`128K` All 8p (`n=480`)** | `128K` Fin-Only | **`128K` Unfinished** (`halves p`) | `join` Enter / Leave (`≥15`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`k325-native`** (`3.2442 bpw`) | Single-run `K3.25-v1` | `98.12%` (`0`) | `87.81%` (`16`) | `86.51%` (`19`) | **`87.16%`** | **`94.02%`** | **`35/480` (`7.29%`, `p=0.73`)** | `28/160` (`8/28 = 28.6%`) |
| **`k325-rebuilt`** (`3.2442 bpw`) | Stitched `K3-v1 + K4-Core-K6` | `98.12%` (`1`) | `86.30%` (`17`) | `87.71%` (`13`) | **`87.01%`** | **`92.81%`** | **`30/480` (`6.25%`, `p=0.57`)** | `22/160` (`9/22 = 40.9%`) |
| **`k345-b1b2`** (`3.4535 bpw`, `L09–L20` K4) | Stitched `K3-v1 + K4-Core-K6` | `98.23%` (`0`) | `84.06%` (`23`) | `89.74%` (`8`) | **`86.90%`** | **`92.90%`** | **`31/480` (`6.46%`, `p=0.008`)** | `26/160` (`8/26 = 30.8%`) |
| **`k4-bf16core`** (`3.9942 bpw`) | Single-run `K4-Core-K6` | `98.70%` (`0`) | `91.56%` (`6`) | `90.16%` (`9`) | **`90.86%`** | **`93.79%`** | **`15/480` (`3.12%`, `p=0.60`)** | `13/160` (`5/13 = 38.5%`) |

*(All `k325-rebuilt` `30/30` 128K unfinished replies ended inside thinking with `content_chars = 0`; per-rep unfinished `r0..r7 = 5, 3, 5, 4, 2, 4, 5, 2`; tails `20 record_list + 5 line_repeat + 5 computation`; finished completion tokens `p25/p50/p75/p90 = 2,936 / 5,064 / 8,310 / 12,379`.)*

---

## 2. Paired 8-Pass 128K Contrasts (`60`-Item Cluster Bootstrap, `10,000` Resamples, Seed `20261123`)

| Contrast (`B − A`, 128K) | Overall Acc `Δ` (95% CI) | Unfinished Rate `Δ` (95% CI) | Both-Finished Acc `Δ` (95% CI) |
|---|---:|---:|---:|
| **`k4-bf16core − k325-rebuilt`** | **`+3.85 pp`** `[+0.70, +6.93]` (`z=+2.43`) | `−3.12 pp` `[−6.25, 0.00]` | `+1.11 pp` `[−0.09, +2.32]` |
| **`k4-bf16core − k325-native`** | **`+3.70 pp`** `[+0.55, +7.06]` (`z=+2.26`) | **`−4.17 pp`** `[−7.08, −1.46]` | `−0.33 pp` `[−1.91, +1.13]` |
| **`k325-rebuilt − k325-native`** | `−0.16 pp` `[−3.72, +3.70]` (`z=−0.08`) | `−1.04 pp` `[−5.00, +2.71]` | **`−1.44 pp` `[−2.76, −0.14]`** |
| **`k345-b1b2 − k325-rebuilt`** | `−0.10 pp` `[−3.10, +2.86]` (`z=−0.07`) | `+0.21 pp` `[−2.71, +3.12]` | `−0.00 pp` `[−1.11, +1.10]` |

---

## 3. Family Breakdown (`160` Replies/Family at 128K, Paired `reps 0–7`)

| Family | `k325-native` (all / fin / unf) | `k325-rebuilt` (all / fin / unf) | `k345-b1b2` (all / fin / unf) | `k4-bf16core` (all / fin / unf) |
|---|---|---|---|---|
| **`join`** | `86.64%` / `99.02%` / `20` (`12.5%`) | `90.70%` / `98.72%` / `13` (`8.1%`) | `87.50%` / `98.59%` / `18` (`11.2%`) | `94.22%` / `99.18%` / `8` (`5.0%`) |
| **`ledger`** | `79.06%` / `86.64%` / `14` (`8.8%`) | `76.41%` / `84.90%` / `16` (`10.0%`) | `77.34%` / `84.18%` / `13` (`8.1%`) | `82.58%` / `85.80%` / `6` (`3.8%`) |
| **`trace`** | `95.78%` / `96.38%` / `1` (`0.6%`) | `93.91%` / `94.50%` / `1` (`0.6%`) | `95.86%` / `95.86%` / `0` (`0.0%`) | `95.78%` / `96.38%` / `1` (`0.6%`) |
