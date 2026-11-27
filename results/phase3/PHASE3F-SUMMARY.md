# Phase 3 Stage 3F Summary — Complement Graft (`k377-not-b1b2`) 8-Pass `join + ledger` 127K Test (`L3F.1`: `22/320` → Intermediate / Partial Recovery)

**Status:** Complete (`2026-11-26`) | **Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core, `max_tokens = 32,768`, concurrency `8`, `0` server `ERROR`s, `0` task errors) | **Protocol:** [`protocols/PHASE3F.md`](../../protocols/PHASE3F.md) (frozen at `d8ccd0e`)

---

## Executive Verdict (`L3F.1`, `[E3.13]–[E3.14]`)

Running the `5/7` complement graft **`k377-not-b1b2`** (`B0 [L03–L08]` + `B3–B6 [L21–L44]` at `100%` K4, leaving `B1–B2 [L09–L20]` at `k325-rebuilt`; `30/42` MoE layers at K4, `3.7674 bpw`, `+18.98 GiB`, `0` header/SHA mismatches across `1,314` checked tensors) on all `8` paired passes of `128K` `join + ledger` (`40` items × `8` = `320` replies, exact `reps 0–7` seeds `900M..907M`) answered the localized-vs-diffuse question cleanly:

1. **Frozen `L3F.1` reading (`22/320` in `20–24/320`): Intermediate / partial recovery (stable across halves: `11 + 11`, Fisher `p = 1.000`).**
   - **`k377-not-b1b2`** scored **`86.33%`** overall (`fin_only = 92.70%`, **`22/320 = 6.88%`** unfinished; per-rep `r0..r7 = 1, 3, 3, 4, 2, 4, 3, 2`), sitting roughly halfway between matched base `k325-rebuilt` (`83.55%`, `29/320`) and ceiling `k4-bf16core` (`88.40%`, `14/320`).
   - Point recovery over `k325-rebuilt` was **`+2.77 pp` / `+4.84 pp` (`57.3%`)** on accuracy (`CI [−1.05, +6.80]`, `z = +1.38`) and **`7/15` (`46.7%`)** on unfinished replies (`−2.19 pp [−6.25, +1.88]`), for **`69.2%`** (`+18.98 / +27.42 GiB`) of K4's expert byte increase (`5/7 = 71.4%` of MoE layers). Neither `k377-not-b1b2 − k325-rebuilt` nor `k377-not-b1b2 − k4-bf16core` (`−2.07 pp [−5.43, +1.68]`, unf `+2.50 pp [−0.62, +5.62]`) excluded `0` on its own.
2. **Sharp family split (`join` vs `ledger`, diagnostic `n = 20` items / `160` replies each):**
   - **`join`:** `k377-not-b1b2` matched full K4 (`93.52%` vs `94.22%`; `9/160` vs `8/160` unf; **`join` loop entry `13/160 = 8.13%`**, identical to K4's `13/160`, vs `22/160` rebuilt [`p = 0.151`], `26/160` `k345-b1b2` [`p = 0.039`], and `28/160` native [`p = 0.018`]; paired accuracy `+6.88 pp [+0.39, +14.92]` vs native and `+6.02 pp [+0.31, +11.80]` vs `k345-b1b2`).
   - **`ledger`:** `k377-not-b1b2` remained high on unfinished replies (**`13/160 = 8.12%`**, vs `k325-rebuilt` `16/160`, `k345-b1b2` `13/160`, `k325-native` `14/160`, and full K4 **`6/160 = 3.75%`**; `k377 − k4` unf `+4.38 pp [+0.00, +8.12]`). Neither `B1–B2` (`13/160`) nor its `5/7` complement (`13/160`) recovered K4's `ledger` stability (`6/160`).
3. **Finished-only cross-library diagnostic (`[E3.12]` follow-up):**
   - With `30/42` MoE layers promoted to `100%` single-library K4 (leaving only `12/42` mixed-within-layer), `k377-not-b1b2`'s `join + ledger` finished-only accuracy returned to **`92.70%`** — identical to single-run `k325-native` (`92.70%`, paired both-finished `−0.14 pp [−1.71, +1.74]`) and single-run `k4-bf16core` (`92.44%`, `+0.42 pp [−1.23, +2.35]`), above the two heavily mixed-layer stitched arms (`k325-rebuilt` `91.88%`, `k345-b1b2` `91.26%`; paired `+0.95 pp [−0.32, +2.23]` and `+1.27 pp [−0.27, +2.82]`). Directional only (CIs span `0`), consistent with the Stage 3E mixed-calibration hypothesis.
4. **Conclusion for `K3.25-v1` targeted repair:**
   - Upgrading the `≤ 14.4K` KL-hot `B1–B2` band (`2/7` of layers, `+7.59 GiB`) yielded no detected `128K` task recovery (`31/320` vs `29/320`), while upgrading its entire `5/7` complement (`+18.98 GiB`, `69%` of K4's extra bytes) recovered only roughly half of the overall gap (`22/320` vs `14/320`) and left `ledger` unfinished replies near baseline (`13/160` vs `6/160`). Because K4's `128K` stability is distributed/super-additive across depth rather than concentrated in a cheap band subset, **no sparse band patch exists on `K3.25-v1` for this behaviour**, and further bisecting `B0 + B3–B6` is not warranted.

---

## 1. Full 5-Arm 8-Pass `join + ledger` `128K` Table (`CuteAFD v6.0.0`, BF16/F32 Core, Paired `reps 0–7`, `n = 320`/Arm)

| Arm | K4 MoE Layers (`+GiB`) | **`join + ledger` Acc** | Fin-Only Acc | **Unfinished** (`r0–3 + r4–7`, halves `p`) | `join` (acc / unf) | `ledger` (acc / unf) | `join` Enter / Leave (`≥ 15`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`k325-native`** (`3.2442 bpw`) | `0/42` (`0.00`) | `82.85%` | `92.70%` | `34/320` (`10.62%`; `16+18`, `p = 0.86`) | `86.64%` / `20` | `79.06%` / `14` | `28/160` (`8/28 = 28.6%`) |
| **`k325-rebuilt`** (`3.2442 bpw`, base) | `0/42` (`0.00`) | `83.55%` | `91.88%` | **`29/320` (`9.06%`; `17+12`, `p = 0.44`)** | `90.70%` / `13` | `76.41%` / `16` | `22/160` (`9/22 = 40.9%`) |
| **`k345-b1b2`** (`3.4535 bpw`) | `12/42` (`+7.59`) | `82.42%` | `91.26%` | `31/320` (`9.69%`; `23+8`, `p = 0.007`) | `87.50%` / `18` | `77.34%` / `13` | `26/160` (`8/26 = 30.8%`) |
| **`k377-not-b1b2`** (`3.7674 bpw`) | **`30/42` (`+18.98`)** | **`86.33%`** | **`92.70%`** | **`22/320` (`6.88%`; `11+11`, `p = 1.00`)** | **`93.52%` / `9`** | **`79.14%` / `13`** | **`13/160` (`4/13 = 30.8%`)** |
| **`k4-bf16core`** (`4.0000 bpw`, ceiling) | `42/42` (`+27.42`) | **`88.40%`** | `92.44%` | **`14/320` (`4.38%`; `5+9`, `p = 0.41`)** | `94.22%` / `8` | `82.58%` / `6` | `13/160` (`5/13 = 38.5%`) |

*(All `22/22` unfinished `k377-not-b1b2` replies ended inside thinking with `content_chars = 0`; tail classes `11 record_list + 6 line_repeat + 5 computation`; finished completion tokens `p25/p50/p75/p90 = 2,094 / 5,502 / 9,754 / 13,792`.)*

---

## 2. Paired 8-Pass `join + ledger` `128K` Contrasts (`40`-Item Cluster Bootstrap, `10,000` Resamples, Seed `20261123`)

| Contrast (`B − A`, `n = 40` items, `320` paired replies) | Overall Acc `Δ` (95% CI) | Unfinished Rate `Δ` (95% CI) | Both-Finished Acc `Δ` (95% CI) |
|---|---:|---:|---:|
| **`k4-bf16core − k325-rebuilt`** (anchor gap) | **`+4.84 pp`** `[+0.39, +9.22]` (`z=+2.14`) | **`−4.69 pp`** `[−9.06, −0.31]` (`z=−2.05`) | `+0.70 pp` `[−0.35, +1.67]` |
| **`k4-bf16core − k325-native`** (anchor gap) | **`+5.55 pp`** `[+1.05, +10.12]` (`z=+2.39`) | **`−6.25 pp`** `[−10.31, −2.19]` (`z=−3.03`) | `−0.49 pp` `[−2.66, +1.47]` |
| **`k377-not-b1b2 − k325-rebuilt`** (primary) | `+2.77 pp` `[−1.05, +6.80]` (`z=+1.38`) | `−2.19 pp` `[−6.25, +1.88]` (`z=−1.03`) | `+0.95 pp` `[−0.32, +2.23]` |
| **`k377-not-b1b2 − k4-bf16core`** (gap to ceiling) | `−2.07 pp` `[−5.43, +1.68]` (`z=−1.13`) | `+2.50 pp` `[−0.62, +5.62]` (`z=+1.55`) | `+0.42 pp` `[−1.23, +2.35]` |
| **`k377-not-b1b2 − k325-native`** | `+3.48 pp` `[−1.13, +8.75]` (`z=+1.38`) | `−3.75 pp` `[−9.06, +0.94]` (`z=−1.49`) | `−0.14 pp` `[−1.71, +1.74]` |
| **`k377-not-b1b2 − k345-b1b2`** (complement vs `B1–B2`) | `+3.91 pp` `[−0.47, +8.36]` (`z=+1.73`) | `−2.81 pp` `[−7.19, +1.56]` (`z=−1.26`) | `+1.27 pp` `[−0.27, +2.82]` |
| **`k345-b1b2 − k325-rebuilt`** (`B1–B2` effect) | `−1.13 pp` `[−5.39, +3.05]` (`z=−0.52`) | `+0.62 pp` `[−3.75, +5.00]` (`z=+0.28`) | `−0.66 pp` `[−1.95, +0.63]` |

---

## 3. Diagnostic Per-Family Paired Contrasts (`20` Items / `160` Replies per Family)

| Family | Contrast (`B − A`) | Overall Acc `Δ` (95% CI) | Unfinished Rate `Δ` (95% CI) | Both-Finished Acc `Δ` (95% CI) |
|---|---|---:|---:|---:|
| **`join`** | `k377-not-b1b2 − k325-rebuilt` | `+2.81 pp` `[−2.03, +7.50]` | `−2.50 pp` `[−6.88, +2.50]` | `+0.13 pp` `[−1.06, +1.35]` |
| **`join`** | `k377-not-b1b2 − k325-native` | **`+6.88 pp`** `[+0.39, +14.92]` | `−6.88 pp` `[−15.00, +0.00]` | `+0.07 pp` `[−0.62, +0.73]` |
| **`join`** | `k377-not-b1b2 − k345-b1b2` | **`+6.02 pp`** `[+0.31, +11.80]` | `−5.62 pp` `[−11.25, +0.00]` | `+0.46 pp` `[−0.71, +1.68]` |
| **`join`** | `k377-not-b1b2 − k4-bf16core` | `−0.70 pp` `[−5.31, +4.06]` | `+0.62 pp` `[−3.75, +5.00]` | `+0.12 pp` `[−1.06, +1.42]` |
| **`ledger`** | `k377-not-b1b2 − k325-rebuilt` | `+2.73 pp` `[−3.20, +9.14]` | `−1.88 pp` `[−8.75, +4.38]` | `+1.77 pp` `[−0.48, +3.90]` |
| **`ledger`** | `k377-not-b1b2 − k325-native` | `+0.08 pp` `[−5.55, +6.64]` | `−0.62 pp` `[−6.88, +5.00]` | `−0.35 pp` `[−3.32, +3.31]` |
| **`ledger`** | `k377-not-b1b2 − k4-bf16core` | `−3.44 pp` `[−8.05, +2.27]` | `+4.38 pp` `[+0.00, +8.12]` | `+0.71 pp` `[−2.36, +4.21]` |
