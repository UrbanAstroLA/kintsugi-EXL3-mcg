# Phase 3 Stage 3C Summary — Loop Entry/Exit Margin Probe (`L3C.1`: PASS) & `K3.45-B1B2` Mixed-Regime Replication (`L3C.2`: Unresolved, `14/240`)

**Status:** Steps 1–2 Complete (`2026-11-25`) | **Engine:** CuteAFD `v6.0.0` (`sha256:f2bd29c7...`, BF16/F32 dense core) | **Protocol:** [`protocols/PHASE3C.md`](../../protocols/PHASE3C.md) (frozen at `37a1ef6`; Step 1 prediction committed pre-3C-B at `55b2dbd`)

---

## Executive Summary

1. **Stage 3C-A (`L3C.1`: PASS against `k325-native`, confounded by `k325-rebuilt` library offset `[E3.8]`, `[E3.10]`):** Moving the teacher-forced decision-margin instrument from `</think>` (`B_close`) to the **exact mid-thinking record-dump loop exit decision** (`M_class(exit_10)`, symmetric class log-odds over `pool = top_10 ∪ want`) separated `k4-bf16core` from `k325-native` at **`5.5σ`** (`−0.4272` nats, `SE 0.0771`, `95% CI [−0.5685, −0.2673]`, median pool coverage `≥ 0.99997`) and passed the authorship check on `k4`-authored dumps (`n = 11`: `−0.1461 < 0`).
   - **Critical matched-control finding (Stage 3D arm #0, `[E3.10]`):** Scoring the matched stitched control **`k325-rebuilt`** (`K3-v1` + `K4-Core-K6`, identical `9,072`-projection K4 map) on the same `72` dump prefixes showed **`k325-rebuilt − k325-native = −0.3466` nats (`CI [−0.4817, −0.2182]`)** — **`81%`** of `k4 − native` (present even on `k345`-authored `−0.379` and `exlr8-k4`-authored `−0.479` dumps, ruling out self-authorship). Against `k325-rebuilt`, `k4-bf16core − k325-rebuilt = −0.0805` (`CI [−0.1987, +0.0454]`, **not detected** at `exit_10`; detected only at `exit_20`: `−0.1365 [−0.2790, −0.0248]`), and `k345-b1b2 − k325-rebuilt = +0.0790 [−0.0318, +0.2001]` (`+0.2265 [+0.1061, +0.3029]` stickier on `ledger`). Stage 3C-A's `k4 − native` separation therefore reflects a quantization-library offset rather than a localized K4-specific effect at `exit_10`, and Stage 3D's 7-band scan was stopped early to test `k325-rebuilt` directly on live 8-pass tasks (Stage 3E).
   - **Prospective `k345-b1b2` prediction committed BEFORE Stage 3C-B (`55b2dbd`):** primary non-`k345`-authored (`n = 44`) recovery share **`65.0%` (`95% CI [25.7%, 98.6%]`)**; all-author (`n = 72`, self-authorship-affected) `62.7%` (`CI [29.7%, 87.8%]`).
   - **Family and depth breakdowns (descriptive; paired depth differences not detected):** `k345-b1b2` (`B1–B2`, `L09–L20` K4) showed `82.2%` of K4's `M_class(exit_10)` softening on `join` (`n = 47`) vs `−12.2%` (`+0.0311`, CI spans `0`) on `ledger` (`n = 25`), and non-`k345` point shares fell with loop depth (`enter 97.2%` → `exit_5 74.4%` → `exit_10 65.0%` → `exit_20 47.2%`, paired `exit_20 − enter = −0.46 [−0.97, +0.17]`, not detected).
2. **Stage 3C-B (`L3C.2`: `14/240` unfinished → frozen reading `"unresolved (13–17/240)"`, `[E3.9]`):** Replicating `k345-b1b2` on a fresh server in the mixed `4K+128K` regime (`reps 8–11`, `SEED_BASE_3C = 950M`, `0` runtime errors) produced **`87.71%`** overall accuracy (`fin_only = 93.14%`) and **`14/240` (`5.83%`)** unfinished (`r8: 5, r9: 3, r10: 3, r11: 3`):
   - Rules out both a strong mixed-regime breakdown (`≥ 18/240`, unlike Stage 1's `23/240`) and a clean K4-level recovery (`≤ 12/240`, unlike Stage 2's `8/240`).
   - **Descriptive pooled 12-rep `k345-b1b2` (`n = 720`, `L3.3` = FAIL not reopened):** `87.17%` overall (`fin_only = 92.98%`, consistently `~0.9–1.2 pp` below `k325-native`'s `94.02%` in all three 4-pass blocks), **`45/720` (`6.25%`)** unfinished vs `k325-native` `35/480` (`7.29%`) and `k4-bf16core` `15/480` (`3.12%`) — a descriptive `25.0%` pooled unfinished recovery (`35.0%` in `reps 8–11` alone).
   - **Prospective prediction check:** Directionally right, magnitude `~2×` high — Stage 3C-B's unfinished recovery share (`35.0%` vs 8-pass anchors; `25.0%` pooled 12-rep) fell inside Stage 3C-A's wide recorded primary 95% CI (`[25.7%, 98.6%]`), about `2×` below its `65.0%` point estimate.
   - **Entering vs. leaving decomposition on `join` (descriptive, all replies including finished):** Applying the `≥ 15`-line run detector to all 128K `join` replies shows `k4-bf16core` enters `≥ 15`-line recitations half as often as `k325-native` (`13/160 = 8.1%` vs `28/160 = 17.5%`, Fisher `p = 0.018`), with a smaller leaving-and-finishing difference (`5/13 = 38.5%` vs `8/28 = 28.6%`); `k345-b1b2` (`12p`) entered `33/240` (`13.8%`, `~40%` of the way to K4) and left `9/33` (`27.3%`). Because `M_class` measures stickiness *once inside* a recitation rather than the upstream decision to begin reciting, this plausibly explains why the probe's point recovery share overshot live task recovery by `~2×` (and why `join` and `ledger` under-recovered similarly in 12-pass tasks, `~22%` vs `~17%`).

---

## 1. Stage 3C-A — Loop Entry/Exit Margin Probe (`L3C.1`: **PASS**)

### 1.1 Primary & Secondary Contrasts (`M_class` unless noted; `10,000` item-cluster resamples, seed `20261123`)

| Slice (`pool` median cov `≥ 0.9997`) | `k4 − k325-native` (95% CI) | `k345 − k325-native` (95% CI) | `k345` Share (`n=44` non-k345) |
|---|---:|---:|---:|
| **`enter` (`n=72`)** | `−0.1858` `[−0.294, −0.052]` | `−0.1374` `[−0.266, +0.025]` | `97.2%` `[36.9%, 139.8%]` |
| **`exit_5` (`n=72`)** | `−0.5176` `[−0.668, −0.348]` | `−0.3899` `[−0.565, −0.193]` | `74.4%` `[52.2%, 99.9%]` |
| **`exit_10` (PRIMARY, `n=72`)** | **`−0.4272` `[−0.569, −0.267]`** | **`−0.2676` `[−0.418, −0.091]`** | **`65.0%` `[25.7%, 98.6%]`** |
| **`exit_20` (`n=61`)** | `−0.4453` `[−0.590, −0.297]` | `−0.1809` `[−0.324, +0.0002]` | `47.2%` `[8.5%, 69.2%]` |
| `M_stick(exit_10)` (`n=72`) | `−0.4485` `[−0.599, −0.277]` | `−0.2877` `[−0.461, −0.092]` | `65.5%` `[30.1%, 97.4%]` |
| `M_raw(exit_10)` (`n=72`) | `−0.4724` `[−0.617, −0.306]` | `−0.3061` `[−0.472, −0.117]` | `65.4%` `[29.6%, 97.4%]` |
| Author: `k325-native` (`n=30`) | `−0.5614` `[−0.751, −0.345]` | `−0.3550` `[−0.575, −0.129]` | `63.2%` |
| Author: `k345-b1b2` (`n=28`) | `−0.4401` `[−0.644, −0.226]` | `−0.2604` `[−0.510, +0.009]` | `59.2%` |
| Author: `k4-bf16core` (`n=11`) | **`−0.1461`** `[−0.503, +0.248]` | `−0.1241` `[−0.532, +0.401]` | `84.9%` |
| Author: `exlr8-k4` (`n=3`, `2` items) | `+0.0053` `[−0.226, +0.121]` | `+0.0126` `[−1.366, +0.702]` | descriptive |
| **Family: `join` (`n=47`)** | **`−0.5191` `[−0.687, −0.316]`** | **`−0.4265` `[−0.577, −0.254]`** | **`82.2%`** |
| **Family: `ledger` (`n=25`)** | **`−0.2543` `[−0.433, −0.064]`** | **`+0.0311` `[−0.200, +0.250]`** | **`−12.2%`** |

- **Within-family positive control check (`M_class` at `exit_10`):** Pooled `dump − control` was `−1.29` to `−1.51` nats (CI spans `0`) due to a family-mix Simpson effect (`controls` are `50%` `ledger` vs `dumps` `35%` `ledger`). Within `join`, dumps are **`+1.07` to `+1.38` nats stickier** than controls on all 3 arms (`k325-native` `4.332` vs `2.953`, `k345-b1b2` `3.906` vs `2.836`, `k4-bf16core` `3.813` vs `2.589`); within `ledger`, finished controls are deliberate sequential block scans (`9.18–9.41`), with `k4-bf16core` still less sticky than `k325-native` and `k345-b1b2` (`9.179` vs `9.401 / 9.413`).

---

## 2. Stage 3C-B — `K3.45-B1B2` Mixed-Regime Replication (`reps 8–11`, `L3C.2`)

### 2.1 Stage-Split Comparison (`128K`, `max_tokens = 32,768`, `0` runtime errors)

| Arm / Block | Regime | `128K` Accuracy | `128K` Fin-Only | `128K` Unfinished | Per-Rep Unf | Tail Classes (`rec / rep / comp`) | Dump Onsets (`≥15`, min/med/max) |
|---|---|---:|---:|---:|---|---|---|
| `k345-b1b2` `reps 0–3` | `4K+128K` mixed | `84.06%` | `92.97%` | `23/240` (`9.58%`) | `3, 4, 5, 11` | `17 / 5 / 1` | `21` (`1,501 / 6,199 / 58,122`) |
| `k345-b1b2` `reps 4–7` | `128K`-only | `89.74%` | `92.83%` | `8/240` (`3.33%`) | `2, 2, 3, 1` | `6 / 1 / 1` | `7` (`1,543 / 8,314 / 27,687`) |
| **`k345-b1b2` `reps 8–11` (3C-B)** | **`4K+128K` mixed (fresh)** | **`87.71%`** | **`93.14%`** | **`14/240` (`5.83%`)** | **`5, 3, 3, 3`** | **`10 / 3 / 1`** | **`8` (`2,763 / 6,615 / 9,181`)** |
| *`k345-b1b2` pooled 12-rep (desc.)* | *All 3 blocks (`n=720`)* | *`87.17%`* | *`92.98%`* | *`45/720` (`6.25%`)* | *mean `3.75`/rep* | *`33 / 9 / 3`* | *`36`* |
| *Ref: `k325-native` (`8p`, `n=480`)* | `r0–3` + `r4–7` | `87.16%` | `94.02%` | `35/480` (`7.29%`) | mean `4.38`/rep | `29 / 5 / 1` | `30` (`1,401 / 7,370 / 22,199`) |
| *Ref: `k4-bf16core` (`8p`, `n=480`)* | `r0–3` + `r4–7` | `90.86%` | `93.79%` | `15/480` (`3.12%`) | mean `1.88`/rep | `11 / 3 / 1` | `11` (`1,335 / 6,312 / 32,058`) |

- **`4K` sanity check (`reps 8–11`, `n = 240`):** `97.86%` overall (`98.27%` fin-only; `1/240` unfinished on `join` in `rep 9` via `line_repeat`, `0` dump onsets).
- **Family breakdown (`128K`):**
  - `join`: `k325-native` `86.64%` (`20/160 = 12.50%` unf) | `k4-bf16core` `94.22%` (`8/160 = 5.00%`) | `k345-b1b2` `r8–11` `88.59%` (`8/80 = 10.00%`), pooled 12-rep `87.86%` (`26/240 = 10.83%`).
  - `ledger`: `k325-native` `79.06%` (`14/160 = 8.75%` unf) | `k4-bf16core` `82.58%` (`6/160 = 3.75%`) | `k345-b1b2` `r8–11` `79.22%` (`6/80 = 7.50%`), pooled 12-rep `77.97%` (`19/240 = 7.92%`).
  - `trace`: `95.78%` (`1/160`) | `95.78%` (`1/160`) | `k345-b1b2` `r8–11` `95.31%` (`0/80`), pooled 12-rep `95.68%` (`0/240`).

### 2.2 Frozen Reading & Step 1 Prediction Score

- **Frozen `L3C.2` reading:** **`unresolved (13–17/240)`** (`14/240` unfinished; `L3.3` = FAIL is not reopened).
- **Prospective Step 1 prediction consistency:**
  - Recorded primary proxy prediction (`55b2dbd`, non-`k345` `n = 44`): **`65.0%` (`95% CI [25.7%, 98.6%]`)**.
  - Stage 3C-B replication unfinished recovery share vs 8-pass anchors (`7.29%` vs `3.125%`): **`35.0%`** → **inside** the recorded 95% CI (`replication_share_within_primary_proxy_ci95 = true`), though below the point estimate (`65.0%`), and descriptive pooled 12-rep unfinished recovery is `25.0%`.
- **Conditional Step 3 status:** Both formal Stage 3C conditions for proposing Step 3 (`L3C.1` PASS and Stage 3C-B within the Step 1 prediction CI) are met; Step 3 (7-band `B0–B6` scan on `M_class`, including `ledger` and `exit_20` where `B1–B2` leaves `53–100%` of K4's softening unrecovered) remains **conditional pending separate authorization**.
