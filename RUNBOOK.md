# RUNBOOK.md — Google Colab execution guide

How to run `EXPERIMENTS_PLAN.md` on **Google Colab (T4 / L4)** after the `niftyhedge` package exists.

Rule: **stop at every gate**. Do not start the next stage if a gate fails. Every gate exists because the first version of this project skipped it.

The thin Colab wrapper is `notebooks/colab_run.ipynb`. Prefer that notebook; the commands below are the CLI contract each cell must call.

---

## 0. What runs where

| Stage | Where | Why |
|---|---|---|
| 0. Data download | **Your laptop in India** | NSE archive endpoints frequently reject cloud / Colab IPs. Download once, upload parquet to Drive. |
| 1. Surfaces, folds, unit tests | Colab CPU or laptop | SVI fitting is CPU work. |
| 2–6. Generators, hedgers, evaluation | **Colab GPU (T4/L4)** | All models are small MLPs. One 16 GB GPU is enough. |

### Compute estimate (one T4-class GPU)

| Stage | Estimate |
|---|---|
| Fit all generators, 6 folds, 5 seeds on main fold + 3 on others | 2–4 h |
| DDPM ablations (E5) | 3–5 h |
| Generate paths, all worlds and folds | under 1 h |
| Train hedgers: 5 worlds × 3 instrument sets × 5 seeds on main fold, 3 seeds elsewhere | 4–8 h |
| Cross-world and real backtest evaluation | under 1 h |
| Sensitivity sweeps (E10) | 4–8 h |
| **Total** | **about 15–25 GPU-hours** |

If any single stage is taking ~10× longer, something is running in a Python loop that should be vectorised. Stop and fix it.

---

## 1. One-time prep (before Colab)

### 1.1 Laptop: clone / open the rebuilt repo

```bash
cd ~/Desktop/Vol-Surfaces   # or wherever the rebuilt package lives
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -c "import niftyhedge, torch; print('ok', torch.__version__)"
```

### 1.2 Laptop: test the NSE UDiFF URL on **one** known trading day

Candidate pattern (unverified — fix if 404):

```text
https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
```

Example (pick a weekday you know NIFTY traded, e.g. 2024-08-01):

```bash
python scripts/00_download.py --config configs/data.yaml \
  --start 2024-08-01 --end 2024-08-01 --probe
```

- If status = `ok` → proceed to full download.
- If `404` → open NSE “All Reports, Derivatives” in a browser, copy the real F&O UDiFF link for that date, put the pattern in `configs/data.yaml`, retry. **Do not continue with partial data.**

### 1.3 Laptop: full download (Gate 1a)

```bash
python scripts/00_download.py --config configs/data.yaml \
  --start 2018-01-02 --end 2026-03-31
python scripts/00_download.py --config configs/data.yaml --retry-errors
```

Outputs under `data/raw/`:
- `nifty_options_raw.parquet`
- `nifty_spot.parquet`, `nifty_futures.parquet`, `india_vix.parquet`
- `download_log.csv` (one row per weekday: ok / holiday / error)

**Gate 1a (must pass before upload):**
- Zero `error` rows after `--retry-errors`.
- Count of `ok` days after 2024-07-08 > 400.
- Max trade date is in March 2026.

Record SHA256:

```bash
shasum -a 256 data/raw/nifty_options_raw.parquet >> data/raw/CHECKSUMS.txt
```

### 1.4 Upload to Google Drive

Create this Drive folder layout (exact names help the Colab notebook):

```text
MyDrive/Vol-Surfaces/
  repo/          # git clone or zip of this repository
  data/raw/      # the parquet files + download_log.csv + CHECKSUMS.txt
  runs/          # empty; Colab writes here
```

Zip tip if the UI is slow:

```bash
cd ~/Desktop/Vol-Surfaces
zip -r niftyhedge_repo.zip . -x '.venv/*' -x '.git/*' -x 'data/raw/*'
# upload zip + data/raw separately
```

---

## 2. Colab session bootstrap (every new runtime)

Open `notebooks/colab_run.ipynb` **or** run these cells in a fresh Colab notebook.

### Cell A — Runtime

1. Runtime → Change runtime type → **GPU** (T4 is fine).
2. Confirm:

```python
import torch
print(torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

### Cell B — Mount Drive and set paths

```python
from google.colab import drive
drive.mount("/content/drive")

import os
ROOT = "/content/drive/MyDrive/Vol-Surfaces"
REPO = f"{ROOT}/repo"          # or f"{ROOT}/Vol-Surfaces" if you unzipped there
DATA_RAW = f"{ROOT}/data/raw"
RUN = f"{ROOT}/runs/" + __import__("datetime").datetime.now().strftime("%Y%m%d")
os.makedirs(RUN, exist_ok=True)
os.environ["NH_ROOT"] = REPO
os.environ["NH_RUN"] = RUN
os.environ["NH_DATA_RAW"] = DATA_RAW
%cd {REPO}
!ls -la
!ls -la {DATA_RAW} | head
```

### Cell C — Install package

```python
!pip install -U pip
!pip install -e ".[dev]"
!python -c "import niftyhedge; print('niftyhedge OK')"
```

### Cell D — Symlink raw data into the repo (so scripts find it)

```python
import os
dst = f"{REPO}/data/raw"
os.makedirs(f"{REPO}/data", exist_ok=True)
if not os.path.islink(dst) and not os.path.exists(dst):
    os.symlink(DATA_RAW, dst)
elif os.path.isdir(dst) and not os.listdir(dst):
    os.rmdir(dst)
    os.symlink(DATA_RAW, dst)
print("data/raw ->", os.path.realpath(f"{REPO}/data/raw"))
```

### Cell E — Gate 0: build verification (CPU-ok, ~few minutes)

```bash
pytest -q
python scripts/check_build.py
make smoke
```

**Pass:** all tests green; `check_build.py` PASS for A1–A13; `make smoke` exits 0.

### Cell F — Legacy audit (documents the broken baseline)

```bash
python legacy/audit_env.py . 300 | tee $NH_RUN/legacy_audit.txt
```

Expected Normal regime (2026-11-13 audit):

| Strategy | Metric (less negative = better) |
|---|---|
| BS +delta (paper baseline) | ≈ -0.0209 |
| No hedge | ≈ -0.0122 |
| DDPM-RL (`best_agent_v2.pt`) | ≈ -0.0087 |
| BS -delta (sign corrected) | ≈ -0.0078 |

If your numbers disagree by more than ~5%, the checkpoints or scenario files are not the ones audited — stop and check `models/rl/` and `data/scenarios/`.

---

## 3. Stage 1 — Surfaces and folds (CPU on Colab)

```bash
python scripts/01_build_surfaces.py --config configs/data.yaml
python scripts/02_make_folds.py     --config configs/folds.yaml
```

Outputs:
- `data/surfaces/surfaces_v2.npy`, `surface_mask_v2.npy`, `surface_dates_v2.npy`, `forwards_v2.parquet`
- `tables/dataset_card.csv`
- `data/folds/fold_{0..5}/...`, `data/folds/fold_manifest.json`

**Gate 1b** (read `tables/dataset_card.csv`):
- Last surface date ≥ 2026-03-01.
- No NaN in `surfaces_v2.npy`.
- Calendar + Durrleman-butterfly violations in under 5% of surfaces.
- Mean ATM 30d IV by year is plausible (2020 highest; calm years in the low teens).
- Extrapolated-cell fraction per grid column is reported. If 7d or 180d is >40% extrapolated, drop that column in `configs/data.yaml` and rebuild.

```python
import pandas as pd
print(pd.read_csv("tables/dataset_card.csv").to_string())
```

---

## 4. Stage 2 — Generators (GPU)

```bash
# Fit W0..W4
python scripts/10_fit_generators.py --config configs/gen/gbm.yaml       --folds all --seeds 0
python scripts/10_fit_generators.py --config configs/gen/heston.yaml    --folds all --seeds 0
python scripts/10_fit_generators.py --config configs/gen/pcavar.yaml    --folds all --seeds 0
python scripts/10_fit_generators.py --config configs/gen/bootstrap.yaml --folds all --seeds 0
python scripts/10_fit_generators.py --config configs/gen/ddpm.yaml      --folds all --seeds 0,1,2
python scripts/10_fit_generators.py --config configs/gen/ddpm.yaml      --folds 5   --seeds 3,4

# 30-day joint paths
python scripts/11_generate_paths.py --worlds gbm,heston,pcavar,bootstrap,ddpm --folds all \
    --n-train 200000 --n-test 100000

# Generator quality (E5)
python scripts/12_eval_generators.py --config configs/experiments/e5_gen_quality.yaml
```

**Gate 2** (`tables/gen_quality.csv`, `tables/world_stats.csv`):
- DDPM one-step CRPS on validation < persistence. Else fix generator before any hedger.
- DDPM mean ATM 30d within 1 vol point of real for the same start dates.
- Conditioning dose-response: generated ATM rises with lagged VIX.
- 90% predictive interval coverage for ATM 30d between 80% and 97%.
- No world has NaN or non-positive vols in its paths.

Optional ablations (background / next session):

```bash
for cfg in configs/gen/ablations/*.yaml; do
  python scripts/10_fit_generators.py --config $cfg --folds 5 --seeds 0,1,2,3,4
done
python scripts/12_eval_generators.py --config configs/experiments/e5_gen_quality.yaml --ablations
```

**Colab tip:** keep Drive sync calm — write heavy tensors under `$NH_RUN` (local `/content` if Drive is slow), then copy finished `models/` and `tables/` to Drive at stage end.

```bash
# optional: fast local scratch, then sync
export NH_SCRATCH=/content/nh_scratch
mkdir -p $NH_SCRATCH
# pass --out-dir $NH_SCRATCH to scripts if supported; else rsync at the end
```

---

## 5. Stage 3 — Hedgers (GPU)

### Gate 3 first (sanity world, a few minutes)

```bash
python scripts/20_train_hedgers.py --config configs/hedge/deep_hedge.yaml \
    --world gbm --fold 5 --instruments I1 --contract configs/contracts/doc_090.yaml \
    --cost-bps 0 --seeds 0
python scripts/check_build.py --only A14
```

**Pass:** deep hedger CVaR95 within 5% of BS-BGK delta in W0 at zero cost, **and** action std across states > 0.05.  
If this fails, nothing downstream is meaningful.

### Full training

```bash
for W in gbm heston pcavar bootstrap ddpm; do
  for I in I1 I2 I3; do
    python scripts/20_train_hedgers.py --config configs/hedge/deep_hedge.yaml \
        --world $W --fold 5 --instruments $I --contract configs/contracts/doc_090.yaml \
        --seeds 0,1,2,3,4
    python scripts/20_train_hedgers.py --config configs/hedge/deep_hedge.yaml \
        --world $W --folds 0,1,2,3,4 --instruments $I --contract configs/contracts/doc_090.yaml \
        --seeds 0,1,2
  done
done

# PPO ablation (main fold, underlying only)
python scripts/20_train_hedgers.py --config configs/hedge/ppo.yaml \
    --world ddpm --fold 5 --instruments I1 --contract configs/contracts/doc_090.yaml \
    --seeds 0,1,2,3,4
```

Each run writes `models/hedge/{world}/{instr}/{contract}/fold_{k}/seed_{s}/` with `policy.pt`, `train_log.csv`, `manifest.json`.

Per-run check: `train_log.csv` validation CVaR is better at the selected checkpoint than at step 0, and the selected step is not the first or last evaluation.

**MVP shortcut (workshop paper):** train only worlds `gbm,heston,pcavar,ddpm` and instruments `I1,I2` on fold 5 with 5 seeds + other folds with 3 seeds. Skip I3 and W3 until Gate 4 looks sane.

---

## 6. Stage 4 — Evaluation (GPU / CPU)

```bash
python scripts/21_cross_world_eval.py --config configs/experiments/e7_cross_world.yaml
python scripts/22_real_backtest.py     --config configs/experiments/e8_real_backtest.yaml
python scripts/30_sensitivity.py       --config configs/experiments/e10_sensitivity.yaml
python scripts/40_make_tables_figures.py --run $NH_RUN
```

Outputs:
- `tables/cross_world_matrix_{I1,I2,I3}.csv`
- `tables/real_backtest_{contract}.csv`
- `tables/sensitivity_{cost,barrier,maturity,risk}.csv`
- `figures/*.pdf`
- `results/paths_pnl/*.parquet`

**Gate 4** (sanity on results, not on direction):
- In every test world, BS-BGK delta beats no hedge.
- In W0 at zero cost, no learned hedger beats BS-BGK by more than Monte Carlo error (else look-ahead in policy inputs).
- Seed std reported next to every learned-hedger number.
- Real backtest table states # start dates, # non-overlapping windows, # knock-outs per contract.

---

## 7. Colab survival checklist

| Risk | Mitigation |
|---|---|
| Runtime disconnects after ~12 h (free) / idle | Save checkpoints often (`manifest.json` skip-if-done). Re-run the same command; it resumes. Use Colab Pro for Stages 2–3. |
| Drive I/O is slow | Train/write under `/content/nh_scratch`, `rsync -a` to Drive at stage end. |
| GPU not attached | Runtime → Change runtime type → GPU. Re-run Cell A. |
| Out of disk on `/content` | Delete old scratch dirs; keep only current `$NH_RUN`. |
| NSE download from Colab | **Do not.** Always laptop → Drive. |
| Session dies mid-loop | Scripts must skip completed `manifest.json`. Use `--force` only when intentional. |

Resume after reconnect:

```python
# remount + reinstall (Cells B–D), then:
!ls $NH_RUN
# re-issue the last unfinished command; completed seeds are skipped
```

---

## 8. What to keep / send for review

**After the build (before heavy GPU):**
- Repo pushed or zipped.
- Output of `pytest -q`, `python scripts/check_build.py`, `make smoke`.
- `legacy_audit.txt`.

**After the GPU run:**
- `tables/*.csv`, `figures/*.pdf`
- `$NH_RUN/**/manifest.json`
- `data/raw/download_log.csv`, `tables/dataset_card.csv`
- `data/raw/CHECKSUMS.txt`

---

## 9. Ordered checklist (print and tick)

```text
[ ] Laptop: probe one UDiFF date
[ ] Laptop: full download 2018-01-02 → 2026-03-31 + retry-errors
[ ] Gate 1a: download_log clean, >400 ok days post-2024-07-08, max date Mar 2026
[ ] Upload repo + data/raw to Drive
[ ] Colab: GPU runtime, mount, pip install -e ".[dev]"
[ ] Gate 0: pytest + check_build + make smoke
[ ] Legacy audit numbers match Part 1
[ ] Stage 1: surfaces + folds
[ ] Gate 1b: dataset_card OK
[ ] Stage 2: fit generators + paths + gen_quality
[ ] Gate 2: CRPS / ATM / coverage OK
[ ] Gate 3: W0 deep hedger ≈ BS-BGK, action std > 0.05
[ ] Stage 3: train deep hedgers (+ PPO ablation)
[ ] Stage 4: cross-world + real backtest + tables
[ ] Gate 4: sanity checks
[ ] Archive tables/figures/manifests
```

---

## 10. Reproducibility rules

- Every script skips work whose `manifest.json` already exists with a matching config hash. Use `--force` to redo.
- Generator seeds and hedger seeds are independent. A hedger run records which generator seed produced its training paths.
- Keep `nifty_options_raw.parquet` out of git; store SHA256 in `data/raw/CHECKSUMS.txt`.
- Do not hand-edit anything in `tables/`. All paper numbers must trace to a script and a manifest.
- Package code under `src/` must never import `google.colab` (acceptance A10).
