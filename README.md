# Diffusion-Generated Volatility Surfaces for Deep Hedging of NIFTY 50 Barrier Options

> Model risk decomposition: how much of a deep hedger's real-world performance
> comes from the learning algorithm vs the market generator it trains on?

**Status (2026-11-13).** The original headline numbers (60.1% / 53.2% / 14.9%) do
**not** hold. They come from a hedge-sign bug and a flat-vol simulator in which
the surface never enters P&L. See [`EXPERIMENTS_PLAN.md`](EXPERIMENTS_PLAN.md)
Part 1 and reproduce with:

```bash
python legacy/audit_env.py . 300
```

| Strategy (Normal, 300 eps, legacy metric) | Score (less negative = better) |
| ----------------------------------------- | ------------------------------ |
| BS +delta (paper baseline)                | ≈ −0.0209                      |
| No hedge                                  | ≈ −0.0122                      |
| DDPM-RL (`best_agent_v2.pt`)              | ≈ −0.0087                      |
| BS −delta (sign corrected)                | ≈ −0.0078                      |

The rebuilt evaluation (SVI surfaces through Mar 2026, leak-free features,
corrected terminal-loss CVaR env, five generators, walk-forward real backtest)
is specified in [`EXPERIMENTS_PLAN.md`](EXPERIMENTS_PLAN.md). Execute it with
[`RUNBOOK.md`](RUNBOOK.md) / [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb).

**Author.** Adya Srivastava, BITS Pilani (`f20220467@pilani.bits-pilani.ac.in`)

---

## Start here

| Document | Role |
| -------- | ---- |
| [`EXPERIMENTS_PLAN.md`](EXPERIMENTS_PLAN.md) | Audit + reframed question + experiments + build contracts + acceptance checklist |
| [`RUNBOOK.md`](RUNBOOK.md) | Laptop download → Drive → Colab GPU stages with stop gates |
| [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) | Thin Colab wrapper for the runbook |
| [`legacy/audit_env.py`](legacy/audit_env.py) | CPU reproduction of the invalidated paper metric |
| [`legacy/NiftyIV_Scaled.ipynb`](legacy/NiftyIV_Scaled.ipynb) | Frozen original notebook |

---

## Colab quick path (after the package is built)

1. **Laptop (India):** probe one NSE UDiFF date, then download 2018-01-02 → 2026-03-31. Pass Gate 1a.
2. Upload repo + `data/raw/` to `MyDrive/Vol-Surfaces/`.
3. Open `notebooks/colab_run.ipynb` with a **GPU** runtime.
4. Run bootstrap → Gate 0 → Stage 1 → Gate 1b → Stage 2 → Gate 2 → Gate 3 → Stage 3 → Stage 4 → Gate 4.
5. Budget: roughly **15–25 GPU-hours** on one T4. A negative/mixed result is still publishable.

Details and exact commands: [`RUNBOOK.md`](RUNBOOK.md).

---

## Repository layout (target after rebuild)

```
Vol-Surfaces/
├── EXPERIMENTS_PLAN.md
├── RUNBOOK.md
├── notebooks/colab_run.ipynb
├── src/niftyhedge/          # package (to be built per Part 4 of the plan)
├── scripts/                 # CLI entrypoints
├── configs/
├── tests/
├── legacy/                  # frozen notebook + audit_env.py
├── data/                    # raw / surfaces / folds / paths
├── models/
├── results/
├── tables/
└── figures/
```

Legacy artefacts from the prototype (checkpoints, scenarios, paper PDF) remain
under `data/`, `models/`, `results/`, `figures/`, `paper/` for regression tests.

---

## Installation (rebuilt package)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
python scripts/check_build.py
```

Until `pyproject.toml` / `src/niftyhedge` exist, only the legacy audit path works:

```bash
pip install -r requirements.txt
python legacy/audit_env.py . 300
```

---

## Data sources

- **NIFTY options.** NSE Bhavcopy (legacy CSV pre–Jul 2024; UDiFF after). Download
  from a laptop in India — cloud IPs are often blocked.
- **NIFTY spot / futures / India VIX.** Alongside the options pull (see plan E1).

No proprietary data is required.

---

## Citation

```bibtex
@article{srivastava2026volsurfaces,
  title   = {Diffusion-Generated Volatility Surfaces for Deep Hedging
             of NIFTY 50 Barrier Options: A Model Risk Decomposition},
  author  = {Srivastava, Adya},
  year    = {2026},
  note    = {Working paper; evaluation rebuilt after 2026-11-13 audit}
}
```
