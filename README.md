# Diffusion-Generated Volatility Surfaces for RL Hedging of NIFTY 50 Barrier Options

> A model risk decomposition: how much of an RL hedger's improvement comes from the
> learning algorithm vs. the quality of the volatility surfaces it trains on?

This repository accompanies the paper
**"Diffusion-Generated Volatility Surfaces for Reinforcement Learning Hedging of
NIFTY 50 Barrier Options: A Model Risk Decomposition"**
([`paper/Volatility_Surfaces_Research.pdf`](paper/Volatility_Surfaces_Research.pdf))
submitted to NeurIPS 2026.

**Author.** Adya Srivastava, BITS Pilani, Pilani Campus
(`f20220467@pilani.bits-pilani.ac.in`)

---

## Headline results

A PPO-trained reinforcement learning hedger that observes DDPM-generated NIFTY IV
surfaces reduces CVaR(95%) of barrier call hedging error by **60.1%** over
Black-Scholes delta hedging. The improvement decomposes as:

| Source                                          | Contribution |
| ----------------------------------------------- | ------------ |
| RL algorithm (Heston-RL vs. BS-δ)               | **+53.2%**   |
| DDPM surface quality (DDPM-RL vs. Heston-RL)    | **+14.9%**   |
| **Total (DDPM-RL vs. BS-δ)**                    | **+60.1%**   |

All comparisons significant at p < 0.0001, 1,000 bootstrap episodes per cell,
across Normal / Stressed / Event regimes. See Table 4 of the paper.

---

## Repository layout

```
Vol-Surfaces/
├── NiftyIV_Scaled.ipynb            # End-to-end research notebook (Colab)
├── README.md
├── requirements.txt
├── .gitignore
│
├── paper/
│   └── Volatility_Surfaces_Research.pdf
│
├── data/
│   ├── raw/                        # NSE-derived surfaces and spot
│   │   ├── nifty_spot.parquet      # 2018-01-02 to 2026-03-31 spot
│   │   ├── surfaces.npy            # 1,268 daily 9×6 IV surfaces
│   │   └── surface_dates.npy
│   ├── processed/                  # DDPM training tensors
│   │   ├── X_train.npy, X_val.npy, X_test.npy   # log-normalised surfaces
│   │   ├── C_train.npy, C_val.npy, C_test.npy   # 112-d conditioning
│   │   ├── log_mean.npy, log_std.npy            # normalisation stats
│   │   └── (chronological split: 2018-21 / 2022 / 2023-26)
│   └── scenarios/                  # Pre-generated RL evaluation surfaces
│       ├── scenarios_normal.npy
│       ├── scenarios_stressed.npy
│       ├── scenarios_event.npy
│       ├── scenarios_oos_test.npy  # 367 OOS surfaces (2023-26)
│       ├── heston_normal.npy
│       ├── heston_stressed.npy
│       └── heston_event.npy
│
├── models/
│   ├── ddpm/
│   │   ├── best_model.pt           # Phase 1 baseline (val loss 0.2630)
│   │   ├── best_model_phase1b.pt   # Phase 1b weighted resampling (0.2266)
│   │   └── best_model_final.pt     # Final DDPM (= Phase 1b)
│   └── rl/
│       ├── best_agent.pt           # Pre-normalisation PPO agent
│       ├── best_agent_v2.pt        # Final DDPM-RL agent (normalised rewards)
│       └── best_agent_heston.pt    # Heston-trained PPO baseline
│
├── results/
│   ├── final_results.pkl           # Three-way CVaR + bootstrap CIs
│   └── oos_results.pkl             # Out-of-sample CVaR for BS / Heston / DDPM
│
└── figures/                        # All paper figures + intermediates
    ├── data_analysis/              # Section 3 — Data and Empirical Analysis
    │   ├── nifty_vol_timeseries.png      (Paper Fig. 1)
    │   ├── surface_low_vol.png           (Fig. 6a)
    │   ├── surface_typical.png           (Fig. 6b)
    │   ├── surface_high_vol.png          (Fig. 6c)
    │   ├── skew_analysis.png             (Fig. 7)
    │   └── arbitrage_violations.png      (Fig. 8)
    ├── ddpm/                       # Section 4 — DDPM Generator
    │   ├── training_curves.png           (Fig. 9)
    │   └── generation_quality.png        (Fig. 2)
    ├── rl/                         # Section 5 — RL Hedging Agent
    │   ├── training_curves.png           (Fig. 3 — V2 PPO diagnostics)
    │   └── hedge_analysis.png
    ├── evaluation/                 # Section 6 — Results
    │   ├── three_way_comparison.png      (Fig. 4)
    │   └── oos_evaluation.png            (Fig. 5)
    └── intermediate/               # Earlier training iterations (kept for record)
```

---

## Notebook walkthrough

`NiftyIV_Scaled.ipynb` is the single source of truth and runs end-to-end on Google
Colab (a free GPU runtime is sufficient). The cells are grouped into five phases:

| Cells   | Phase                                                           |
| ------- | --------------------------------------------------------------- |
| 0 – 8   | **Data.** NSE Bhavcopy download (dual format, pre/post Jul 2024), IV extraction in total-variance space, 9×6 surface construction, arbitrage diagnostics, dataset preparation |
| 9 – 14  | **DDPM Phase 1.** ResBlock MLP (574k params), cosine schedule (T=1000), SNR-weighted arbitrage penalty, 1,200 epochs |
| 15 – 20 | **DDPM Phase 1b.** Weighted resampling of high-vol days (3×/8×/20× by ATM bucket), 500 fine-tune epochs at lower lr |
| 21 – 31 | **RL hedging.** Down-and-out call (K=S₀, B=0.9S₀) environment, PPO agent (402k params), reward normalisation, 6,000 episodes, Heston-RL baseline trained with identical hyperparameters |
| 32 – 45 | **Evaluation.** Bootstrap CVaR CIs, three-way comparison, out-of-sample evaluation on held-out 2023-26 surfaces, paper figure generation |

Cells 16, 28, 29, 35, 36 are Colab-specific Drive sync utilities and can be
skipped if you are running locally (the artefacts they sync are already in this
repo under `data/`, `models/`, and `results/`).

---

## Reproducing key numbers

If you just want to verify the headline numbers from the saved artefacts:

```python
import pickle, numpy as np

with open('results/final_results.pkl', 'rb') as f:
    res = pickle.load(f)
# Per-regime normalised CVaR(95%) with 95% bootstrap CIs for BS-δ, Heston-RL, DDPM-RL

with open('results/oos_results.pkl', 'rb') as f:
    oos = pickle.load(f)
# Out-of-sample (2023-26) CVaR(95%) for all three methods
```

To re-train from scratch, open `NiftyIV_Scaled.ipynb` in Colab and run top to
bottom. End-to-end training takes ~3 hours on a T4 GPU (1.5h DDPM Phase 1 + 0.5h
Phase 1b + 1h PPO for DDPM-RL + 1h for the Heston-RL baseline).

---

## Installation

```bash
pip install -r requirements.txt
```

The notebook expects Python ≥ 3.10 and PyTorch ≥ 2.0 (GPU strongly recommended
for training; inference and evaluation run fine on CPU).

---

## Data sources

- **NIFTY options.** NSE Bhavcopy (publicly available). The notebook handles both
  the legacy CSV format (pre-July 2024) and the new UDiFF format (July 2024
  onward) automatically.
- **NIFTY spot.** Yahoo Finance (`^NSEI`) via `yfinance`.
- **Macro event calendar.** Hard-coded list of RBI MPC meetings, Union Budget
  dates, and general election result days (2018-2026), covering 13.2% of trading
  days.

No proprietary data is required.

---

## Known limitations

These are discussed in detail in Section 7 of the paper:

1. **ATM vol underestimation.** DDPM generates 12.9% mean ATM 30-day vol vs. real
   16.5% (21% underestimation). Phase 1b resampling reduces but does not close
   the gap.
2. **Heston calibration asymmetry.** The Heston baseline uses θ=0.04 (20% ATM),
   which exceeds DDPM's 12.9%. The 14.9% DDPM contribution is therefore a
   conservative lower bound.
3. **Simulation gap.** Spot dynamics use only the scalar ATM vol; the agent
   observes the full 54-d surface. A joint surface-spot diffusion is future work.
4. **OOS evaluation scope.** The 367 OOS surfaces are DDPM-generated from 2023-26
   conditioning vectors, not raw market prices.

---

## Citation

```bibtex
@article{srivastava2026volsurfaces,
  title   = {Diffusion-Generated Volatility Surfaces for Reinforcement Learning
             Hedging of NIFTY 50 Barrier Options: A Model Risk Decomposition},
  author  = {Srivastava, Adya},
  journal = {arXiv preprint (NeurIPS 2026 submission)},
  year    = {2026}
}
```
