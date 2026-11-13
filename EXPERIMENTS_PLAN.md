# EXPERIMENTS_PLAN.md

Paper working title: **"Diffusion-Generated Volatility Surfaces for Deep Hedging of NIFTY 50 Barrier Options: A Model Risk Decomposition"**

> Title decision (locked): pathwise deep hedging is the main agent; PPO is an ablation. "Reinforcement Learning" in the title is replaced by "Deep Hedging".

Repo audited: `github.com/Adya6714/ddpm-vol-hedging` at commit `6b4a3aa` (single notebook now at `legacy/NiftyIV_Scaled.ipynb`)  
Audit date: 2026-11-13  
Primary compute target: **Google Colab (T4 / L4)**. Data download stays on a laptop in India (NSE blocks many cloud IPs).

How to use this file:
- Part 1 — audit: what the current code actually does, with evidence.
- Part 2 — reframed research question.
- Part 3 — experiment list (what, why, output, pass gate).
- Part 4 — build spec (modules, function contracts, tests).
- Part 5 — build acceptance checklist.
- `RUNBOOK.md` — Colab cell-by-cell / command-by-command execution order with stop gates.
- `legacy/audit_env.py` — reproduces the invalidated paper metric on CPU.

---

## Part 1. Current stage: audit findings

Summary in one line: the pipeline runs end to end and the paper is well organised, but the headline numbers (60.1% / 53.2% / 14.9%) come from an environment bug and do not survive a corrected baseline. The work is at "prototype with invalid evaluation" stage, not "needs a few more tables" stage. Everything below is fixable, and the fixed version is a more interesting paper.

All audit numbers were reproduced on CPU from the saved checkpoints and scenario files using `legacy/audit_env.py` (the notebook's V2 environment and metric copied verbatim, 300 episodes per regime, common random numbers).

### 1.1 Blocking findings (invalidate the headline result)

| # | Finding | Evidence |
|---|---|---|
| F1 | Hedge sign bug. The env computes `hedge_err = dV + hedge * dS`. The BS baseline passes `hedge = +delta`. For a call, `dV` and `+delta*dS` have the same sign, so the "BS delta hedge" doubles the exposure instead of cancelling it. | Notebook cells 24 and 30 (`NiftyBarrierEnv.step`, `NiftyBarrierEnvV2.step`). Audit, Normal regime: BS +delta (paper baseline) = -0.0209, no hedge at all = -0.0122, BS with corrected sign (-delta) = -0.0078. Doing nothing beats the paper's baseline by 41%. |
| F2 | The corrected BS delta beats DDPM-RL. | Audit: sign-corrected BS delta -0.0078 / -0.0076 / -0.0075 vs DDPM-RL -0.0087 / -0.0084 / -0.0083 (Normal / Stressed / Event). RL is about 10% worse than the classical hedge, not 60% better. |
| F3 | Both trained agents are constants, not policies. `best_agent_v2.pt` outputs hedge = -0.577 with std 0.0045 across all states. `best_agent_heston.pt` outputs -0.256 with std 0.019. Correlation of action with true delta is about 0.1. | Audit. A hard-coded constant hedge of -0.5 scores -0.0088, same as DDPM-RL. So the "14.9% DDPM surface quality contribution" is the difference between two constants (-0.577 vs -0.256), not anything about surfaces. |
| F4 | The simulator makes surface quality irrelevant by construction. Spot is GBM driven by the ATM 30d vol scalar, and the option is marked with the flat-vol BS barrier formula at that same scalar. Skew, wings and term structure never touch the P&L. | Cells 24 and 30. The research question (does generator quality matter) cannot be answered in this environment. |
| F5 | The metric is not CVaR of hedging error. It is the mean of the worst 5% of the 30 daily P&Ls inside one episode (the worst 1 to 2 days), averaged over episodes. The bootstrap CI is a CI on that mean. The standard metric is CVaR of terminal hedging loss across paths. | `get_episode_cvar`, cell 38 `collect_normalised_cvars`. |
| F6 | Checkpoint selection is noise. Best checkpoint is chosen on a 50-episode validation estimate. Training logs show validation getting worse with training for the V1 agent and the Heston agent ("Best episode: 1" in both logs). PPO updates use one 30-step rollout per update. | Cells 26 and 40 outputs. Heston val metric goes from -195 at episode 1 to about -640 to -690 for the rest of training. The saved Heston checkpoint outputs a constant -0.256, so which training iteration it corresponds to cannot be established from the repo. |
| F7 | Heston-RL was not trained identically. 5,000 episodes vs 6,000 for DDPM-RL (the restore cell resets `N_EPISODES_TRAIN = 5000`). | Cell 36, cell 40 log ends at 5000. |

### 1.2 Data findings

| # | Finding | Evidence |
|---|---|---|
| D1 | Data ends 2024-07-05, not 2026-03-31. The UDiFF branch of the downloader failed silently (bare `except: pass`, wrong URL pattern). | Cell 2 output: "Date range: 2018-01-02 to 2024-07-05, Unique days: 1604". `surface_dates.npy` max = 2024-07-05. The paper says 2018 to 2026 in title-level claims, abstract, Table 1 and split description. |
| D2 | Counts differ between paper and repo. Paper: 1,268 surfaces, train 653, test 366. Repo: 1,261 surfaces, train 645, test 367. Test set covers Jan 2023 to Jul 2024 only. | `data/raw/surfaces.npy`, `data/processed/X_*.npy`. |
| D3 | 18.7% of grid cells are missing and are filled with that surface's own mean. 85.7% of surfaces have at least one filled cell. 7d column is 41% missing, 180d is 34% missing. ATM 30d itself is missing on 14.3% of days. | `surfaces.npy` NaN counts; `prepare_dataset` in cell 8. Mean-filling creates flat fake patches and fake arbitrage. |
| D4 | IV extraction uses spot (no dividend or futures basis), mixes calls and puts at the same strike, and extrapolates linearly in moneyness. | Cells 4 and 5. Standard practice: implied forward from put-call parity or the futures price, OTM options only, smoothing by SVI. |
| D5 | The butterfly test is not a butterfly arbitrage test. It checks convexity of total variance in moneyness. The correct static conditions are call-price convexity in strike (Durrleman's g(k) >= 0) and total variance non-decreasing in maturity at fixed forward log-moneyness. The "99.8% of days violate butterfly" statistic is mostly an artifact of D3 and D5. | Cell 6 `check_arbitrage`. |

### 1.3 Generator findings

| # | Finding | Evidence |
|---|---|---|
| G1 | Target leakage in conditioning. `ewma5[t] = 0.4 * X[t] + 0.6 * ewma5[t-1]`, so the conditioning vector contains 40% of the surface the model is asked to generate. Same-day return, squared return and India VIX are also in the conditioning. | Cell 8. Mean per-dimension correlation between `X_train` and the EWMA5 block of `C_train` is 0.92. |
| G2 | Normalisation leakage. `log_mean` / `log_std` are computed over train + val + test. | Cell 8. |
| G3 | Event flag uses hindsight. The list includes the three COVID crash days and a Delhi assembly election. The paper says RBI MPC, Union Budget, general elections only. | Cell 8 `EVENT_DATES`. |
| G4 | Regime conditioning has no effect. Stressed scenarios have the same ATM vol as Normal (12.4% vs 12.4%). The notebook's own acceptance check reports 0 of 4 targets met, then proceeds. | Cell 19 output: "Stressed minus Normal ATM: -0.7%", "Targets met: 0/4". `scenarios_*.npy` ATM means 12.38 / 12.43 / 12.22. |
| G5 | The arbitrage claim in the paper is not supported. Paper: generated penalty 0.005 below market 0.009. Notebook, same metric on both: generated calendar 0.00898 vs market 0.00615, generated butterfly 0.0264 vs market 0.0122, and 99 to 100% of generated surfaces have violations. | Cells 14 and 19 outputs. The 0.005 figure is the per-cell training-loss average, a different quantity. |
| G6 | Table 2 skew numbers (2.30% normal, 1.85% stressed) do not match the saved scenarios (3.01%, 3.69%). | Cell 20 output. |
| G7 | 21% ATM level bias (12.4% generated vs 16.3% real). Likely causes: mean-filled inputs, clamping inside the sampler (`clamp(-4, 2)` on x0), no EMA weights, modelling levels instead of day-over-day changes. | Cells 14, 20. |
| G8 | Validation loss is computed with fresh random timesteps and noise each epoch, so "best epoch" is partly noise. | Cell 13. |

### 1.4 Baseline and evaluation findings

| # | Finding | Evidence |
|---|---|---|
| B1 | Heston baseline is uncalibrated (textbook parameters), static (independent perturbed surfaces, not a simulated path), and 31 to 56% of its cells are NaN. The 30d 0.95-moneyness cell is NaN in 100% of Normal and Event surfaces, so the agent's skew features are the 0.16 fallback constant. | `heston_*.npy`; cell 39 output "Skew=nan%". |
| B2 | ATM level is confounded with "surface quality": Heston 20 to 34% vs DDPM 12%. | Paper section 5.3 acknowledges this; it is a confound, not a conservative bound. |
| B3 | "Out of sample" evaluation is the same DDPM with test-period conditioning. No real-market test exists. | Cell 9, paper section 6.3. |
| B4 | Single seed for every model. No ablations. No sensitivity to transaction cost, barrier level, maturity. No classical baselines beyond one delta hedge. | Whole notebook. |
| B5 | Barrier is monitored daily but priced with the continuous-monitoring formula. T uses days/252 while the surface grid is calendar days. Reward mixes squared rupee error with linear rupee cost. | Cells 23, 30. |

### 1.5 Paper / README text that must change regardless of experiments

- Date range, surface counts, split sizes (D1, D2).
- Remove or recompute the arbitrage claim (G5) and Table 2 (G6).
- Remove "Submitted to NeurIPS 2026" from README and paper footer. The checklist cites "Appendix A hyperparameter tables" that do not exist.
- Do not claim the 60.1% / 53.2% / 14.9% numbers. Point readers to Part 1 and `legacy/audit_env.py`.
- 7 references is far too few. Minimum set to add and position against: Buehler et al. 2019 (deep hedging), Wiese et al. 2019 "Learning to Simulate Equity Option Markets" (arXiv 1911.01700), Buehler et al. 2021 "Learning Risk-Neutral Implied Volatility Dynamics" (arXiv 2103.11948), Cao et al. 2021 (RL hedging), Cont and Vuletic (arbitrage-free IV simulation, VolGAN), Jin and Agarwal 2025 (arXiv 2511.07571), Francois et al. 2025 (arXiv 2504.06208), Horvath, Teichmann, Zuric 2021 (deep hedging under rough vol), Lutkebohmert, Schmidt, Sester (robust deep hedging under model uncertainty), Limmer and Horvath (robust hedging GANs), Carr and Chou / Derman, Ergener, Kani (static replication of barriers), Broadie, Glasserman, Kou 1997 (discrete barrier correction), Gatheral and Jacquier 2014 (SVI), recent VAE and diffusion IV surface papers (arXiv 2509.01743, 2609.00332).
- Every "first to our knowledge" claim needs a real literature search behind it.

---

## Part 2. The reframed research question

Keep the core idea. Change how it is tested.

- Question: when a deep hedger for a NIFTY barrier option is trained in a simulated market, how much of its real-world performance depends on (a) the learning algorithm vs classical hedges, and (b) which market generator produced the training paths?
- Why the current setup cannot answer it: F4. The surface has to enter the P&L.
- Two changes that make the surface matter:
  1. Terminal P&L accounting on joint (spot, surface) paths. No BS marking inside the env. Loss = payoff minus hedge gains plus costs.
  2. Vanilla options priced off the surface are available as hedging instruments (spot only, spot + ATM option, spot + ATM option + barrier-strike put). This is the Francois et al. setup and it is where skew dynamics drive results.
- The decisive test is on real NIFTY paths and real surfaces (walk-forward), not on generated ones.
- A negative or mixed result ("a calibrated Heston or a PCA-VAR trains as good a hedger as the diffusion model", or "deep hedging only wins with transaction costs") is still publishable if the evaluation is rigorous. Do not optimise for a positive number.

---

## Part 3. Experiments

Priority: P0 = blocking for any venue. P1 = core paper. P2 = strengthens, needed for a top venue.

### E0. Freeze and regression-test the legacy result (P0)
- What: keep the notebook in `legacy/`, keep `legacy/audit_env.py`, add a test that reproduces the paper's Table 3 numbers within 5% and the audit table in Part 1.
- Why: documents exactly what changed and proves the new env is different for the stated reasons.
- Output: `results/legacy_audit.json`.

### E1. Rebuild the dataset through 2026-03-31 (P0)
- What:
  - Fix the UDiFF downloader for dates from 2024-07-08. Candidate pattern to verify on one date first: `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip`. Fail loudly: log every date as ok / holiday / error, never bare `except`.
  - **Download from a laptop in India.** NSE often blocks Colab / cloud IPs. Test the URL on one known trading day before a full sweep.
  - Save raw option rows (`nifty_options_raw.parquet`). The current repo does not have them, which makes the dataset non-reproducible.
  - Forward per expiry from put-call parity on the 3 strikes nearest ATM (fallback: NIFTY futures settlement from the same bhavcopy).
  - OTM only: puts for K < F, calls for K >= F. Filters: volume >= 50 contracts, price >= 1, time to expiry >= 3 calendar days, no-arbitrage price bounds.
  - Per expiry slice: fit raw SVI in total variance with butterfly constraint, reject fits with RMSE above threshold. Interpolate across maturity linearly in total variance at fixed log-forward-moneyness.
  - Grid stays 9 x 6. Store a boolean mask of cells supported by data (within the traded strike and maturity range) vs extrapolated.
  - Download India VIX and NIFTY futures series alongside.
- Why: D1 to D5. Also extends the test period by about 21 months, including the 2025 drawdown.
- Outputs: `data/raw/nifty_options_raw.parquet`, `data/surfaces/surfaces_v2.npy`, `surface_mask_v2.npy`, `surface_dates_v2.npy`, `forwards_v2.parquet`, `tables/dataset_card.csv` (days downloaded, days with surface, coverage by year and by grid cell, SVI fit RMSE distribution).
- Gates: last date >= 2026-03-01. Extrapolated-cell fraction reported per column. Zero NaN in the final grid. Butterfly (Durrleman) and calendar violation rates on fitted surfaces below 5% of surfaces.

### E2. Leak-free features and walk-forward splits (P0)
- What:
  - Conditioning for day t uses only information up to t-1: previous surface, EWMA of past surfaces, past returns, previous VIX close. Event flag for day t is allowed only from a pre-announced calendar (RBI MPC dates, Union Budget, general election counting days, optionally monthly expiry days). No COVID dates.
  - Normalisation statistics from the training fold only.
  - Walk-forward folds: train up to end of year Y, validate on last 6 months of training span, test on year Y+1. Folds: test 2020 (COVID), 2021, 2022, 2023, 2024, 2025 to Mar 2026.
- Why: G1 to G3. The 2020 fold gives real barrier knock-outs, which 2023 to 2026 alone barely has.
- Output: `data/folds/fold_{k}/{X,C,dates}_{train,val,test}.npy`, `fold_manifest.json`.
- Gate: a unit test asserting no feature at row t depends on data dated >= t (perturb day t inputs, check row t conditioning is unchanged).

### E3. Correct hedging environment and metric (P0)
- What:
  - Convention: the hedger is short one barrier option. Loss `L = payoff - p0 - sum_t h_t (S_{t+1} - S_t) - sum_t q_t . (O_{t+1} - O_t) + costs`. `h_t` is the underlying position (positive = long), `q_t` the vanilla option positions.
  - Fully vectorised in torch over paths. No per-step Python loop over episodes.
  - Discrete daily barrier monitoring. Payoff computed from the path only.
  - Vanilla instrument prices come from the path's surface at each step (interpolation in total variance), not from a flat vol.
  - Costs: proportional, separate rates for underlying and options. Default underlying 2 bps (NIFTY futures are far cheaper than the 10 bps used now), options 1% of premium. Both swept in E10.
  - `p0` = same number for every strategy on a given path set (BS barrier price at inception ATM IV with Broadie-Glasserman-Kou shift), so it cancels in comparisons.
  - Metrics on terminal `L` across paths: mean, std, VaR95, CVaR95, CVaR99, all as % of S0 and as % of p0.
- Why: F1, F4, F5, B5.
- Gates (unit tests):
  - GBM world, zero cost, vanilla call, 1-minute-equivalent fine steps: delta-hedge loss std goes to about 0 and halves when steps quadruple.
  - GBM world, daily steps: BS barrier delta CVaR95 at least 50% lower than no-hedge.
  - Sign test: in this convention the correct hedge for the short option is `h = +delta` (long the underlying). `h = +delta` must beat no hedge. `h = -delta` must be worse than no hedge.
  - Knocked-out paths: payoff is exactly 0 and hedge gains stop accruing only if the policy unwinds.

### E4. Market generators with a common interface (P1)
All generators produce joint 30-day paths `(S_0..S_H, surface_0..surface_H)` started from a real historical state (a conditioning date), so every world has the same initial ATM level distribution. This removes the B2 confound.

| World | Model | Role |
|---|---|---|
| W0 | GBM, constant vol = inception ATM IV | Sanity world. BS delta is near optimal here. |
| W1 | Heston, calibrated per fold to training-fold surfaces (global kappa, theta, xi, rho; v0 from each start day). Spot and variance simulated; surface each day from the COS pricer (or a precomputed interpolation table over v, tau, k). | Proper parametric baseline (replaces B1). |
| W2 | PCA-VAR: PCA on log-IV (3 to 5 factors) + log return, VAR(1) with bootstrapped residuals. | Cheap data-driven baseline. A diffusion model that cannot beat this is not adding value. |
| W3 | Stationary block bootstrap of real (return, surface change) sequences from the training fold. | Nonparametric baseline with real joint dynamics. |
| W4 | DDPM, joint one-step model: generates (log return_t, delta log-IV surface_t) given leak-free history, rolled out autoregressively for 30 steps. | The proposed generator. |
| W5 (P2) | VolGAN-style GAN or a conditional VAE with the same conditioning. | Generative baseline from the literature. |

- DDPM changes vs the notebook: model day-over-day change in log-IV plus the return (fixes level bias G7 and gives spot-vol correlation for free); x0 or v prediction; EMA weights; no clamping inside the sampler; fixed validation noise (G8); classifier-free guidance on the event flag.
- Outputs: `models/gen/{world}/fold_{k}/seed_{s}/`, `data/paths/{world}/fold_{k}/paths_{train,test}.pt`.
- Gates: each world's generated start-of-path ATM distribution equals the real one by construction; 30-day-ahead ATM vol distribution reported per world in `tables/world_stats.csv`.

### E5. Generator quality evaluation, independent of hedging (P1)
- Metrics, all on the held-out fold, all against real data:
  - One-step-ahead probabilistic forecast: CRPS per grid cell, energy score for the full surface, vs persistence baseline, PCA-VAR, Heston.
  - Calibration: coverage of 50% and 90% predictive intervals, PIT histogram for ATM 30d.
  - Marginals of ATM 30d, 30d skew, term slope, 30d butterfly: KS and Wasserstein-1 distances.
  - Dynamics: autocorrelation of ATM vol changes (lags 1 to 20), correlation of return with ATM change (leverage), vol-of-vol, PCA explained-variance spectrum of generated vs real surface changes.
  - Static arbitrage with the correct conditions (E1), fraction of surfaces and mean magnitude.
  - Conditioning dose-response: sweep lagged VIX and event flag, plot generated ATM and skew. This directly tests G4.
  - Memorisation: nearest-neighbour distance from generated surfaces to training set vs to held-out set.
- DDPM ablations (5 seeds each on the main fold, 3 on others): arbitrage penalty weight in {0, 0.05, 0.2, 1.0}; SNR weighting on/off; event flag on/off; conditioning set {none, lagged surface, + returns, + VIX, + event}; DDIM steps {10, 50, 250}; level vs change parameterisation; width {128, 256, 512}.
- Why: the paper's contribution 2 and 3 currently rest on one unconditional histogram and an unsupported arbitrage number.
- Outputs: `tables/gen_quality.csv`, `tables/gen_ablation.csv`, `figures/gen_*.pdf`.

### E6. Hedging agents and classical baselines (P1)
- Instruments sets: I1 underlying only. I2 underlying + ATM 30d call. I3 underlying + ATM call + put struck at the barrier.
- Classical baselines:
  - No hedge.
  - BS barrier delta at current ATM IV, continuous formula.
  - BS barrier delta with Broadie-Glasserman-Kou barrier shift for daily monitoring.
  - Smile-aware delta: barrier delta using IV at the barrier strike for the knock-out leg.
  - Transaction-cost band (Whalley-Wilmott) around the BGK delta.
  - I2: delta-vega hedge. I3: static replication (Carr-Chou / Derman-Ergener-Kani) plus delta residual.
- Learned agents:
  - Deep hedger (main): MLP policy, inputs log(S/K), log(S/B), time to expiry, previous positions, surface features, optional BS delta; trained by backprop through the vectorised env on generator paths; objective CVaR95 via the Rockafellar-Uryasev form `w + E[(L - w)^+] / (1 - alpha)`.
  - PPO agent (ablation): same inputs, vectorised envs (>= 4096 parallel), rollouts of full episodes, terminal reward. Reports whether model-free RL reaches the pathwise solution.
- Input ablations: no surface features / ATM only / handcrafted 10 features / PCA scores; with and without BS delta as an input; residual policy (BS delta + learned correction).
- Seeds: 5 per (world, instrument set) on the main fold.
- Gates: in W0 with zero cost the deep hedger must be within 5% of BS-BGK delta CVaR95. If not, the training is broken; do not proceed. Action std across states must be > 0.05 (guards against F3).

### E7. The decomposition: train-world by test-world matrix (P1)
- What: for each training world w in {W0..W4}, train agent A_w. Evaluate every A_w and every classical baseline in every test world, 100k test paths each, common random numbers.
- Report:
  - Matrix of CVaR95 loss, rows = hedger, columns = test world, with paired bootstrap CI.
  - Algorithm effect: best classical vs A_w in its own world.
  - Model-risk gap: A_w in test world v != w minus A_w in w.
- Why: this is the actual "model risk decomposition" the title promises. It replaces the Heston-RL minus BS subtraction.
- Output: `tables/cross_world_matrix_{I1,I2,I3}.csv`, heatmap figure.

### E8. Real-market walk-forward backtest (P1, the key table)
- What: for every test-fold start date with 30 following trading days, sell the barrier option (K = S0, B = 0.9 S0) and hedge daily on the real spot path with real surfaces as state and as marks for vanilla instruments. All hedgers: classical baselines and A_w for every world.
- Extra contracts so that the barrier is active in real data: down-and-out call with B in {0.90, 0.95, 0.97}, up-and-out call with B in {1.03, 1.05}.
- Statistics: overlapping windows, so use a stationary block bootstrap (block about 30 days) for CIs on CVaR95 and on pairwise differences; also report the non-overlapping subset. Report per fold and pooled. State effective sample size.
- Report: generator effect on real data = CVaR95(A_w on real) across w, relative to A_Heston and to the best classical hedge. This is the number that replaces "14.9%".
- Output: `tables/real_backtest_{contract}.csv`, `figures/real_backtest_*.pdf`, per-path P&L dumps.
- Gate: none on the result. Whatever comes out is the finding.

### E9. Statistical protocol (P0, applies to everything)
- Simulated worlds: >= 100k test paths, common random numbers across hedgers, paired bootstrap (10k resamples) on CVaR differences.
- Seeds: mean and std over training seeds reported separately from Monte Carlo error.
- Multiple comparisons: Holm correction within each table.
- No p < 0.0001 from a t-test on independent samples of a tail statistic.

### E10. Robustness and sensitivity (P2)
- Underlying cost in {0, 1, 2, 5, 10} bps. Expectation from the literature: learned hedgers gain mainly when costs are non-zero.
- Barrier level in {0.85, 0.90, 0.95}; maturity in {30, 60} days; rebalancing in {daily, every 2 days, weekly}.
- Risk measure in {CVaR95, CVaR99, entropic, variance}.
- Training set size for the generator: {25%, 50%, 100%} of the fold's history.
- Number of generated training paths: {10k, 50k, 200k}.

### E11. Interpretation (P2)
- Policy maps: learned hedge vs BS-BGK delta over (S/B, time to expiry) at three vol levels.
- Where the difference comes from: bucket test paths by minimum distance to barrier and by realised vol; report CVaR per bucket.
- P&L attribution: delta, gamma, vega, cost components for classical vs learned.

### Minimum viable paper
- Workshop or ICAIF-level submission: E0 to E4 (W0, W1, W2, W4), E5 core metrics, E6 with I1 and I2, E7, E8, E9.
- Stronger venue or a journal (Quantitative Finance, JCF): add W3, W5, I3, E10, E11, all folds with 5 seeds.

---

## Part 4. Build spec

General rules:
- Python >= 3.10, PyTorch >= 2.1, numpy, pandas, scipy, pyarrow, pyyaml, matplotlib, pytest.
- **No notebook logic inside `src/`.** Colab / Drive mounting lives only in `RUNBOOK.md` and `notebooks/colab_run.ipynb`. The package itself must run identically on Colab, a local GPU box, or CPU.
- Every script takes `--config <yaml>` and `--seed <int>`, sets all RNG seeds, writes outputs under a run directory with a `manifest.json` (config, git hash, seed, timestamps).
- No bare `except`. No silent NaN replacement. NaN in any tensor raises.
- Everything in the hedging path is batched torch. A 100k-path, 30-step evaluation must run in under 10 seconds on GPU.
- Type hints and docstrings on public functions. One responsibility per module.

### 4.1 Layout

```
Vol-Surfaces/   # or ddpm-vol-hedging/
  pyproject.toml            # package name: niftyhedge
  requirements.txt
  Makefile
  README.md
  EXPERIMENTS_PLAN.md
  RUNBOOK.md
  notebooks/
    colab_run.ipynb         # thin wrapper: mounts Drive, runs RUNBOOK stages
  configs/
    data.yaml
    folds.yaml
    gen/{gbm,heston,pcavar,bootstrap,ddpm}.yaml
    gen/ablations/*.yaml
    hedge/{deep_hedge,ppo}.yaml
    contracts/{doc_090,doc_095,doc_097,uoc_103,uoc_105}.yaml
    experiments/{e5_gen_quality,e7_cross_world,e8_real_backtest,e10_sensitivity}.yaml
  src/niftyhedge/
    utils/{seed.py, io.py, config.py}
    data/
      download.py           # bhavcopy (legacy + UDiFF), spot, futures, VIX
      forwards.py           # implied forward per expiry
      iv.py                 # vectorised BS price, vega, implied vol
      svi.py                # raw SVI slice fit with constraints
      surface.py            # daily 9x6 grid + mask
      events.py             # pre-announced calendar only
      features.py           # leak-free conditioning
      splits.py             # walk-forward folds
    arb/checks.py           # calendar + Durrleman butterfly
    gen/
      base.py               # World interface
      gbm.py
      heston.py             # COS pricer, calibration, simulation
      pcavar.py
      bootstrap.py
      ddpm.py               # network, schedule, sampler, EMA
      train_ddpm.py
      rollout.py            # autoregressive 30-step path generation
    hedge/
      contracts.py          # barrier payoffs, discrete monitoring
      instruments.py        # vanilla price from surface along a path
      env.py                # vectorised terminal-loss computation
      baselines.py          # all classical hedges
      policy.py             # MLP policy
      deep_hedge.py         # pathwise CVaR training
      ppo.py                # vectorised PPO
      risk.py               # VaR, CVaR, entropic, RU objective
    eval/
      gen_metrics.py        # CRPS, energy score, KS, W1, ACF, leverage, PCA
      hedge_metrics.py
      stats.py              # paired bootstrap, block bootstrap, Holm
      backtest.py           # real-data walk-forward
      cross_world.py
      plots.py
  scripts/
    00_download.py
    01_build_surfaces.py
    02_make_folds.py
    10_fit_generators.py
    11_generate_paths.py
    12_eval_generators.py
    20_train_hedgers.py
    21_cross_world_eval.py
    22_real_backtest.py
    30_sensitivity.py
    40_make_tables_figures.py
    check_build.py          # prints PASS/FAIL per item in Part 5
  tests/
    test_iv.py test_svi.py test_arb.py test_features_no_leak.py
    test_env_sign.py test_env_convergence.py test_barrier.py
    test_risk.py test_worlds_interface.py test_heston_pricer.py
    test_ddpm_shapes.py test_legacy_regression.py
  legacy/
    NiftyIV_Scaled.ipynb
    audit_env.py
  data/                     # large artefacts; raw parquet often on Drive
  models/
  results/
  tables/
  figures/
```

### 4.2 Key contracts

```python
# gen/base.py
class World(Protocol):
    name: str
    def fit(self, fold: FoldData, seed: int) -> None: ...
    def sample_paths(self, start_states: StartStates, n_paths: int,
                     horizon: int, seed: int) -> Paths: ...

@dataclass
class Paths:
    S: Tensor         # [n, H+1] spot, S[:, 0] = real start spot
    surf: Tensor      # [n, H+1, 9, 6] implied vol, strictly positive, no NaN
    start_idx: Tensor # [n] index of the real start date used
```

```python
# hedge/env.py
def terminal_loss(paths: Paths, contract: Contract,
                  h: Tensor,                 # [n, H] underlying position held over (t, t+1]
                  q: Tensor | None,          # [n, H, n_opt] vanilla positions
                  instruments: InstrumentSet,
                  cost: CostModel, p0: Tensor) -> Tensor:   # [n] loss, positive = bad
    """L = payoff - p0 - sum h*dS - sum q*dO + costs. Short one option."""
```

```python
# hedge/risk.py
def cvar(loss: Tensor, alpha: float = 0.95) -> Tensor
def ru_objective(loss: Tensor, w: Tensor, alpha: float) -> Tensor  # differentiable
```

```python
# hedge/baselines.py  (each returns h, q for a batch of paths, using only info up to t)
no_hedge, bs_barrier_delta, bs_barrier_delta_bgk, smile_delta,
ww_band, delta_vega, static_replication
```

```python
# data/features.py
def build_conditioning(surfaces, dates, spot, vix, events, lag: int = 1) -> np.ndarray
# row t must be a function of data dated <= t - lag only (events calendar excepted)
```

```python
# eval/stats.py
def paired_bootstrap_diff(loss_a, loss_b, stat, n_boot=10_000, seed=0) -> (est, lo, hi)
def block_bootstrap_ci(series, stat, block=30, n_boot=5_000, seed=0) -> (est, lo, hi)
def holm(pvals) -> np.ndarray
```

### 4.3 Implementation notes that are easy to get wrong
- Barrier delta near the barrier: use the analytic or autograd derivative of the closed form, not a 0.1% finite difference with clipping.
- BGK shift for a down barrier monitored at step dt: `B_adj = B * exp(-0.5826 * sigma * sqrt(dt))`.
- Vanilla instrument on a path: fix strike and expiry at inception; at step t look up IV at (K / F_t, remaining tau) by interpolation in total variance; price with BS. Once remaining tau drops below the shortest grid maturity, use the shortest column's IV.
- Heston: validate the COS pricer against a reference value in `test_heston_pricer.py` (for example S=100, K=100, T=1, r=0, kappa=2, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, compare to a high-precision quadrature computed in the test).
- DDPM rollout: generation of day t+1 conditions on the generated day t. Update EWMA features inside the rollout exactly as `features.py` does for real data. Test: rolling out with the real next-day values plugged in reproduces `build_conditioning` rows.
- Deep hedge training: resample fresh generator paths every epoch or use >= 200k fixed paths; evaluate on a disjoint path set; early stopping on a fixed validation path set of >= 50k paths.
- The policy must not see the path's future surfaces. Add a test that perturbs surf[:, t+1:] and asserts h[:, t] is unchanged.

---

## Part 5. Build acceptance checklist

`python scripts/check_build.py` must print PASS for each line. These are the checks that will be run to confirm the build before any GPU time is spent.

| ID | Check |
|---|---|
| A1 | `pip install -e .` works; `import niftyhedge` works. |
| A2 | `pytest -q` passes, at least the 12 test files listed in 4.1 exist and are non-trivial. |
| A3 | `test_legacy_regression.py` reproduces the Part 1 audit ordering: BS +delta worse than no hedge worse than DDPM-RL worse than BS -delta. |
| A4 | `test_env_sign.py`: in W0, correct delta hedge CVaR95 at least 50% below no hedge; wrong-sign delta worse than no hedge. |
| A5 | `test_env_convergence.py`: vanilla call, GBM, zero cost: loss std roughly halves when the number of steps quadruples. |
| A6 | `test_features_no_leak.py`: perturbing day t data leaves conditioning row t unchanged. |
| A7 | `test_arb.py`: an SVI surface built to be arbitrage-free reports zero violations; a surface with a deliberately inverted calendar slice is flagged. |
| A8 | All five worlds (W0 to W4) implement `World` and pass `test_worlds_interface.py` (shapes, positivity, no NaN, S[:,0] equals real start spot). |
| A9 | Every script in `scripts/` runs with `--smoke` (tiny sizes, CPU, under 2 minutes each) end to end on the committed sample data. |
| A10 | No occurrence of `google.colab`, `drive.mount`, bare `except:`, or `np.where(np.isnan` used as a fill in `src/`. |
| A11 | 100k-path, 30-step `terminal_loss` call runs in under 10 s on GPU (under 60 s on CPU). |
| A12 | `manifest.json` written by every script; rerunning with the same seed gives identical metrics. |
| A13 | Downloader log lists every date from 2018-01-02 to 2026-03-31 with a status; UDiFF dates after 2024-07-08 are present. |
| A14 | In W0 with zero cost, a deep hedger trained with `configs/hedge/deep_hedge.yaml` reaches within 5% of BS-BGK delta CVaR95 and has action std > 0.05. This one needs a few GPU minutes and is run as Gate 3 in the runbook. |

---

## Colab notes (for the executor)

1. Build the package in this repo first (Cursor / local). Upload the repo + `data/raw/` to Google Drive.
2. Follow `RUNBOOK.md` stages 0 → 4. Stop at every gate.
3. Rough budget: **15–25 GPU-hours** on one T4. Prefer Colab Pro / uninterrupted runtime for Stages 2–3.
4. If a stage takes ~10× longer than the estimate, something is still in a Python loop — stop and fix.
5. Negative or mixed hedging results are acceptable. Do not chase the old 60% number.
