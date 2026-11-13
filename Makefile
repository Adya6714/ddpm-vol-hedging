.PHONY: smoke test install check

install:
	pip install -U pip
	pip install -e ".[dev]"

test:
	pytest -q

check:
	python scripts/check_build.py

smoke:
	python scripts/00_download.py --smoke --config configs/data.yaml
	python scripts/01_build_surfaces.py --smoke --config configs/data.yaml
	python scripts/02_make_folds.py --smoke --config configs/folds.yaml
	python scripts/10_fit_generators.py --smoke --config configs/gen/gbm.yaml --folds 0 --seeds 0
	python scripts/10_fit_generators.py --smoke --config configs/gen/heston.yaml --folds 0 --seeds 0
	python scripts/10_fit_generators.py --smoke --config configs/gen/pcavar.yaml --folds 0 --seeds 0
	python scripts/10_fit_generators.py --smoke --config configs/gen/bootstrap.yaml --folds 0 --seeds 0
	python scripts/10_fit_generators.py --smoke --config configs/gen/ddpm.yaml --folds 0 --seeds 0
	python scripts/11_generate_paths.py --smoke --worlds gbm,heston,pcavar,bootstrap,ddpm --folds 0 --n-train 64 --n-test 32
	python scripts/12_eval_generators.py --smoke --config configs/experiments/e5_gen_quality.yaml
	python scripts/20_train_hedgers.py --smoke --config configs/hedge/deep_hedge.yaml --world gbm --fold 0 --instruments I1 --contract configs/contracts/doc_090.yaml --seeds 0
	python scripts/21_cross_world_eval.py --smoke --config configs/experiments/e7_cross_world.yaml
	python scripts/22_real_backtest.py --smoke --config configs/experiments/e8_real_backtest.yaml
	python scripts/30_sensitivity.py --smoke --config configs/experiments/e10_sensitivity.yaml
	python scripts/40_make_tables_figures.py --smoke --run runs/smoke
