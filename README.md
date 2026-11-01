# Time-series ML research

Developed by **Salvatore ([Spirax-M1](https://github.com/Spirax-M1))**.

An offline reference pipeline for evaluating directional models without mixing future observations into training. The maintained implementation uses a small logistic classifier, explicit execution timing, chronological evaluation and comparison baselines.

**Status: research rebuild.** The demo uses synthetic prices. It does not establish predictive skill or profitability. Historical V26/V29 experiments are preserved under [`legacy/`](legacy/); their performance claims are withdrawn pending independent reproduction. There is no supported live trading path.

## Run

Python 3.11+ and NumPy are required. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m trading_research --demo
```

On Windows, activate with `.venv\Scripts\activate` instead. Existing experiment files are never overwritten; use a fresh `--output` directory for each run. The experiment writes `artifacts/report.json` and `artifacts/model.json`. Reports include source and dataset fingerprints, timestamped test observations, environment versions, split timestamps, development-fold scores, held-out classification metrics and net returns for three strategies. Model parameters use JSON, not executable serialization.

To use your own data:

```sh
python -m trading_research --csv prices.csv --cost-bps 2 --output artifacts/experiment-01
```

The exact CSV header is `time,open,high,low,close`. Timestamps denote bar opens, must be unique, increasing and explicitly UTC (for example `2024-01-01T00:00:00+00:00`). Supply at least 80 completed bars for a single instrument and consistent timeframe. Do not include a currently forming bar. The reader rejects missing, non-finite and inconsistent OHLC values. No interpolation is performed.

## Experiment design

1. Four features use observations available at the close of signal bar t. The initial ten bars remain undefined and are excluded.
2. The target is the direction from open to close of bar t+1. A simulated position enters at that next open and exits at its close.
3. Three expanding development folds precede a final 20% chronological holdout. Samples whose labels reach the validation boundary are purged.
4. Each model fits its normalization using its training subset only. Logistic optimization settings are fixed; there is no hyperparameter or threshold search on the holdout.
5. Compare the classifier against always-long and no-trade strategies. Classification also includes a training-majority baseline.

Every active bar is a separate trade charged the specified round-trip cost in basis points. Capital compounds with unlevered returns; drawdown is measured at bar closes including initial capital. Costs are simplified, with no financing, liquidity, bid/ask execution, intrabar drawdown or margin model. Signals at the close followed by fills at the next open assume negligible processing delay. Missing calendar intervals are retained: “next bar” means next observed bar.

## What this demonstrates

- Causal features and purged chronological splits.
- One feature schema and model implementation for fitting and prediction.
- Regression tests that perturb future data and verify past features and fitted models stay unchanged.
- Explicit assumptions, serializable artifacts and baseline comparisons.

The logistic implementation is deliberately a reference baseline, not a replacement claim for the old XGBoost/LSTM results. A library-backed model and real-data evaluation belong in a subsequent experiment using the same protocol.

## Repository

- `trading_research/`: maintained NumPy pipeline and CLI.
- `tests/`: timing, cost, data-validation and reproducibility regression tests.
- `docs/audit.md`: source-based review of the original experiments.
- `legacy/`: historical code and documentation, not supported for execution.

## Next research milestones

- Record the provenance, usage permissions, timestamp semantics and coverage of a real dataset.
- Run a preregistered real-data baseline; preserve negative results.
- Compare XGBoost using the same splits and feature schema; tune only inside development folds.
- Add causal multi-timeframe features with explicit bar-availability timestamps.
- Evaluate cost sensitivity and performance across separate periods before considering deployment.

See [LICENSE](LICENSE). Contributions should include a regression test for the failure being addressed.

## Continuing this work

The active draft is [PR #1](https://github.com/Spirax-M1/algorithmic-trading-ml/pull/1), branch `rebuild/reproducible-research`. Start with [RESUME.md](docs/RESUME.md) on another computer.

- [Methodology and limitations](docs/methodology.md)
- [Data preparation](docs/data.md)
- [Source audit](docs/audit.md)
- [Contributing](CONTRIBUTING.md)

Contact: [smanciagli1@outlook.com](mailto:smanciagli1@outlook.com).
