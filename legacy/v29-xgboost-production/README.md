# V29: historical XGBoost experiment

Author: Salvatore (Spirax-M1).

The directory name is retained for traceability; it does not describe deployment readiness. Python entrypoints are disabled.

The original training combines H1/H4 indicators, three opportunity labels and Optuna tuning. It labels each row from extrema in the next 100 bars. The backtest instead simulates stop/target exits over up to 48 bars. Training and the bot have incompatible feature lists and model-directory settings.

Backward filling can introduce future observations. Neither cross-validation nor the final temporal split purges the label horizon. Scaler and class weights are calculated before CV. H4 alignment also needs a verified timestamp convention before its causal validity can be established.

Historical win rate, profit factor, drawdown, latency and memory figures are not reproducible from the supplied repository and are withdrawn. The maintained replacement uses a different, explicitly documented target; it is not a repaired or validated V29 model.

See the [audit](../../docs/audit.md), [current pipeline](../../README.md) and [resume instructions](../../docs/RESUME.md). The dependency list is historical evidence, not a supported installation environment.

Original snapshot: [source and previous documentation](https://github.com/Spirax-M1/algorithmic-trading-ml/tree/930e3aa552236727e24ea68f9cec5280d2bd5ed8).
