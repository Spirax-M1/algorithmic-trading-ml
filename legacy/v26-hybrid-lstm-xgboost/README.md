# V26: historical LSTM and XGBoost experiment

Author: Salvatore (Spirax-M1).

This directory preserves the original training, backtest and MT5 bot for review. Python entrypoints are disabled. It is not a supported application.

The training script builds two unidirectional LSTM layers and a separate XGBoost classifier. It reads per-symbol M5 CSV files from `Dati_Fresh_2026` and writes model/scaler files into `cervelli_IA_2026`. Those datasets and model artifacts are not included.

The scaler was fitted before validation separation. LSTM windows ended one bar earlier than the XGBoost input for the same target. The six-bar label horizon was not purged at the validation boundary. These defects prevent treating the historical scores as a reliable out-of-sample evaluation.

Earlier descriptions of bidirectional layers, attention and a custom loss did not match this source. No accuracy, Sharpe, latency or memory claims are retained.

See the [audit](../../docs/audit.md), [current pipeline](../../README.md) and [resume instructions](../../docs/RESUME.md). The dependency list is historical evidence, not a supported installation environment.

Original snapshot: [source and previous documentation](https://github.com/Spirax-M1/algorithmic-trading-ml/tree/930e3aa552236727e24ea68f9cec5280d2bd5ed8).
