# Historical architecture notes

Author: Salvatore (Spirax-M1).

V26 contains independent LSTM and XGBoost training paths over M5 data. V29 contains XGBoost training over H1/H4 data. Their scripts duplicate feature calculations between training, backtesting and live inference. The duplication resulted in inconsistent schemas and timing.

The original architecture narrative overstated what was implemented. It has been replaced by the source-based [audit](../../docs/audit.md). For the supported offline architecture and execution contract, read the [root README](../../README.md).

Future work should reuse one feature implementation and save the exact schema, preprocessing parameters and training cutoff with each model. Multi-timeframe features require availability timestamps rather than an unchecked forward fill on bar-open timestamps.

Original snapshot: [source and previous documentation](https://github.com/Spirax-M1/algorithmic-trading-ml/tree/930e3aa552236727e24ea68f9cec5280d2bd5ed8).
