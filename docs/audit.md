# Original implementation audit

Scope: static review of all six Python entrypoints and repository documentation at original Git tree `930e3aa552236727e24ea68f9cec5280d2bd5ed8`. Historical CSVs, models and execution logs were not supplied. Original TensorFlow/XGBoost/MT5 pipelines were not run. These findings do not measure real performance.

## V29: incompatible training and live inputs (critical)

Training derives its feature list by exclusion and writes a FEATURES JSON artifact. The live bot ignores it and hardcodes a different schema: e.g. returns_2, stoch_k and rsi_divergence_h4. Training writes cervelli_IA_V29_PRODUCTION while live reads cervelli_IA_V29. Independently calculated indicators further risk formula drift. The claimed end-to-end deployment cannot be reproduced from the published files without reconciliation.

## V29: temporal contamination (critical)

- Training and backtest call bfill after feature calculation, allowing future values to fill early observations.
- Labels depend on the next 100 bars. The 80/20 split and TimeSeriesSplit have no horizon purge: training labels near boundaries depend on validation prices.
- Scaler and class weights are calculated before CV folds. Fold validation observations therefore influence preprocessing/weighting.
- H4 aggregates are forward-filled at their input timestamps. If timestamps denote bar opens, completed H4 values become visible before they were available. Input timestamp semantics require real-data verification.

## V29: evaluation contract (high)

Training removes the final 100 rows to create labels; backtesting splits a differently sized feature dataframe independently. Consequently the backtest boundary is not tied to the saved training boundary. Confidence thresholds are compared on that backtest period: selecting the best result would require a new untouched test period. The simulator skips 48 bars even if stop/take-profit exited early, so its scheduling does not follow actual exit timestamps. Its final “approved for live deployment” message is not a valid release criterion.

## V26: normalization and feature timing (critical)

MinMaxScaler is fitted before the temporal split. LSTM sequences use data[i-SEQ_LEN:i], ending at i-1, while XGBoost uses data[i] and the label is anchored at i. The two models therefore do not receive the same information cutoff. The split also lacks a purge for the six-bar future label horizon.

## V26: evaluation limitations (high)

The historical simulator explicitly ignores commission/swap and uses a simplified contract-size conversion. Its calendar test start is not a validated artifact-level separation from training. These assumptions must be resolved before interpreting historical P&L or comparing symbols/account currencies.

## Documentation mismatches (high)

The root README described a three-layer bidirectional LSTM with attention and a custom loss. The published builder contains two unidirectional LSTM layers and sparse categorical cross-entropy. Claims of 50+ multi-timeframe features do not match the published V26 feature preparation. Performance ranges lack reproducible datasets and reports. The clone URL contains a placeholder. The profile's environment-variable secret claim does not describe the bots' token placeholder configuration.

## Rebuild decisions

The supported path is now an intentionally smaller offline baseline. Legacy entrypoints are disabled and original claims explicitly withdrawn. The replacement implements causal single-timeframe features, next-open/next-close execution, purged folds, fold-local scaling, fixed model settings, baselines and JSON artifacts. This retires incompatible paths rather than pretending their defects have been patched and validated.

## What remains unproven

Real-market predictive value, broker integration, GPU training, XGBoost/LSTM model quality, deployment reliability and historical profitability. Synthetic demo output proves execution and determinism only. Reintroduce advanced models only after obtaining documented real data and freezing an evaluation protocol.
