# Evaluation contract

The maintained pipeline is a reference binary logistic baseline. It replaces the supported execution path, not the historical scientific experiment: its target and features differ from V26/V29.

## Information and execution timing

A row's timestamp is the opening of a completed bar. The signal uses that bar's closing price, one- and five-bar log returns, ten-bar return volatility and high-low range divided by close. No future row enters those calculations. The first ten rows are excluded; there is no backward fill.

The label is 1 if the next observed bar closes above its open, otherwise 0. A position is +1 at probability >= 0.5 and -1 otherwise. It enters at the next open and exits at that bar's close. Zero-return bars are class 0; the strategy still incurs costs. Gaps are not filled: the next observed row is used, so input cadence and missing intervals must be reviewed separately.

## Training and evaluation

Three expanding development windows start validation at 40%, 55% and 70% of the input rows and end before 55%, 70% and 80%. A separate final holdout starts at 80%. Each split excludes training labels whose outcome bar reaches the first validation signal bar. Each fold fits its own normalization.

The logistic optimizer uses 400 full-batch gradient steps, learning rate 0.1 and L2 coefficient 0.01. The intercept is unpenalized; zero-variance features use scale 1. These settings are fixed. This is a transparent reference implementation, not a claim of equivalence to a particular library solver or guaranteed convergence for arbitrary data.

The final model is trained only on the pre-holdout data. The report includes development accuracy, holdout accuracy and a majority-class comparator determined from training labels. No model choice should be made by repeatedly examining the holdout; once used to guide changes, that period becomes development data and a new holdout is needed.

## Simulated returns

For execution bar i, gross return is `position * (close / open - 1)`. Net return subtracts `abs(position) * cost_bps / 10000`. The cost is a complete round trip per active bar, even for consecutive positions in the same direction. Always-long uses the same independent-trade convention; it is not buy-and-hold. No-trade stays in cash with no interest or costs.

Equity starts at 1 and compounds net returns. Maximum drawdown compares each bar-close equity with the running peak including initial capital. Win rate is the fraction of active trades with positive net return. No-trade win rate is null.

There is no leverage, currency conversion, borrow fee, financing, stop/target logic, liquidity or intrabar price-path model. Next-open fills assume negligible processing delay. No annualized Sharpe is reported because the input frequency and missing-bar calendar are not inferred.

## Reproduction

The JSON report records normalized data and source hashes, exact Python/NumPy versions, split timestamps, configuration and each holdout observation. The model JSON records the feature schema and fitted parameters. Source hashes identify content, not a signed provenance record. NumPy is a compatible version range rather than a lockfile; small floating-point differences across environments are possible. Retain the environment versions for each real-data experiment.
