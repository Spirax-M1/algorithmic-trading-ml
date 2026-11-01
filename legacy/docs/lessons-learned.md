# Findings from the rebuild

Author: Salvatore (Spirax-M1).

The original narrative mixed implementation notes with unverified outcomes. This replacement records findings tied to code:

- Forward filling is causal only when the source value was already available. Completed higher-timeframe bars cannot be exposed at their opening time.
- A chronological split alone does not prevent leakage when training labels extend into the next period.
- Preprocessing must be fitted separately inside each training fold.
- Training and prediction need one feature schema and the same observation cutoff.
- Backtest cost assumptions and execution timing must match the target being evaluated.
- A successful synthetic run establishes software behavior, not market performance.

Regression tests now check future-data invariance, split separation, costs, serialization and CLI input handling. The [audit](../../docs/audit.md) and [handoff](../../docs/RESUME.md) distinguish completed work from pending real-data evaluation.

Original snapshot: [source and previous documentation](https://github.com/Spirax-M1/algorithmic-trading-ml/tree/930e3aa552236727e24ea68f9cec5280d2bd5ed8).
