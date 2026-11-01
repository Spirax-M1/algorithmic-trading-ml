# Development

Maintainer: Salvatore (Spirax-M1), smanciagli1@outlook.com.

Use Python 3.11+ and install the root requirements.txt in a virtual environment. The supported code is in trading_research; legacy scripts are disabled and excluded from the supported runtime.

Before proposing changes:

```sh
python -m unittest discover -s tests -v
python -m trading_research --demo --output artifacts/review-01
```

Use a new output directory for every experiment. Include regression tests for timing, data validation or accounting changes. Changes to targets, costs, splits or feature availability must also update docs/methodology.md. Never describe synthetic metrics as financial results.

Keep the active rebuild PR in draft until the real-data protocol and its limitations have been reviewed. Preserve historical source for traceability. Document unfinished work in docs/RESUME.md so another environment can continue without chat history.
