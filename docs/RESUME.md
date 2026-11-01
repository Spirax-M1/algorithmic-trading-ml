# Resume this project on another computer

Last updated: 2026-11-01. Maintainer: Salvatore (Spirax-M1).

## Active work

Repository: https://github.com/Spirax-M1/algorithmic-trading-ml

Draft PR: https://github.com/Spirax-M1/algorithmic-trading-ml/pull/1

Branch: `rebuild/reproducible-research`. Do not start from main: the rebuild is deliberately unmerged.

```sh
git clone --branch rebuild/reproducible-research https://github.com/Spirax-M1/algorithmic-trading-ml.git
cd algorithmic-trading-ml
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m trading_research --demo --output artifacts/resume-01
```

On Windows use `.venv\Scripts\activate`. No broker, credentials or GPU are needed. Choose a new output directory if an experiment already exists.

## User intent and decisions

Review the GitHub projects one by one, correct them and improve their professional presentation. The first project is this ML repository. Upload progress as a draft so it can be continued from another computer. Rewrite contradictory/generated-looking documentation, remove template placeholders and avoid unsupported performance claims.

The user confirmed the historical CSVs are on another computer. They are not lost, but are unavailable here. Do not claim real-data evaluation is complete or request that the present machine supply them. Other projects (WoW localization, Keep contribution, profile and Simple-tools) have not been modified during this rebuild.

## Completed

- Static audit of the six historical entrypoints and their documentation.
- Archived V26/V29 source under legacy; entrypoints stop before imports or external connections.
- Replaced the supported path with a smaller NumPy logistic baseline using causal features, purged temporal splits, fold-local scaling and fixed settings.
- Added cost-aware comparisons, JSON model/report output, source/data fingerprints and timestamped holdout observations.
- Added input/model validation and protection against overwriting previous experiment files.
- Rewrote all historical Markdown descriptions; old prose remains recoverable at original commit `930e3aa552236727e24ea68f9cec5280d2bd5ed8`.
- Added data instructions, methodology, contribution guide and this handoff.

Local verification: 18 tests passed on Python 3.14.8 / NumPy 2.5.4. CI tests Linux and Windows on Python 3.11 and 3.13. The previous draft commit f158478cbf6f58a31e8000ed8a849708e8a8f715 passed CI; check the latest PR checks for this update rather than assuming the earlier result applies.

## Remaining work

1. Inspect the original CSV headers and a sanitized sample on the computer that has them. Verify timezone, cadence, instrument, completed bars and data provenance using docs/data.md.
2. Implement a tested conversion if necessary. Never silently infer broker timezone or repair gaps with future values.
3. Freeze a real-data evaluation period, target and cost assumptions before viewing results. Run the baseline and retain both negative and positive results.
4. Compare a library-backed logistic model and XGBoost using identical information cutoffs; tune only inside development folds. The current reference baseline does not restore V26/V29 functionality.
5. Consider multi-timeframe features only with explicit availability timestamps and future-data-invariance tests. Reintroduce LSTM only if it answers a defined experiment.
6. Review historical broker execution separately; never re-enable archived bots merely because the offline tests pass.
7. Once the project is ready, review the draft and decide when to merge. Continue with the WoW project, then profile and other repositories.

## Known limits

No original dataset, trained historical artifact or broker session has been evaluated. No trading edge, live deployment readiness or historical profit claim is established. Dependencies use a version range, not a fully locked environment. Input cadence and calendar gaps require external review. The current baseline uses a different target from V26/V29 and deliberately omits GPU and broker integrations.

## Prompt for a fresh session

“Continue https://github.com/Spirax-M1/algorithmic-trading-ml/pull/1 on branch rebuild/reproducible-research. Read docs/RESUME.md, docs/audit.md and docs/methodology.md first. Inspect the current PR and CI. Keep the work in draft and update the handoff. The original CSVs are on another computer; do not claim validation without inspecting them.”
