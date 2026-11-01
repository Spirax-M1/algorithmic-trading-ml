"""Run an offline experiment: python -m trading_research --demo."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .core import read_csv, run, synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Use explicitly synthetic data')
    source.add_argument('--csv', type=Path, help='UTC bar-open time,open,high,low,close CSV')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--cost-bps', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        bars = synthetic(args.seed) if args.demo else read_csv(args.csv)
        report, model = run(bars, args.cost_bps)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    fingerprint = json.dumps({'time': bars.time, 'prices': bars.prices.tolist()},
                             separators=(',', ':')).encode()
    report.update({'data_source': 'SYNTHETIC DEMO' if args.demo else 'USER CSV',
                   'seed': args.seed if args.demo else None,
                   'data_sha256': hashlib.sha256(fingerprint).hexdigest(),
                   'python': platform.python_version(), 'numpy': np.__version__,
                   'limitations': ['No evidence of profitability from synthetic data.',
                                   'Fixed costs; no financing, liquidity or intrabar risk model.',
                                   'Historical OHLC only; no broker execution.']})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('report', report), ('model', model)]:
        (args.output / f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(f"{report['data_source']}: {report['test_samples']} held-out samples")
    print(f"Report: {args.output / 'report.json'}")


if __name__ == '__main__':
    main()
