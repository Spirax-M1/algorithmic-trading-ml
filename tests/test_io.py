import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from trading_research.core import FEATURES, evaluate, fit, predict, read_csv, run, synthetic


class BoundaryTests(unittest.TestCase):
    def test_csv_roundtrip_and_minimum_size(self):
        bars = synthetic(size=80)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bars.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['time', 'open', 'high', 'low', 'close'])
                writer.writerows((t, *row) for t, row in zip(bars.time, bars.prices))
            report, _ = run(read_csv(path))
            self.assertEqual(report['test_samples'], 15)
            for fold in report['development_folds']:
                self.assertLess(fold['last_training_label_bar'], fold['first_validation_signal_bar'])

    def test_malformed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.csv'
            for row in ['2020,1,2,3', '2020,1,2,3,4,5', '2020,1,,3,4']:
                path.write_text('time,open,high,low,close\n'+row, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_csv(path)

    def test_invalid_training_and_model_parameters(self):
        x = np.ones((30, len(FEATURES)))
        for y in [np.ones(29), np.full(30, np.nan), np.full(30, 2)]:
            with self.assertRaises(ValueError):
                fit(x, y)
        model = fit(x, np.ones(30))
        self.assertTrue(np.isfinite(predict(model, x)).all())
        model['scale'][0] = 0
        with self.assertRaises(ValueError):
            predict(model, x)

    def test_invalid_returns(self):
        for returns in [[], [float('nan')], [[.1]]]:
            with self.assertRaises(ValueError):
                evaluate(returns, np.zeros(np.asarray(returns).shape), 0)

    def test_cli_demo_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            cmd = [sys.executable, '-m', 'trading_research', '--demo', '--output', directory]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['data_source'], 'SYNTHETIC DEMO')
            self.assertEqual(len(report['test_observations']), report['test_samples'])
            self.assertEqual(len(report['source_sha256']), 64)
            before = (Path(directory)/'report.json').read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(before, (Path(directory)/'report.json').read_bytes())

    def test_cli_bad_csv_is_readable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.csv'
            path.write_text('time,open,high,low,close\n2020,1,2\n')
            result = subprocess.run([sys.executable, '-m', 'trading_research', '--csv', str(path),
                                     '--output', str(Path(directory)/'out')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('Traceback', result.stderr)
