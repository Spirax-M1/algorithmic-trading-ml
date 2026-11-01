"""Small reference pipeline with explicit timing and no financial dependencies."""
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FEATURES = ('return_1', 'return_5', 'volatility_10', 'range_fraction')


@dataclass(frozen=True)
class Bars:
    time: tuple[str, ...]
    prices: np.ndarray  # open, high, low, close


def validate(bars):
    p = np.asarray(bars.prices, dtype=float)
    if p.shape != (len(bars.time), 4) or len(p) < 80:
        raise ValueError('Need at least 80 OHLC bars with matching timestamps')
    dates = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in bars.time]
    if any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in dates):
        raise ValueError('Timestamps must have explicit UTC timezone')
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError('Timestamps must be unique and strictly increasing')
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('OHLC values must be finite and positive')
    if ((p[:, 1] < p.max(axis=1)) | (p[:, 2] > p.min(axis=1))).any():
        raise ValueError('Inconsistent OHLC bounds')
    return bars


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['time', 'open', 'high', 'low', 'close']:
            raise ValueError('Required columns: time,open,high,low,close (in this order)')
        rows = list(reader)
    return validate(Bars(tuple(r['time'] for r in rows), np.array([
        [float(r[k]) for k in ('open', 'high', 'low', 'close')] for r in rows
    ])))


def synthetic(seed=42, size=1000):
    """Random artificial hourly bars; never evidence of market performance."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .002, size)))
    opening = np.r_[100., close[:-1]] * np.exp(rng.normal(0, .0003, size))
    high = np.maximum(opening, close) * (1 + rng.uniform(0, .001, size))
    low = np.minimum(opening, close) * (1 - rng.uniform(0, .001, size))
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return validate(Bars(tuple((start + timedelta(hours=i)).isoformat() for i in range(size)),
                         np.column_stack((opening, high, low, close))))


def features(bars):
    """Row t is available only after bar t closes. Warm-up rows stay NaN."""
    p = bars.prices
    close = p[:, 3]
    returns = np.r_[np.nan, np.diff(np.log(close))]
    result = np.full((len(p), len(FEATURES)), np.nan)
    for t in range(10, len(p)):
        result[t] = (returns[t], np.log(close[t] / close[t-5]),
                     np.std(returns[t-9:t+1]), (p[t, 1]-p[t, 2])/close[t])
    return result


def samples(bars):
    """Predict next bar open-to-close direction, enter at that next open."""
    x = features(bars)
    idx = np.arange(10, len(x)-1)
    realized = bars.prices[idx+1, 3] / bars.prices[idx+1, 0] - 1
    return x[idx], (realized > 0).astype(float), realized, idx


def purged_split(indices, boundary):
    """Exclude targets ending at or beyond the first test signal bar."""
    train = np.flatnonzero(indices + 1 < boundary)
    test = np.flatnonzero(indices >= boundary)
    if len(train) < 20 or len(test) < 1:
        raise ValueError('Insufficient observations after purging')
    return train, test


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit(x, y, epochs=400):
    """Fixed logistic baseline; normalization is fitted on training only."""
    if len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Invalid training data')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1., scale)
    z = (x - mean) / scale
    w, bias = np.zeros(x.shape[1]), 0.
    for _ in range(epochs):
        error = sigmoid(z @ w + bias) - y
        w -= .1 * (z.T @ error / len(z) + .01 * w)
        bias -= .1 * error.mean()
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': w.tolist(),
            'bias': float(bias), 'features': list(FEATURES)}


def predict(model, x):
    if model['features'] != list(FEATURES) or x.shape[1] != len(FEATURES):
        raise ValueError('Feature schema mismatch')
    if not np.isfinite(x).all():
        raise ValueError('Non-finite features')
    return sigmoid((x-np.array(model['mean']))/np.array(model['scale']) @
                   np.array(model['weights']) + model['bias'])


def evaluate(realized, positions, cost_bps):
    """One separate trade per bar; full round-trip cost for every active bar.

    Returns are unlevered fractions of capital, compounded at bar closes.
    Intrabar drawdown, financing, liquidity and broker fills are not modeled.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('cost_bps must be finite and nonnegative')
    positions = np.asarray(positions)
    if positions.shape != realized.shape or not np.isin(positions, [-1, 0, 1]).all():
        raise ValueError('Invalid positions')
    net = realized * positions - np.abs(positions) * cost_bps / 10000
    if (net <= -1).any():
        raise ValueError('Simulation exhausted capital')
    equity = np.r_[1., np.cumprod(1+net)]
    dd = equity / np.maximum.accumulate(equity) - 1
    active = positions != 0
    return {'trades': int(active.sum()), 'total_return': float(equity[-1]-1),
            'max_close_drawdown': float(-dd.min()),
            'win_rate': float((net[active] > 0).mean()) if active.any() else None,
            'net_returns': net.tolist()}


def run(bars, cost_bps=2.):
    validate(bars)
    x, y, realized, idx = samples(bars)
    boundary = int(len(bars.time)*.8)
    train, test = purged_split(idx, boundary)
    # Expanding folds confined to the development period. Hyperparameters are fixed.
    folds = []
    for start, stop in zip((.4, .55, .7), (.55, .7, .8)):
        edge = int(len(bars.time)*start)
        tr, va = purged_split(idx, edge)
        va = va[idx[va]+1 < int(len(bars.time)*stop)]
        if not len(va):
            raise ValueError('Empty validation fold')
        model = fit(x[tr], y[tr])
        probability = predict(model, x[va])
        folds.append({'train_samples': len(tr), 'validation_samples': len(va),
                      'accuracy': float(((probability >= .5) == y[va]).mean())})
    model = fit(x[train], y[train])
    probability = predict(model, x[test])
    positions = np.where(probability >= .5, 1, -1)
    strategies = {
        'logistic': evaluate(realized[test], positions, cost_bps),
        'always_long': evaluate(realized[test], np.ones(len(test)), cost_bps),
        'no_trade': evaluate(realized[test], np.zeros(len(test)), cost_bps),
    }
    report = {'schema_version': 1, 'features': list(FEATURES), 'cost_bps_round_trip': cost_bps,
              'train_samples': len(train), 'test_samples': len(test),
              'last_training_label_bar': bars.time[idx[train[-1]]+1],
              'first_test_signal_bar': bars.time[idx[test[0]]],
              'last_test_exit_bar': bars.time[idx[test[-1]]+1],
              'development_folds': folds,
              'test_accuracy': float(((probability >= .5) == y[test]).mean()),
              'majority_baseline_accuracy': float((y[test] == (y[train].mean() >= .5)).mean()),
              'strategies': strategies}
    return report, model
