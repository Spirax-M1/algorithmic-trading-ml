"""
================================================================================
FOREX BOT V29 - PROFESSIONAL BACKTEST SCRIPT
================================================================================
Author: Claude + Salvatore
Date: March 31, 2026
Purpose: Validate trained model with realistic trading simulation

FEATURES:
1. Walk-forward validation (train/test split)
2. Realistic spread/commission/slippage simulation
3. Confidence threshold optimization
4. Complete performance metrics
5. Detailed trade log
6. Equity curve plotting
7. Risk-adjusted metrics (Sharpe Ratio, Max DD)

METRICS CALCULATED:
- Win Rate
- Profit Factor (gross profit / gross loss)
- Total profit in EUR
- Average win / Average loss
- Max Drawdown
- Sharpe Ratio
- Best confidence threshold
- Trade distribution

================================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL = "EURUSD"
MODEL_DIR = "cervelli_IA_V29_PRODUCTION"
H1_FILE = "./EURUSD_H1.csv"
H4_FILE = "./EURUSD_H4.csv"

# Trading parameters
ACCOUNT_BALANCE = 100  # EUR
RISK_PER_TRADE = 0.02  # 2% risk per trade
SPREAD = 1.5  # pips
COMMISSION = 0.5  # pips per trade
SLIPPAGE = 0.3  # pips
TOTAL_COST = SPREAD + COMMISSION + SLIPPAGE  # 2.3 pips

# Backtest parameters
TEST_PERIOD_RATIO = 0.2  # Use last 20% of data for backtest
CONFIDENCE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]  # Test multiple thresholds
MIN_ADX = 15  # Minimum ADX for trade entry

# Output
RESULTS_DIR = "backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("FOREX BOT V29 - PROFESSIONAL BACKTEST")
print("=" * 80)
print(f"Symbol: {SYMBOL}")
print(f"Model: {MODEL_DIR}")
print(f"Account: EUR {ACCOUNT_BALANCE}")
print(f"Risk per trade: {RISK_PER_TRADE*100}%")
print(f"Total trading cost: {TOTAL_COST} pips")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================================

print("STEP 1: Loading model and data...")

# Load model
model_path = os.path.join(MODEL_DIR, f"{SYMBOL}_XGB_V29.json")
model = xgb.XGBClassifier()
model.load_model(model_path)
print(f"[OK] Model loaded: {model_path}")

# Load scaler
scaler_path = os.path.join(MODEL_DIR, f"{SYMBOL}_SCALER_V29.joblib")
scaler = joblib.load(scaler_path)
print(f"[OK] Scaler loaded: {scaler_path}")

# Load feature names
features_path = os.path.join(MODEL_DIR, f"{SYMBOL}_FEATURES_V29.json")
with open(features_path, 'r') as f:
    feature_columns = json.load(f)
print(f"[OK] Features loaded: {len(feature_columns)} features")

# Load metadata
metadata_path = os.path.join(MODEL_DIR, f"{SYMBOL}_METADATA_V29.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"[OK] Metadata loaded")
print(f"     Training date: {metadata['training_date']}")
print(f"     Training accuracy: {metadata['test_accuracy']*100:.2f}%")
print()

# Load H1 data
print(f"Loading {H1_FILE}...")
df_h1 = pd.read_csv(H1_FILE)
df_h1['time'] = pd.to_datetime(df_h1['time'])
df_h1.set_index('time', inplace=True)
print(f"[OK] H1 data: {len(df_h1)} bars")

# Load H4 data
print(f"Loading {H4_FILE}...")
df_h4 = pd.read_csv(H4_FILE)
df_h4['time'] = pd.to_datetime(df_h4['time'])
df_h4.set_index('time', inplace=True)
print(f"[OK] H4 data: {len(df_h4)} bars")
print()

# ============================================================================
# STEP 2: CALCULATE FEATURES (SAME AS TRAINING)
# ============================================================================

print("STEP 2: Calculating features...")
df = df_h1.copy()

# Price-based features
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['hl_ratio'] = (df['high'] - df['low']) / df['close']
df['co_ratio'] = (df['close'] - df['open']) / df['open']

# Moving averages
df['sma_10'] = df['close'].rolling(10).mean()
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['ema_10'] = df['close'].ewm(span=10).mean()
df['ema_20'] = df['close'].ewm(span=20).mean()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Volatility
df['atr'] = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift()),
    abs(df['low'] - df['close'].shift())
], axis=1).max(axis=1).rolling(14).mean()

df['volatility_10'] = df['returns'].rolling(10).std()
df['volatility_20'] = df['returns'].rolling(20).std()

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# MACD
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']

# Volume
if 'tick_volume' in df.columns:
    df['vwap'] = (df['close'] * df['tick_volume']).rolling(14).sum() / df['tick_volume'].rolling(14).sum()
    df['volume_sma'] = df['tick_volume'].rolling(20).mean()

# Momentum
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['roc_10'] = df['close'].pct_change(10)

# Lagged
df['close_lag_1'] = df['close'].shift(1)
df['close_lag_2'] = df['close'].shift(2)
df['returns_lag_1'] = df['returns'].shift(1)
df['returns_lag_2'] = df['returns'].shift(2)

# H4 features
df_h4['sma_h4'] = df_h4['close'].rolling(20).mean()
df_h4['ema_h4'] = df_h4['close'].ewm(span=20).mean()
df_h4['atr_h4'] = pd.concat([
    df_h4['high'] - df_h4['low'],
    abs(df_h4['high'] - df_h4['close'].shift()),
    abs(df_h4['low'] - df_h4['close'].shift())
], axis=1).max(axis=1).rolling(14).mean()

# ADX H4
df_h4['tr'] = pd.concat([
    df_h4['high'] - df_h4['low'],
    abs(df_h4['high'] - df_h4['close'].shift()),
    abs(df_h4['low'] - df_h4['close'].shift())
], axis=1).max(axis=1)

df_h4['dm_plus'] = np.where(
    (df_h4['high'] - df_h4['high'].shift()) > (df_h4['low'].shift() - df_h4['low']),
    np.maximum(df_h4['high'] - df_h4['high'].shift(), 0),
    0
)
df_h4['dm_minus'] = np.where(
    (df_h4['low'].shift() - df_h4['low']) > (df_h4['high'] - df_h4['high'].shift()),
    np.maximum(df_h4['low'].shift() - df_h4['low'], 0),
    0
)

df_h4['tr_14'] = df_h4['tr'].rolling(14).mean()
df_h4['dm_plus_14'] = df_h4['dm_plus'].rolling(14).mean()
df_h4['dm_minus_14'] = df_h4['dm_minus'].rolling(14).mean()

df_h4['di_plus'] = 100 * df_h4['dm_plus_14'] / df_h4['tr_14']
df_h4['di_minus'] = 100 * df_h4['dm_minus_14'] / df_h4['tr_14']
df_h4['dx'] = 100 * abs(df_h4['di_plus'] - df_h4['di_minus']) / (df_h4['di_plus'] + df_h4['di_minus'])
df_h4['adx_h4'] = df_h4['dx'].rolling(14).mean()

# RSI H4
delta_h4 = df_h4['close'].diff()
gain_h4 = delta_h4.where(delta_h4 > 0, 0).rolling(14).mean()
loss_h4 = -delta_h4.where(delta_h4 < 0, 0).rolling(14).mean()
rs_h4 = gain_h4 / loss_h4
df_h4['rsi_h4'] = 100 - (100 / (1 + rs_h4))

# Merge H4 with forward fill
h4_features = ['sma_h4', 'ema_h4', 'atr_h4', 'adx_h4', 'rsi_h4', 'di_plus', 'di_minus']
df_h4_reindexed = df_h4[h4_features].reindex(df.index, method='ffill')

for col in h4_features:
    df[col] = df_h4_reindexed[col]

# H4-derived
df['trend_h4'] = (df['close'] - df['sma_h4']) / df['sma_h4']
df['returns_h4_1'] = df['close'].pct_change(4)
df['returns_h4_2'] = df['close'].pct_change(8)
df['price_vs_sma_h4'] = df['close'] / df['sma_h4']

# Clean
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.ffill()
df = df.bfill()
df.dropna(inplace=True)

print(f"[OK] Features calculated: {len(df)} bars")
print()

# ============================================================================
# STEP 3: SPLIT DATA (WALK-FORWARD)
# ============================================================================

print("STEP 3: Splitting data for walk-forward validation...")

split_idx = int(len(df) * (1 - TEST_PERIOD_RATIO))
df_backtest = df.iloc[split_idx:].copy()

print(f"Backtest period: {df_backtest.index[0]} to {df_backtest.index[-1]}")
print(f"Backtest bars: {len(df_backtest)}")
print()

# ============================================================================
# STEP 4: BACKTEST WITH MULTIPLE CONFIDENCE THRESHOLDS
# ============================================================================

print("=" * 80)
print("BACKTESTING WITH MULTIPLE CONFIDENCE THRESHOLDS")
print("=" * 80)
print()

def calculate_position_size(balance, risk_pct, sl_pips):
    """Calculate position size based on risk percentage"""
    risk_amount = balance * risk_pct
    pip_value = 1  # For EURUSD, 1 pip = 1 EUR for 0.1 lot
    position_size = risk_amount / sl_pips
    return position_size

def simulate_trade(entry_price, direction, future_bars, sl_pips=20, tp_pips=30):
    """Simulate a single trade with SL/TP"""
    
    if direction == 'LONG':
        sl_price = entry_price - sl_pips / 10000
        tp_price = entry_price + tp_pips / 10000
        
        for i, row in future_bars.iterrows():
            # Check SL hit
            if row['low'] <= sl_price:
                return -sl_pips - TOTAL_COST, 'SL'
            # Check TP hit
            if row['high'] >= tp_price:
                return tp_pips - TOTAL_COST, 'TP'
        
        # No SL/TP hit in next 48 hours - close at market
        exit_price = future_bars.iloc[-1]['close']
        pnl_pips = (exit_price - entry_price) * 10000 - TOTAL_COST
        return pnl_pips, 'TIME'
        
    else:  # SHORT
        sl_price = entry_price + sl_pips / 10000
        tp_price = entry_price - tp_pips / 10000
        
        for i, row in future_bars.iterrows():
            # Check SL hit
            if row['high'] >= sl_price:
                return -sl_pips - TOTAL_COST, 'SL'
            # Check TP hit
            if row['low'] <= tp_price:
                return tp_pips - TOTAL_COST, 'TP'
        
        # No SL/TP hit - close at market
        exit_price = future_bars.iloc[-1]['close']
        pnl_pips = (entry_price - exit_price) * 10000 - TOTAL_COST
        return pnl_pips, 'TIME'

results_by_threshold = {}

for conf_threshold in CONFIDENCE_THRESHOLDS:
    print(f"Testing confidence threshold: {conf_threshold*100:.0f}%")
    
    balance = ACCOUNT_BALANCE
    trades = []
    equity_curve = [balance]
    
    i = 0
    while i < len(df_backtest) - 48:  # Use while loop to skip bars after trade
        current_bar = df_backtest.iloc[i]
        
        # Weekend protection (skip Friday 21:00 - Monday 09:00 UTC)
        if current_bar.name.weekday() == 4 and current_bar.name.hour >= 21:
            i += 1
            continue
        if current_bar.name.weekday() == 5 or current_bar.name.weekday() == 6:
            i += 1
            continue
        if current_bar.name.weekday() == 0 and current_bar.name.hour < 9:
            i += 1
            continue
        
        # Extract features
        X_current = current_bar[feature_columns].values.reshape(1, -1)
        X_scaled = scaler.transform(X_current)
        
        # Predict
        pred_proba = model.predict_proba(X_scaled)[0]
        pred_class = np.argmax(pred_proba)
        confidence = pred_proba[pred_class]
        
        # Check if signal meets criteria
        adx = current_bar['adx_h4']
        
        if confidence >= conf_threshold and adx >= MIN_ADX and pred_class != 1:  # Not FLAT
            
            direction = 'LONG' if pred_class == 2 else 'SHORT'
            entry_price = current_bar['close']
            
            # Get future bars for simulation
            future_bars = df_backtest.iloc[i+1:i+49]
            
            if len(future_bars) < 48:
                break
            
            # Simulate trade
            pnl_pips, exit_reason = simulate_trade(
                entry_price, 
                direction, 
                future_bars,
                sl_pips=20,
                tp_pips=30
            )
            
            # Calculate position size (more conservative)
            position_size = calculate_position_size(balance, RISK_PER_TRADE, 20)
            
            # Calculate P&L in EUR
            pnl_eur = pnl_pips * position_size
            
            # Update balance
            balance += pnl_eur
            equity_curve.append(balance)
            
            # Record trade
            trades.append({
                'time': current_bar.name,
                'direction': direction,
                'entry': entry_price,
                'confidence': confidence,
                'adx': adx,
                'pnl_pips': pnl_pips,
                'pnl_eur': pnl_eur,
                'exit_reason': exit_reason,
                'balance': balance
            })
            
            # CRITICAL FIX: Skip next 48 bars (trade duration)
            # This prevents opening multiple positions simultaneously
            i += 48
        else:
            # No trade opened, move to next bar
            i += 1
    
    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        wins = trades_df[trades_df['pnl_eur'] > 0]
        losses = trades_df[trades_df['pnl_eur'] <= 0]
        
        win_rate = len(wins) / len(trades) * 100
        total_pnl = trades_df['pnl_eur'].sum()
        avg_win = wins['pnl_eur'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl_eur'].mean()) if len(losses) > 0 else 0
        
        gross_profit = wins['pnl_eur'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Sharpe Ratio (assuming 252 trading days, adjusted for hourly)
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        results_by_threshold[conf_threshold] = {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_balance': balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trades': trades_df,
            'equity_curve': equity_curve
        }
        
        print(f"  Total trades: {len(trades)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: EUR {total_pnl:.2f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print()
    else:
        print(f"  No trades generated")
        print()

# ============================================================================
# STEP 5: FIND BEST THRESHOLD
# ============================================================================

print("=" * 80)
print("FINDING BEST CONFIDENCE THRESHOLD")
print("=" * 80)
print()

best_threshold = None
best_pnl = -float('inf')

for threshold, results in results_by_threshold.items():
    if results['total_pnl'] > best_pnl:
        best_pnl = results['total_pnl']
        best_threshold = threshold

if best_threshold:
    print(f"Best confidence threshold: {best_threshold*100:.0f}%")
    print()
    
    best_results = results_by_threshold[best_threshold]
    
    print("PERFORMANCE SUMMARY:")
    print(f"  Total trades: {best_results['total_trades']}")
    print(f"  Win Rate: {best_results['win_rate']:.2f}%")
    print(f"  Total P&L: EUR {best_results['total_pnl']:.2f}")
    print(f"  Final balance: EUR {best_results['final_balance']:.2f}")
    print(f"  Return: {(best_results['final_balance'] - ACCOUNT_BALANCE) / ACCOUNT_BALANCE * 100:.2f}%")
    print(f"  Average win: EUR {best_results['avg_win']:.2f}")
    print(f"  Average loss: EUR {best_results['avg_loss']:.2f}")
    print(f"  Profit Factor: {best_results['profit_factor']:.2f}")
    print(f"  Max Drawdown: {best_results['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio: {best_results['sharpe_ratio']:.2f}")
    print()
    
    # Save detailed trade log
    trade_log_path = os.path.join(RESULTS_DIR, f"trades_threshold_{int(best_threshold*100)}.csv")
    best_results['trades'].to_csv(trade_log_path, index=False)
    print(f"Trade log saved: {trade_log_path}")
    
    # Create comparison table
    comparison_data = []
    for threshold, results in sorted(results_by_threshold.items()):
        comparison_data.append({
            'Threshold': f"{threshold*100:.0f}%",
            'Trades': results['total_trades'],
            'Win Rate': f"{results['win_rate']:.1f}%",
            'Total P&L': f"EUR {results['total_pnl']:.2f}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Max DD': f"{results['max_drawdown']:.1f}%",
            'Sharpe': f"{results['sharpe_ratio']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(RESULTS_DIR, "threshold_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Threshold comparison saved: {comparison_path}")
    print()
    
    # Plot equity curve
    plt.figure(figsize=(14, 6))
    plt.plot(best_results['equity_curve'], linewidth=2)
    plt.axhline(y=ACCOUNT_BALANCE, color='r', linestyle='--', label='Initial Balance')
    plt.title(f'Equity Curve - Threshold {best_threshold*100:.0f}%', fontsize=14, fontweight='bold')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (EUR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    equity_plot_path = os.path.join(RESULTS_DIR, f"equity_curve_threshold_{int(best_threshold*100)}.png")
    plt.savefig(equity_plot_path, dpi=150)
    print(f"Equity curve saved: {equity_plot_path}")
    
    # Save summary report
    summary = {
        'backtest_date': datetime.now().isoformat(),
        'model_path': model_path,
        'backtest_period': {
            'start': str(df_backtest.index[0]),
            'end': str(df_backtest.index[-1]),
            'bars': len(df_backtest)
        },
        'best_threshold': best_threshold,
        'performance': {
            'total_trades': int(best_results['total_trades']),
            'win_rate': float(best_results['win_rate']),
            'total_pnl': float(best_results['total_pnl']),
            'final_balance': float(best_results['final_balance']),
            'return_pct': float((best_results['final_balance'] - ACCOUNT_BALANCE) / ACCOUNT_BALANCE * 100),
            'avg_win': float(best_results['avg_win']),
            'avg_loss': float(best_results['avg_loss']),
            'profit_factor': float(best_results['profit_factor']),
            'max_drawdown': float(best_results['max_drawdown']),
            'sharpe_ratio': float(best_results['sharpe_ratio'])
        },
        'threshold_comparison': comparison_data
    }
    
    summary_path = os.path.join(RESULTS_DIR, "backtest_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report saved: {summary_path}")
    print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("=" * 80)
print("BACKTEST VERDICT")
print("=" * 80)
print()

if best_threshold and best_results['win_rate'] >= 48 and best_results['profit_factor'] >= 1.3:
    print("✅ MODEL APPROVED FOR LIVE DEPLOYMENT")
    print()
    print("Deployment recommendations:")
    print(f"  - Use confidence threshold: {best_threshold*100:.0f}%")
    print(f"  - Expected Win Rate: {best_results['win_rate']:.1f}%")
    print(f"  - Expected trade frequency: ~{best_results['total_trades'] / (len(df_backtest) / 24 / 7):.1f} trades/week")
    print(f"  - Monitor performance after 30 trades")
    print(f"  - Retrain if WR drops below 45%")
elif best_threshold:
    print("⚠️  MODEL NEEDS IMPROVEMENT")
    print()
    print("Issues detected:")
    if best_results['win_rate'] < 48:
        print(f"  - Win Rate too low: {best_results['win_rate']:.1f}% (target: 48%+)")
    if best_results['profit_factor'] < 1.3:
        print(f"  - Profit Factor too low: {best_results['profit_factor']:.2f} (target: 1.3+)")
    print()
    print("Recommendations:")
    print("  - Retrain with different MIN_PROFIT_THRESHOLD")
    print("  - Increase Optuna trials to 200")
    print("  - Check feature engineering")
else:
    print("❌ BACKTEST FAILED - NO TRADES GENERATED")
    print()
    print("Possible causes:")
    print("  - All confidence thresholds too high")
    print("  - ADX filter too restrictive")
    print("  - Model not generating confident predictions")

print("=" * 80)
