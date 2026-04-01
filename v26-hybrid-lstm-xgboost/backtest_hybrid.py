# backtest_v26.py
# BACKTEST WALK-FORWARD: Test modelli V26 su dati Out-of-Sample
# Simula trade reali con risk management identico al bot live

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
except:
    print("❌ TensorFlow non trovato")
    sys.exit(1)

try:
    import xgboost as xgb
except:
    print("❌ XGBoost non trovato")
    sys.exit(1)

# ============================================
# CONFIGURAZIONE
# ============================================
DATA_DIR = "Dati_Fresh_2026"
MODELS_DIR = "cervelli_IA_2026"
SYMBOLS = ["GBPUSD", "AUDUSD", "EURUSD", "USDJPY"]

# Parametri identici al bot live
CAPITAL_ALLOCATION = {'LSTM': 400.0, 'XGB': 520.0}
RISK_PER_TRADE = {'LSTM': 0.005, 'XGB': 0.006}
SEQ_LEN = 24

MIN_CONFIDENCE = {
    'LSTM': {'GBPUSD': 0.58, 'AUDUSD': 0.58, 'EURUSD': 0.65, 'USDJPY': 0.62},
    'XGB': {'GBPUSD': 0.62, 'AUDUSD': 0.62, 'EURUSD': 0.68, 'USDJPY': 0.65}
}

ADX_THRESHOLD = {'GBPUSD': 18, 'AUDUSD': 18, 'EURUSD': 22, 'USDJPY': 20}

# Risk Management
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.5
SPREAD_PIPS = {'GBPUSD': 1.5, 'AUDUSD': 1.2, 'EURUSD': 1.0, 'USDJPY': 1.5}

# ============================================
# INDICATORI (Identici al training)
# ============================================
def calculate_adx(df, period=14):
    df = df.copy()
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0.0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0.0)
    df['tr_smooth'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100.0 * (df['plus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    minus_di = 100.0 * (df['minus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_features(df):
    df = df.copy()
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    elif 'volume' not in df.columns:
        df['volume'] = 0.0
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['SMA_200'] = df['close'].rolling(200).mean()
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean() + 1e-9)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    df['ADX_14'] = calculate_adx(df)
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['BBM_20_2.0_2.0'] = sma20
    df['BBU_20_2.0_2.0'] = sma20 + 2.0 * std20
    df['BBL_20_2.0_2.0'] = sma20 - 2.0 * std20
    df['BBB_20_2.0_2.0'] = df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']
    df['BBP_20_2.0_2.0'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBB_20_2.0_2.0'] + 1e-9)
    
    prev = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev).abs(),
        (df['low'] - prev).abs()
    ], axis=1).max(axis=1)
    df['ATRr_14'] = tr.rolling(14).mean() / (df['close'] + 1e-9)
    df['ATR_14'] = tr.rolling(14).mean()
    
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    k = 100 * (df['close'] - low14) / (high14 - low14 + 1e-9)
    df['STOCHk_14_3_3'] = k.rolling(3).mean()
    df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(3).mean()
    df['STOCHh_14_3_3'] = df['STOCHk_14_3_3'] - df['STOCHd_14_3_3']
    
    df['dayofweek'] = df.index.dayofweek
    df['hourofday'] = df.index.hour
    if 'spread' not in df.columns:
        df['spread'] = 0.0
    
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def get_pip_size(symbol):
    return 0.01 if "JPY" in symbol.upper() else 0.0001

# ============================================
# SIMULAZIONE TRADE
# ============================================
class Trade:
    def __init__(self, symbol, model_type, direction, entry_price, sl, tp, volume, entry_time):
        self.symbol = symbol
        self.model_type = model_type
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.volume = volume
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pips = 0.0
        self.outcome = None  # 'WIN_TP', 'LOSS_SL', 'TIMEOUT'

def simulate_trade_outcome(trade, future_bars, max_duration_bars=100):
    """
    Simula l'esito del trade nei bar successivi
    Returns: updated trade object
    """
    pip = get_pip_size(trade.symbol)
    
    for i, row in future_bars.iterrows():
        # Check SL/TP hit
        if trade.direction == "LONG":
            if row['low'] <= trade.sl:
                trade.exit_price = trade.sl
                trade.exit_time = row.name
                trade.outcome = 'LOSS_SL'
                break
            elif row['high'] >= trade.tp:
                trade.exit_price = trade.tp
                trade.exit_time = row.name
                trade.outcome = 'WIN_TP'
                break
        else:  # SHORT
            if row['high'] >= trade.sl:
                trade.exit_price = trade.sl
                trade.exit_time = row.name
                trade.outcome = 'LOSS_SL'
                break
            elif row['low'] <= trade.tp:
                trade.exit_price = trade.tp
                trade.exit_time = row.name
                trade.outcome = 'WIN_TP'
                break
        
        # Check max duration
        if len(future_bars.loc[:row.name]) >= max_duration_bars:
            trade.exit_price = row['close']
            trade.exit_time = row.name
            trade.outcome = 'TIMEOUT'
            break
    
    # Calculate P&L
    if trade.exit_price is not None:
        if trade.direction == "LONG":
            trade.pips = (trade.exit_price - trade.entry_price) / pip
        else:
            trade.pips = (trade.entry_price - trade.exit_price) / pip
        
        # Simplified P&L (ignoring commission/swap)
        contract_size = 100000.0
        trade.pnl = trade.pips * pip * contract_size * trade.volume
    
    return trade

# ============================================
# BACKTEST ENGINE
# ============================================
def backtest_symbol(symbol, lstm_model, xgb_model, scaler, test_start_date):
    """
    Backtesta un simbolo da test_start_date in poi
    """
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST: {symbol}")
    print(f"{'='*60}")
    
    # Carica dati
    csv_path = os.path.join(DATA_DIR, f"{symbol}_M5.csv")
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    print(f"📅 Dati totali: {df.index.min()} → {df.index.max()}")
    
    # Calcola features
    df_feat = compute_features(df)
    
    # Split train/test
    df_test = df_feat[df_feat.index >= test_start_date].copy()
    print(f"🧪 Test period: {df_test.index.min()} → {df_test.index.max()}")
    print(f"   Test bars: {len(df_test)}")
    
    if len(df_test) < SEQ_LEN + 100:
        print(f"⚠️  Test period troppo corto, skip")
        return None
    
    # Feature columns
    feature_cols = [
        'spread', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',
        'ATRr_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHh_14_3_3',
        'SMA_50', 'SMA_200', 'dayofweek', 'hourofday',
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    trades = []
    pip = get_pip_size(symbol)
    spread_pips = SPREAD_PIPS[symbol]
    
    # Walk-forward simulation
    for i in range(SEQ_LEN, len(df_test) - 100):  # Lascia 100 bar per trade outcome
        current_idx = df_test.index[i]
        
        # Skip orari notturni (semplificato)
        if current_idx.hour >= 23 or current_idx.hour < 1:
            continue
        
        # Prepara sequenza
        seq_data = df_test.iloc[i-SEQ_LEN:i][feature_cols].values
        if np.isnan(seq_data).any():
            continue
        
        seq_scaled = scaler.transform(seq_data)
        
        # Metriche correnti
        adx = df_test.iloc[i]['ADX_14']
        rsi = df_test.iloc[i]['RSI_14']
        atr = df_test.iloc[i]['ATR_14']
        entry_price = df_test.iloc[i]['close']
        
        # Check ADX
        if adx < ADX_THRESHOLD[symbol]:
            continue
        
        # Predizioni
        # LSTM
        seq_lstm = seq_scaled.reshape(1, SEQ_LEN, -1)
        pred_lstm = lstm_model.predict(seq_lstm, verbose=0)[0]
        conf_lstm = float(np.max(pred_lstm))
        signal_lstm = {0: "SHORT", 1: "FLAT", 2: "LONG"}[int(np.argmax(pred_lstm))]
        
        # XGB
        last_row = seq_scaled[-1].reshape(1, -1)
        dmatrix = xgb.DMatrix(last_row)
        pred_xgb = xgb_model.predict(dmatrix)[0]
        conf_xgb = float(np.max(pred_xgb))
        signal_xgb = {0: "SHORT", 1: "FLAT", 2: "LONG"}[int(np.argmax(pred_xgb))]
        
        # Check confidenza e crea trade
        for model_type, signal, conf in [('LSTM', signal_lstm, conf_lstm), ('XGB', signal_xgb, conf_xgb)]:
            if signal == "FLAT":
                continue
            
            min_conf = MIN_CONFIDENCE[model_type][symbol]
            if conf < min_conf:
                continue
            
            # Calcola SL/TP
            if signal == "LONG":
                entry_adj = entry_price + spread_pips * pip
                sl = entry_adj - (atr * SL_ATR_MULT)
                tp = entry_adj + (atr * TP_ATR_MULT)
                direction = "LONG"
            else:
                entry_adj = entry_price - spread_pips * pip
                sl = entry_adj + (atr * SL_ATR_MULT)
                tp = entry_adj - (atr * TP_ATR_MULT)
                direction = "SHORT"
            
            # Volume (semplificato, fixed)
            capital = CAPITAL_ALLOCATION[model_type]
            risk_pct = RISK_PER_TRADE[model_type]
            pips_at_risk = abs(entry_adj - sl) / pip
            risk_amount = capital * risk_pct
            contract_size = 100000.0
            loss_per_lot = pips_at_risk * pip * contract_size
            volume = risk_amount / loss_per_lot if loss_per_lot > 0 else 0.01
            volume = min(max(0.01, round(volume, 2)), 0.10)
            
            # Crea trade
            trade = Trade(symbol, model_type, direction, entry_adj, sl, tp, volume, current_idx)
            
            # Simula outcome
            future_bars = df_test.iloc[i+1:i+101]
            trade = simulate_trade_outcome(trade, future_bars)
            
            if trade.outcome is not None:
                trades.append(trade)
                
                # Skip prossimi N bar (simula posizione aperta)
                # i += 10  # Non possiamo modificare i nel loop, quindi gestiamo con logic
    
    print(f"\n✅ Simulazione completata: {len(trades)} trade")
    return trades

# ============================================
# ANALISI RISULTATI
# ============================================
def analyze_results(all_trades):
    """Analizza e stampa metriche"""
    
    if not all_trades:
        print("\n❌ Nessun trade eseguito nel backtest")
        return
    
    df_trades = pd.DataFrame([{
        'symbol': t.symbol,
        'model': t.model_type,
        'direction': t.direction,
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'pnl': t.pnl,
        'pips': t.pips,
        'outcome': t.outcome,
        'volume': t.volume
    } for t in all_trades])
    
    print(f"\n{'='*70}")
    print("📊 RISULTATI BACKTEST V26")
    print(f"{'='*70}")
    
    # Globali
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['pnl'] > 0])
    losses = len(df_trades[df_trades['pnl'] < 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = df_trades['pnl'].sum()
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
    
    # Drawdown
    df_trades = df_trades.sort_values('entry_time')
    df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
    running_max = df_trades['cumulative_pnl'].cummax()
    drawdown = df_trades['cumulative_pnl'] - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    print(f"\n🎯 METRICHE GLOBALI")
    print(f"{'─'*70}")
    print(f"Total Trades:      {total_trades}")
    print(f"Wins / Losses:     {wins} / {losses}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"\nTotal P&L:         €{total_pnl:.2f}")
    print(f"Gross Profit:      €{gross_profit:.2f}")
    print(f"Gross Loss:        €{gross_loss:.2f}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"\nAvg Win:           €{avg_win:.2f}")
    print(f"Avg Loss:          €{avg_loss:.2f}")
    print(f"Max Drawdown:      €{max_drawdown:.2f}")
    
    # Per Modello
    print(f"\n📊 PER MODELLO")
    print(f"{'─'*70}")
    for model in ['LSTM', 'XGB']:
        df_model = df_trades[df_trades['model'] == model]
        if len(df_model) == 0:
            continue
        
        trades_model = len(df_model)
        wins_model = len(df_model[df_model['pnl'] > 0])
        wr_model = (wins_model / trades_model * 100) if trades_model > 0 else 0
        pnl_model = df_model['pnl'].sum()
        
        gp = df_model[df_model['pnl'] > 0]['pnl'].sum()
        gl = abs(df_model[df_model['pnl'] < 0]['pnl'].sum())
        pf = (gp / gl) if gl > 0 else 0
        
        print(f"\n{model}:")
        print(f"  Trades: {trades_model} | Win Rate: {wr_model:.2f}%")
        print(f"  P&L: €{pnl_model:.2f} | PF: {pf:.2f}")
    
    # Per Simbolo
    print(f"\n📊 PER SIMBOLO")
    print(f"{'─'*70}")
    for symbol in SYMBOLS:
        df_sym = df_trades[df_trades['symbol'] == symbol]
        if len(df_sym) == 0:
            continue
        
        trades_sym = len(df_sym)
        wins_sym = len(df_sym[df_sym['pnl'] > 0])
        wr_sym = (wins_sym / trades_sym * 100) if trades_sym > 0 else 0
        pnl_sym = df_sym['pnl'].sum()
        
        print(f"{symbol}: {trades_sym} trades | WR: {wr_sym:.2f}% | P&L: €{pnl_sym:.2f}")
    
    # Verdetto
    print(f"\n{'='*70}")
    print("🎯 VERDETTO")
    print(f"{'='*70}")
    
    if win_rate >= 50 and profit_factor >= 1.3:
        print("✅ ECCELLENTE - Bot pronto per demo")
    elif win_rate >= 48 and profit_factor >= 1.1:
        print("✅ BUONO - Bot pronto per demo con monitoring")
    elif win_rate >= 45 and profit_factor >= 1.0:
        print("⚠️  ACCETTABILE - Demo consigliata ma prudenza")
    else:
        print("❌ INSUFFICIENTE - NON deploy, serve ottimizzazione")
    
    print(f"\nTarget minimi:")
    print(f"  Win Rate: ≥48% (attuale: {win_rate:.2f}%)")
    print(f"  Profit Factor: ≥1.1 (attuale: {profit_factor:.2f})")
    print(f"  Max Drawdown: <€50 (attuale: €{max_drawdown:.2f})")
    
    return df_trades

# ============================================
# MAIN
# ============================================
def main():
    print("="*70)
    print("🧪 BACKTEST V26 HYBRID - Walk-Forward Validation")
    print("="*70)
    
    # Determina test period (ultimi 2 mesi)
    end_date = datetime(2026, 1, 28)
    test_start = end_date - timedelta(days=60)
    
    print(f"\n📅 Test Period: {test_start.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   (Ultimi ~2 mesi di dati out-of-sample)")
    
    all_trades = []
    
    for symbol in SYMBOLS:
        try:
            # Carica modelli
            lstm_path = os.path.join(MODELS_DIR, f"{symbol}_LSTM.h5")
            xgb_path = os.path.join(MODELS_DIR, f"{symbol}_XGB.json")
            scaler_path = os.path.join(MODELS_DIR, f"{symbol}_SCALER.joblib")
            
            lstm_model = load_model(lstm_path)
            xgb_model = xgb.Booster()
            xgb_model.load_model(xgb_path)
            scaler = joblib.load(scaler_path)
            
            # Backtest
            trades = backtest_symbol(symbol, lstm_model, xgb_model, scaler, test_start)
            
            if trades:
                all_trades.extend(trades)
        
        except Exception as e:
            print(f"\n❌ Errore su {symbol}: {e}")
            continue
    
    # Analisi finale
    if all_trades:
        df_results = analyze_results(all_trades)
        
        # Salva risultati
        output_path = "backtest_results_v26.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n💾 Risultati salvati: {output_path}")
    else:
        print("\n❌ Nessun trade nel backtest")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
