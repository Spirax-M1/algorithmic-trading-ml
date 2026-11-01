raise SystemExit("Historical experiment disabled: see docs/audit.md; run python -m trading_research --demo")

# bot_HYBRID_v26.py
# STRATEGIA IBRIDA: LSTM (Trend) + XGBoost (Pattern)
# Capitale: €920 → Pool LSTM €400 + Pool XGB €520
# Safety: Margine 30%, Max 3 posizioni globali, Coordinamento pool

import os
import sys

# Fix encoding Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import math
import threading
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import requests

warnings.filterwarnings('ignore')

try:
    import MetaTrader5 as mt5
except:
    mt5 = None
    print("❌ MetaTrader5 non trovato")

try:
    from tensorflow.keras.models import load_model
except:
    load_model = None
    print("❌ TensorFlow non trovato")

try:
    import xgboost as xgb
except:
    xgb = None
    print("❌ XGBoost non trovato")

# ============================================
# TELEGRAM CONFIGURATION
# ============================================
# Replace with your own Telegram bot token and chat ID
# Get token from @BotFather and chat_id from @userinfobot on Telegram
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

def send_telegram(msg: str, silent: bool = False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_notification": silent
        }
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ============================================
# CONFIGURAZIONE V26 HYBRID
# ============================================
MODELS_DIR = "cervelli_IA_2026"
SYMBOLS = ["GBPUSD", "AUDUSD", "EURUSD", "USDJPY"]

# POOL VIRTUALI (€920 totale)
CAPITAL_ALLOCATION = {
    'LSTM': 400.0,    # Pool per trend medio-termine
    'XGB': 520.0      # Pool per pattern veloci
}

# Risk per pool (calcolato sul capitale allocato)
RISK_PER_TRADE = {
    'LSTM': 0.005,    # 0.5% di €400 = €2
    'XGB': 0.006      # 0.6% di €520 = €3.12
}

# Confidenza minima per modello
MIN_CONFIDENCE = {
    'LSTM': {
        'GBPUSD': 0.58,
        'AUDUSD': 0.58,
        'EURUSD': 0.65,
        'USDJPY': 0.62
    },
    'XGB': {
        'GBPUSD': 0.62,
        'AUDUSD': 0.62,
        'EURUSD': 0.68,
        'USDJPY': 0.65
    }
}

# Filtri ADX
ADX_THRESHOLD = {
    'GBPUSD': 18,
    'AUDUSD': 18,
    'EURUSD': 22,
    'USDJPY': 20
}

# ============================================
# SAFETY LAYERS (CRITICI)
# ============================================
MAX_GLOBAL_POSITIONS = 3           # Max 3 pos totali
MAX_POSITIONS_PER_SYMBOL = 2       # Max 2 per simbolo (1 LSTM + 1 XGB)
MAX_POSITIONS_PER_MODEL_SYMBOL = 1 # Max 1 dello stesso modello/simbolo

MIN_MARGIN_FREE_PCT = 0.30         # 30% margine libero minimo

MAX_TRADES_PER_DAY = 8             # Limite giornaliero
DAILY_LOSS_LIMIT = -25.0           # Circuit breaker a -€25

# ATR & SL/TP
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.5
SPREAD_MAX_PIPS = 2.5

# Trailing
TRAILING_START_ATR = 1.8
TRAILING_STEP_ATR = 0.8

# Timeframes
TIMEFRAME_M = mt5.TIMEFRAME_M5 if mt5 else 0
TIMEFRAME_HTF = mt5.TIMEFRAME_H1 if mt5 else 0

# Orari
NO_TRADE_START_HOUR = 23
NO_TRADE_END_HOUR = 1
HOUR_STOP_FRIDAY = 20

# Magic Numbers (Base + Offset)
MAGIC_BASE_LSTM = 202600
MAGIC_BASE_XGB = 202700

# ============================================
# STATO GLOBALE (Thread-Safe)
# ============================================
daily_stats = {
    'profit': 0.0,
    'trades': 0,
    'date': datetime.now().date()
}
stats_lock = threading.Lock()

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "bot_v26_hybrid.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================================
# GESTIONE LIMITI
# ============================================
def reset_daily_stats_if_needed():
    global daily_stats
    with stats_lock:
        today = datetime.now().date()
        if daily_stats['date'] != today:
            daily_stats = {
                'profit': 0.0,
                'trades': 0,
                'date': today
            }
            logging.info("♻️ Daily stats reset")

def can_open_trade(model_type: str, symbol: str) -> tuple:
    """
    Verifica TUTTI i safety layers prima di aprire
    Returns: (bool, str reason)
    """
    with stats_lock:
        reset_daily_stats_if_needed()
        
        # 1. Daily Loss
        if daily_stats['profit'] <= DAILY_LOSS_LIMIT:
            return False, "Daily loss limit"
        
        # 2. Daily Trades
        if daily_stats['trades'] >= MAX_TRADES_PER_DAY:
            return False, "Max trades per day"
    
    # 3. Margine Libero (CRITICO)
    acct = mt5.account_info()
    if not acct:
        return False, "No account info"
    
    margin_free_pct = acct.margin_free / acct.equity
    if margin_free_pct < MIN_MARGIN_FREE_PCT:
        return False, f"Margine basso ({margin_free_pct:.1%} < 30%)"
    
    # 4. Posizioni Globali
    all_pos = mt5.positions_get()
    if all_pos and len(all_pos) >= MAX_GLOBAL_POSITIONS:
        return False, f"Max global positions ({MAX_GLOBAL_POSITIONS})"
    
    # 5. Posizioni per Simbolo
    symbol_pos = [p for p in all_pos if p.symbol == symbol] if all_pos else []
    if len(symbol_pos) >= MAX_POSITIONS_PER_SYMBOL:
        return False, f"Max positions on {symbol}"
    
    # 6. Posizioni stesso modello/simbolo
    if model_type == 'LSTM':
        magic_range = range(MAGIC_BASE_LSTM, MAGIC_BASE_LSTM + 10)
    else:
        magic_range = range(MAGIC_BASE_XGB, MAGIC_BASE_XGB + 10)
    
    model_symbol_pos = [p for p in symbol_pos if p.magic in magic_range]
    if len(model_symbol_pos) >= MAX_POSITIONS_PER_MODEL_SYMBOL:
        return False, f"{model_type} already on {symbol}"
    
    return True, ""

def register_trade_open():
    with stats_lock:
        daily_stats['trades'] += 1

def register_trade_close(pnl: float):
    with stats_lock:
        reset_daily_stats_if_needed()
        daily_stats['profit'] += pnl

# ============================================
# UTILITIES
# ============================================
def get_pip_size(symbol: str):
    return 0.01 if "JPY" in symbol.upper() else 0.0001

def find_artifacts(symbol: str):
    """Trova modelli LSTM, XGB e scaler"""
    lstm_path = os.path.join(MODELS_DIR, f"{symbol}_LSTM.h5")
    xgb_path = os.path.join(MODELS_DIR, f"{symbol}_XGB.json")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_SCALER.joblib")
    
    if not all(os.path.exists(p) for p in [lstm_path, xgb_path, scaler_path]):
        return None, None, None
    
    return lstm_path, xgb_path, scaler_path

def load_models(symbol: str):
    """Carica LSTM, XGBoost e Scaler"""
    lstm_path, xgb_path, scaler_path = find_artifacts(symbol)
    
    if not all([lstm_path, xgb_path, scaler_path]):
        raise FileNotFoundError(f"Modelli mancanti per {symbol}")
    
    lstm_model = load_model(lstm_path)
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_path)
    scaler = joblib.load(scaler_path)
    
    return lstm_model, xgb_model, scaler

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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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

def compute_atr_h1(symbol: str):
    try:
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_HTF, 0, 200)
        if rates is None or len(rates) < 20:
            return 0.0008
        df = pd.DataFrame(rates)
        prev = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev).abs(),
            (df['low'] - prev).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        return max(atr, 0.0008)
    except:
        return 0.0008

def check_macro_trend(symbol: str):
    try:
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_HTF, 0, 300)
        if rates is None or len(rates) < 200:
            return "NEUTRAL"
        df = pd.DataFrame(rates)
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        tick = mt5.symbol_info_tick(symbol)
        price = (tick.bid + tick.ask) / 2.0 if tick else df['close'].iloc[-1]
        if price > sma200:
            return "BULLISH"
        if price < sma200:
            return "BEARISH"
        return "FLAT"
    except:
        return "NEUTRAL"

# ============================================
# RISK MANAGEMENT
# ============================================
def compute_volume_by_risk(model_type: str, symbol: str, entry_price: float, sl_price: float):
    """
    Calcola lot size basato sul POOL VIRTUALE
    """
    acct = mt5.account_info()
    if not acct:
        return 0.0, "no_account"
    
    # Capitale allocato al modello
    allocated_capital = CAPITAL_ALLOCATION[model_type]
    risk_pct = RISK_PER_TRADE[model_type]
    
    pip = get_pip_size(symbol)
    pips_at_risk = abs(entry_price - sl_price) / pip
    
    si = mt5.symbol_info(symbol)
    contract_size = float(si.trade_contract_size) if si else 100000.0
    
    loss_per_lot = pips_at_risk * pip * contract_size
    if loss_per_lot <= 0:
        return 0.0, "zero_loss"
    
    risk_amount = allocated_capital * risk_pct
    ideal_lots = risk_amount / loss_per_lot
    
    final_lot = math.floor(ideal_lots / 0.01) * 0.01
    final_lot = max(0.01, min(final_lot, 0.10))
    
    return final_lot, None

def safe_order_send(req: dict):
    try:
        res = mt5.order_send(req)
        if res and res.retcode == 10009:
            return res
        logging.warning(f"Order failed: {res.comment if res else 'None'}")
        return None
    except Exception as e:
        logging.error(f"Order error: {e}")
        return None

# ============================================
# SYMBOL WORKER (Dual Model)
# ============================================
class HybridWorker(threading.Thread):
    def __init__(self, symbol, lstm_model, xgb_model, scaler, symbol_idx):
        super().__init__(daemon=True)
        self.symbol = symbol
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.magic_lstm = MAGIC_BASE_LSTM + symbol_idx
        self.magic_xgb = MAGIC_BASE_XGB + symbol_idx
        self.last_candle_time = None
        self.running = True
        self.seq_len = 24
        
        self.cached_atr = None
        self.last_atr_time = 0
        
        self.open_tickets = {'LSTM': set(), 'XGB': set()}
    
    def _get_cached_atr(self):
        now = time.time()
        if self.cached_atr is None or (now - self.last_atr_time > 900):
            self.cached_atr = compute_atr_h1(self.symbol)
            self.last_atr_time = now
        return self.cached_atr
    
    def _manage_trailing(self):
        """Trailing stop per entrambi i modelli"""
        try:
            all_pos = mt5.positions_get(symbol=self.symbol)
            if not all_pos:
                return
            
            lstm_pos = [p for p in all_pos if p.magic == self.magic_lstm]
            xgb_pos = [p for p in all_pos if p.magic == self.magic_xgb]
            
            # Track chiusure
            current_lstm = {p.ticket for p in lstm_pos}
            current_xgb = {p.ticket for p in xgb_pos}
            
            closed_lstm = self.open_tickets['LSTM'] - current_lstm
            closed_xgb = self.open_tickets['XGB'] - current_xgb
            
            for ticket in closed_lstm:
                self._notify_close(ticket, 'LSTM')
            for ticket in closed_xgb:
                self._notify_close(ticket, 'XGB')
            
            self.open_tickets['LSTM'] = current_lstm
            self.open_tickets['XGB'] = current_xgb
            
            # Trailing
            atr = self._get_cached_atr()
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return
            
            activation = TRAILING_START_ATR * atr
            step = TRAILING_STEP_ATR * atr
            
            for pos in (lstm_pos + xgb_pos):
                if pos.type == mt5.POSITION_TYPE_BUY:
                    if (tick.bid - pos.price_open) > activation:
                        new_sl = tick.bid - step
                        if new_sl > pos.sl:
                            mt5.order_send({
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": self.symbol,
                                "sl": new_sl,
                                "tp": pos.tp
                            })
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    if (pos.price_open - tick.ask) > activation:
                        new_sl = tick.ask + step
                        if pos.sl == 0 or new_sl < pos.sl:
                            mt5.order_send({
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": self.symbol,
                                "sl": new_sl,
                                "tp": pos.tp
                            })
        except Exception as e:
            logging.error(f"Trailing error: {e}")
    
    def _notify_close(self, ticket: int, model_type: str):
        """Notifica chiusura trade"""
        try:
            from_time = datetime.now() - timedelta(days=1)
            deals = mt5.history_deals_get(from_time, datetime.now(), position=ticket)
            
            if not deals:
                return
            
            total_pnl = sum(d.profit + d.commission + d.swap for d in deals)
            entry = deals[0]
            exit = deals[-1]
            
            pip = get_pip_size(self.symbol)
            if entry.type == mt5.ORDER_TYPE_BUY:
                pips = (exit.price - entry.price) / pip
                direction = "BUY"
            else:
                pips = (entry.price - exit.price) / pip
                direction = "SELL"
            
            emoji = "✅" if total_pnl >= 0 else "❌"
            
            msg = f"""{emoji} <b>{model_type} CHIUSO</b>

💰 P&L: €{total_pnl:.2f}
📊 {self.symbol} {direction}
📈 Pips: {pips:+.1f}

🧠 Pool: {model_type}
"""
            send_telegram(msg)
            register_trade_close(total_pnl)
            
        except Exception as e:
            logging.error(f"Notify error: {e}")
    
    def _try_open_trade(self, model_type: str, signal: str, conf: float, adx: float, rsi: float, macro: str):
        """Tenta apertura con safety checks"""
        
        # Check permessi
        can_trade, reason = can_open_trade(model_type, self.symbol)
        if not can_trade:
            logging.debug(f"[{self.symbol}] {model_type} skip: {reason}")
            return
        
        # Check confidenza
        min_conf = MIN_CONFIDENCE[model_type][self.symbol]
        if conf < min_conf:
            return
        
        # Check ADX
        min_adx = ADX_THRESHOLD[self.symbol]
        if adx < min_adx:
            return
        
        # Check macro
        if (signal == "LONG" and macro == "BEARISH") or (signal == "SHORT" and macro == "BULLISH"):
            return
        
        # Check spread
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return
        
        pip = get_pip_size(self.symbol)
        spread = (tick.ask - tick.bid) / pip
        if spread > SPREAD_MAX_PIPS:
            return
        
        # Calcola SL/TP
        atr = self._get_cached_atr()
        
        magic = self.magic_lstm if model_type == 'LSTM' else self.magic_xgb
        
        if signal == "LONG":
            entry = tick.ask
            sl = entry - (atr * SL_ATR_MULT)
            tp = entry + (atr * TP_ATR_MULT)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry = tick.bid
            sl = entry + (atr * SL_ATR_MULT)
            tp = entry - (atr * TP_ATR_MULT)
            order_type = mt5.ORDER_TYPE_SELL
        
        # Volume
        vol, err = compute_volume_by_risk(model_type, self.symbol, entry, sl)
        if vol <= 0:
            return
        
        # Invia ordine
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": vol,
            "type": order_type,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": f"V26-{model_type}"
        }
        
        res = safe_order_send(req)
        if res:
            pips_sl = abs(entry - sl) / pip
            pips_tp = abs(tp - entry) / pip
            
            msg = f"""🔵 <b>{model_type} {signal}</b>

📊 {self.symbol}
💼 Volume: {vol} lot
💰 Entry: {entry:.5f}

🛡️ SL: {sl:.5f} ({pips_sl:.1f} pips)
🎯 TP: {tp:.5f} ({pips_tp:.1f} pips)

🤖 Conf: {conf:.1%}
📈 ADX: {adx:.1f} | RSI: {rsi:.1f}
🧭 Macro: {macro}
"""
            send_telegram(msg)
            register_trade_open()
            
            if model_type == 'LSTM':
                self.open_tickets['LSTM'].add(res.order)
            else:
                self.open_tickets['XGB'].add(res.order)
    
    def run(self):
        logging.info(f"[{self.symbol}] 🚀 Hybrid Worker Started")
        
        while self.running:
            try:
                now = datetime.now()
                
                # Orari bloccati
                if (now.hour >= NO_TRADE_START_HOUR or now.hour < NO_TRADE_END_HOUR):
                    time.sleep(60)
                    continue
                
                if now.weekday() == 4 and now.hour >= HOUR_STOP_FRIDAY:
                    time.sleep(60)
                    continue
                
                # Trailing sempre attivo
                self._manage_trailing()
                
                # Check nuova candela
                last_tick = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME_M, 0, 1)
                if last_tick is None:
                    time.sleep(1)
                    continue
                
                latest_time = pd.to_datetime(last_tick['time'][0], unit='s')
                if self.last_candle_time == latest_time:
                    time.sleep(0.5)
                    continue
                
                # Carica dati
                rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME_M, 0, 1000)
                if rates is None:
                    time.sleep(1)
                    continue
                
                # Calcola features
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time')
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                
                df_feat = compute_features(df)
                if len(df_feat) < self.seq_len + 1:
                    continue
                
                # Estrai sequenza
                seq = df_feat.iloc[-(self.seq_len + 1):-1]
                
                feature_cols = [
                    'spread', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
                    'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',
                    'ATRr_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHh_14_3_3',
                    'SMA_50', 'SMA_200', 'dayofweek', 'hourofday',
                    'open', 'high', 'low', 'close', 'volume'
                ]
                
                if not all(c in seq.columns for c in feature_cols):
                    continue
                
                seq_data = seq[feature_cols].values.astype(np.float32)
                if np.isnan(seq_data).any():
                    self.last_candle_time = latest_time
                    continue
                
                seq_scaled = self.scaler.transform(seq_data)
                
                # Metriche
                adx = df_feat['ADX_14'].iloc[-2]
                rsi = df_feat['RSI_14'].iloc[-2]
                macro = check_macro_trend(self.symbol)
                
                # ============================================
                # PREDIZIONI DUALI
                # ============================================
                
                # 1. LSTM (sequenze)
                seq_lstm = seq_scaled.reshape(1, self.seq_len, -1)
                pred_lstm = self.lstm_model.predict(seq_lstm, verbose=0)[0]
                idx_lstm = int(np.argmax(pred_lstm))
                conf_lstm = float(np.max(pred_lstm))
                signal_lstm = {0: "SHORT", 1: "FLAT", 2: "LONG"}[idx_lstm]
                
                # 2. XGBoost (flat)
                last_row = seq_scaled[-1].reshape(1, -1)
                dmatrix = xgb.DMatrix(last_row)
                pred_xgb = self.xgb_model.predict(dmatrix)[0]
                idx_xgb = int(np.argmax(pred_xgb))
                conf_xgb = float(np.max(pred_xgb))
                signal_xgb = {0: "SHORT", 1: "FLAT", 2: "LONG"}[idx_xgb]
                
                # ============================================
                # TENTATIVI APERTURA (Indipendenti)
                # ============================================
                
                if signal_lstm != "FLAT":
                    self._try_open_trade('LSTM', signal_lstm, conf_lstm, adx, rsi, macro)
                
                if signal_xgb != "FLAT":
                    self._try_open_trade('XGB', signal_xgb, conf_xgb, adx, rsi, macro)
                
                self.last_candle_time = latest_time
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Worker error {self.symbol}: {e}")
                time.sleep(5)

# ============================================
# MAIN
# ============================================
def notify_startup():
    acct = mt5.account_info()
    if not acct:
        return
    
    msg = f"""🤖 <b>BOT V26 HYBRID AVVIATO</b>

💰 Capitale: €{acct.equity:.2f}

🧠 Pool LSTM: €{CAPITAL_ALLOCATION['LSTM']:.0f} (0.5% risk)
⚡ Pool XGB: €{CAPITAL_ALLOCATION['XGB']:.0f} (0.6% risk)

🛡️ Safety:
• Max 3 posizioni globali
• Margine minimo 30%
• Daily loss: -€25

📊 Symbols: {', '.join(SYMBOLS)}
🕐 {datetime.now().strftime('%H:%M')}

✅ Strategia Ibrida Operativa!
"""
    send_telegram(msg)

def main():
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return
    
    notify_startup()
    
    workers = []
    for idx, symbol in enumerate(SYMBOLS):
        try:
            lstm_model, xgb_model, scaler = load_models(symbol)
            worker = HybridWorker(symbol, lstm_model, xgb_model, scaler, idx)
            workers.append(worker)
            logging.info(f"✅ {symbol} models loaded")
        except Exception as e:
            logging.error(f"❌ {symbol} load failed: {e}")
            send_telegram(f"⚠️ Errore caricamento {symbol}: {e}")
    
    for w in workers:
        w.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        send_telegram("🛑 <b>BOT FERMATO</b>")
        mt5.shutdown()

if __name__ == "__main__":
    main()
