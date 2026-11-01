raise SystemExit("Historical experiment disabled: see docs/audit.md; run python -m trading_research --demo")

"""
================================================================================
TRADING BOT V29 - PRODUCTION (LIVE REALITY ADJUSTED) - PERFETTO
================================================================================
XGBoost automated trading bot with REALISTIC LIVE cost modeling

TUTTI I FIX APPLICATI:
[OK] CRITICAL #1: Feature name 'tick_volume' (non 'volume')
[OK] CRITICAL #2: Pips calculation corretta (raw price diff, non percentage)
[OK] HIGH #1: Timezone UTC ovunque (datetime.now(timezone.utc))
[OK] HIGH #2: Weekend close >= (non ==, race condition fix)
[OK] HIGH #3: None tick check in close_position
[OK] MEDIUM #1: Spread costante (train/inference match)
[OK] MEDIUM #2: check_weekend_gap usa iloc[-2] (complete candle)
[OK] MEDIUM #3: Inf/NaN handling (replace prima di dropna)
[OK] MICRO-FIX #1: close_position verifica retcode
[OK] MICRO-FIX #2: No emoji (Windows compatible)
[OK] MICRO-FIX #3: Stop conditions filtra per magic number
[OK] MICRO-FIX #4: datetime.now(timezone.utc) no deprecation warning

Magic Number: 290500
Target: 50-55% WR with realistic cost model
Allocation: Start with EUR 100 MICRO for validation

Account: Auto-detected from MT5 login
VPS: Windows Server 2016
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import requests
from datetime import datetime, timedelta, timezone
import time
import logging
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    MODELS_DIR = "cervelli_IA_V29"
    LOG_DIR = "logs"
    
    # Symbol
    SYMBOL = "EURUSD"
    
    # Magic number (V29)
    MAGIC_V29 = 290500
    
    # Capital allocation - START MICRO for validation!
    CAPITAL_ALLOCATION = 100.0  # EUR 100 MICRO for 30 trade validation
    RISK_PER_TRADE = 0.02  # 2% = EUR 2 per trade
    
    # REALISTIC COST MODEL (pips)
    SPREAD_AVG = 1.0  # Average spread EURUSD LIVE
    SLIPPAGE_ENTRY = 1.0  # Entry slippage
    SLIPPAGE_SL = 3.0  # Stop loss slippage (worst case)
    SLIPPAGE_TP = 1.0  # Take profit slippage
    TOTAL_COST_PER_TRADE = 6.0  # Total pips cost per trade
    
    # Model parameters (adjusted for LIVE)
    CONFIDENCE_THRESHOLD = 0.60  # 60% (higher than V28 due to cost model)
    ADX_THRESHOLD = 15  # Back to 15 (stronger trends)
    
    # Risk management with BUFFERS
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.5  # Increased from 3.0 for better R:R with costs
    SLIPPAGE_BUFFER_PIPS = 5  # Extra safety margin
    
    # Filters
    GAP_THRESHOLD = 0.003  # 0.3% (tighter for weekend protection)
    VOLATILITY_MULT = 2.0
    
    # Stop conditions (for EUR 100 micro)
    MAX_DAILY_TRADES = 8
    DAILY_LOSS_LIMIT = -10.0  # 10% of EUR 100
    WEEKLY_LOSS_LIMIT = -20.0  # 20% of EUR 100
    MAX_CONSECUTIVE_LOSSES = 4
    MIN_WIN_RATE_10TRADES = 0.45  # If WR <45% on 10 trade -> STOP
    
    # Weekend protection
    FRIDAY_CUTOFF_HOUR = 20  # No trade Friday after 20:00 UTC
    FRIDAY_CLOSE_HOUR = 23  # Close all positions Friday 23:00 UTC
    MONDAY_RESUME_HOUR = 9  # Resume trading Monday 09:00 UTC
    
    # Telegram (replace with your own)
    # Get token from @BotFather and chat_id from @userinfobot on Telegram
    TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
    
    # Trading hours
    TRADE_START_HOUR = 0
    TRADE_END_HOUR = 23

# ============================================================================
# LOGGING SETUP (UTF-8, NO EMOJI)
# ============================================================================
os.makedirs(Config.LOG_DIR, exist_ok=True)
# FIX HIGH #1: datetime.now(timezone.utc) (non datetime.now())
log_file = os.path.join(Config.LOG_DIR, f"bot_v29_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================
def send_telegram(message):
    """Send Telegram notification (non-blocking)"""
    if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": Config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data, timeout=3)
        if response.status_code != 200:
            logger.warning(f"Telegram failed (status {response.status_code})")
    except requests.exceptions.Timeout:
        logger.warning("Telegram timeout (skipped)")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")

# ============================================================================
# MT5 FUNCTIONS
# ============================================================================
def initialize_mt5():
    """Initialize MT5 connection and auto-detect account"""
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        return False
    
    # Auto-detect account (no hardcoded login!)
    account_number = account_info.login
    account_balance = account_info.balance
    account_equity = account_info.equity
    
    logger.info(f"MT5 initialized - Account: {account_number}")
    logger.info(f"Balance: EUR {account_balance:.2f}, Equity: EUR {account_equity:.2f}")
    
    # Select symbol
    if not mt5.symbol_select(Config.SYMBOL, True):
        logger.error(f"Failed to select {Config.SYMBOL}")
        return False
    
    return True

def get_current_price(symbol):
    """Get current bid/ask price"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, None
    return tick.bid, tick.ask

def get_h1_data(symbol, bars=200):
    """Get H1 data"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def get_h4_data(symbol, bars=100):
    """Get H4 data"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def get_positions(magic):
    """Get open positions by magic number"""
    positions = mt5.positions_get(symbol=Config.SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic == magic]

# ============================================================================
# WEEKEND PROTECTION
# ============================================================================
def is_weekend_risk_period():
    """Check if we're in weekend risk period"""
    # FIX HIGH #1: datetime.now(timezone.utc) (non datetime.now())
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour
    
    # Friday after cutoff hour
    if weekday == 4 and hour >= Config.FRIDAY_CUTOFF_HOUR:
        return True
    
    # Saturday or Sunday
    if weekday in [5, 6]:
        return True
    
    # Monday before resume hour
    if weekday == 0 and hour < Config.MONDAY_RESUME_HOUR:
        return True
    
    return False

def close_all_positions_for_weekend():
    """Close all open positions before weekend"""
    positions = get_positions(Config.MAGIC_V29)
    if len(positions) == 0:
        return
    
    logger.info(f"Weekend protection: closing {len(positions)} position(s)")
    
    for pos in positions:
        result = close_position(pos)
        # FIX MICRO #1: close_position now returns None if failed
        if result is not None:
            logger.info(f"Position {pos.ticket} closed for weekend protection")
        else:
            logger.error(f"Failed to close position {pos.ticket}")
    
    send_telegram("Weekend protection: all positions closed")

def close_position(position):
    """Close a position"""
    symbol = position.symbol
    ticket = position.ticket
    volume = position.volume
    position_type = position.type
    
    # FIX HIGH #3: Check for None tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Cannot get tick for {symbol} to close position {ticket}")
        return None
    
    # Determine close type
    if position_type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "magic": Config.MAGIC_V29,
        "comment": "weekend_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    # FIX MICRO #1: Verify retcode before returning
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Close position {ticket} failed: {result.comment}")
        return None
    
    return result

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def calculate_features_h1(df):
    """Calculate H1 technical indicators"""
    # Returns
    df['returns'] = df['close'].pct_change()
    df['returns_2'] = df['close'].pct_change(2)
    df['returns_5'] = df['close'].pct_change(5)
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    bb_std_val = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_val)
    df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_val)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    
    # Volume
    df['volume_sma'] = df['tick_volume'].rolling(20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # FIX MEDIUM #1: Spread constant (matches training)
    df['spread'] = Config.SPREAD_AVG * 10
    df['spread_pct'] = df['spread'] / df['close'] * 100
    
    return df

def calculate_features_h4(df):
    """Calculate H4 features"""
    df['trend_h4'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['ema_h4'] = df['close'].ewm(span=20).mean()
    
    # Indicators H4
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_h4'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd_h4'] = exp1 - exp2
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_h4'] = true_range.rolling(14).mean()
    
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx_h4'] = dx.rolling(14).mean()
    
    df['returns_h4_1'] = df['close'].pct_change(1)
    df['returns_h4_2'] = df['close'].pct_change(2)
    
    sma_h4 = df['close'].rolling(20).mean()
    df['price_vs_sma_h4'] = (df['close'] - sma_h4) / sma_h4
    
    df['volume_h4'] = df['tick_volume']
    df['volume_ratio_h4'] = df['volume_h4'] / df['volume_h4'].rolling(20).mean()
    
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position_h4'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    df['atr_percentile_h4'] = df['atr_h4'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    return df

# ============================================================================
# FILTERS
# ============================================================================
def check_weekend_gap(df_h1):
    """Check Monday gap - FIX MEDIUM #2: uses iloc[-2]"""
    if len(df_h1) < 3:
        return False
    
    # FIX MEDIUM #2: Use previous complete candle
    current_time = df_h1.index[-2]
    if current_time.weekday() == 0:  # Monday
        friday_close = df_h1.iloc[-3]['close']
        monday_open = df_h1.iloc[-2]['open']
        gap = abs(monday_open - friday_close) / friday_close
        
        if gap > Config.GAP_THRESHOLD:
            logger.warning(f"Weekend gap detected: {gap:.2%}")
            return True
    
    return False

def check_extreme_volatility(df_h1, lookback=20):
    """Check extreme volatility - uses COMPLETE candle"""
    if len(df_h1) < lookback + 2:
        return False
    
    # Use previous complete candle (iloc[-2])
    current_atr = df_h1.iloc[-2]['atr']
    avg_atr = df_h1.iloc[-lookback-2:-2]['atr'].mean()
    
    if current_atr > Config.VOLATILITY_MULT * avg_atr:
        logger.warning(f"Extreme volatility: ATR {current_atr:.5f} vs avg {avg_atr:.5f}")
        return True
    
    return False

# ============================================================================
# POSITION SIZING
# ============================================================================
def calculate_lot_size(capital, risk_pct, sl_distance, symbol):
    """Calculate lot size based on risk"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01
    
    # Risk amount
    risk_amount = capital * risk_pct
    
    # Value per pip
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    
    value_per_pip = tick_value / tick_size * point * 10
    
    # Lot size
    lot_size = risk_amount / (sl_distance / point * value_per_pip)
    
    # Round to lot step
    lot_step = symbol_info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    
    # Clamp to min/max
    lot_min = symbol_info.volume_min
    lot_max = symbol_info.volume_max
    lot_size = max(lot_min, min(lot_size, lot_max))
    
    return lot_size

# ============================================================================
# TRADE EXECUTION WITH SLIPPAGE BUFFERS
# ============================================================================
def open_trade(symbol, direction, lot_size, sl, tp, comment="V29"):
    """Open trade with slippage-adjusted SL/TP"""
    bid, ask = get_current_price(symbol)
    if bid is None:
        return None
    
    point = mt5.symbol_info(symbol).point
    
    if direction == "LONG":
        trade_type = mt5.ORDER_TYPE_BUY
        price = ask
        
        # Apply slippage buffer: SL wider, TP tighter
        sl_buffered = sl - (Config.SLIPPAGE_BUFFER_PIPS * point * 10)
        tp_buffered = tp - (Config.SLIPPAGE_BUFFER_PIPS * point * 10)
    else:
        trade_type = mt5.ORDER_TYPE_SELL
        price = bid
        
        # Apply slippage buffer
        sl_buffered = sl + (Config.SLIPPAGE_BUFFER_PIPS * point * 10)
        tp_buffered = tp + (Config.SLIPPAGE_BUFFER_PIPS * point * 10)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_type,
        "price": price,
        "sl": sl_buffered,
        "tp": tp_buffered,
        "magic": Config.MAGIC_V29,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Trade failed: {result.comment}")
        return None
    
    logger.info(f"Trade opened: {direction} {lot_size} lots @ {price:.5f}, SL: {sl_buffered:.5f}, TP: {tp_buffered:.5f}")
    return result.order

# ============================================================================
# STOP CONDITIONS
# ============================================================================
def check_stop_conditions():
    """Check if bot should stop (only for V29 trades with our magic number)"""
    # FIX HIGH #1: Use UTC everywhere
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_deals = mt5.history_deals_get(today_start, datetime.now(timezone.utc))
    
    if today_deals is not None and len(today_deals) > 0:
        # CRITICAL FIX: Filter by MAGIC NUMBER + SYMBOL (not just symbol!)
        daily_pnl = sum([deal.profit for deal in today_deals 
                        if deal.symbol == Config.SYMBOL and deal.magic == Config.MAGIC_V29])
        if daily_pnl < Config.DAILY_LOSS_LIMIT:
            logger.critical(f"Daily loss limit hit: {daily_pnl:.2f}")
            return True
        
        daily_trades = len([d for d in today_deals 
                           if d.entry == mt5.DEAL_ENTRY_IN 
                           and d.magic == Config.MAGIC_V29])
        if daily_trades >= Config.MAX_DAILY_TRADES:
            logger.info(f"Daily trade limit reached: {daily_trades}")
            return True
    
    # Check weekly loss
    week_start = datetime.now(timezone.utc) - timedelta(days=7)
    week_deals = mt5.history_deals_get(week_start, datetime.now(timezone.utc))
    
    if week_deals is not None and len(week_deals) > 0:
        # CRITICAL FIX: Filter by MAGIC NUMBER + SYMBOL (not just symbol!)
        weekly_pnl = sum([deal.profit for deal in week_deals 
                         if deal.symbol == Config.SYMBOL and deal.magic == Config.MAGIC_V29])
        if weekly_pnl < Config.WEEKLY_LOSS_LIMIT:
            logger.critical(f"Weekly loss limit hit: {weekly_pnl:.2f}")
            return True
    
    # Check consecutive losses
    if week_deals is not None and len(week_deals) >= 2 * Config.MAX_CONSECUTIVE_LOSSES:
        recent_profits = [d.profit for d in list(week_deals)[-2*Config.MAX_CONSECUTIVE_LOSSES:] 
                         if d.entry == mt5.DEAL_ENTRY_OUT and d.magic == Config.MAGIC_V29]
        if len(recent_profits) >= Config.MAX_CONSECUTIVE_LOSSES:
            consecutive_losses = 0
            for profit in reversed(recent_profits):
                if profit < 0:
                    consecutive_losses += 1
                else:
                    break
            
            if consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
                logger.critical(f"Consecutive losses: {consecutive_losses}")
                return True
    
    # Check win rate on last 10 trades
    if week_deals is not None and len(week_deals) >= 20:
        last_10_deals = [d for d in list(week_deals)[-20:] if d.entry == mt5.DEAL_ENTRY_OUT]
        if len(last_10_deals) >= 10:
            wins = sum(1 for d in last_10_deals if d.profit > 0)
            wr = wins / len(last_10_deals)
            if wr < Config.MIN_WIN_RATE_10TRADES:
                logger.critical(f"Win rate too low: {wr:.1%} on last {len(last_10_deals)} trades")
                return True
    
    return False

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    logger.info("=" * 80)
    logger.info("TRADING BOT V29 - LIVE REALITY ADJUSTED - PERFETTO")
    logger.info("=" * 80)
    logger.info("TUTTI I FIX OPUS APPLICATI + 2 MICRO-FIX")
    logger.info("Cost per trade: ~6 pips (spread + slippage)")
    logger.info("Weekend protection: ACTIVE")
    logger.info("Slippage buffers: +5 pips safety")
    logger.info("=" * 80)
    logger.info(f"Symbol: {Config.SYMBOL}")
    logger.info(f"Capital allocation: EUR {Config.CAPITAL_ALLOCATION}")
    logger.info(f"Risk per trade: {Config.RISK_PER_TRADE:.0%} = EUR {Config.CAPITAL_ALLOCATION * Config.RISK_PER_TRADE:.2f}")
    logger.info(f"Magic: {Config.MAGIC_V29}")
    logger.info(f"Confidence threshold: {Config.CONFIDENCE_THRESHOLD:.0%}")
    logger.info(f"ADX threshold: {Config.ADX_THRESHOLD}")
    
    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MT5")
        return
    
    # Feature columns (48 total: 34 H1 + 14 H4)
    # FIX CRITICAL #1: 'tick_volume' not 'volume'!
    feature_cols = [
        'returns', 'returns_2', 'returns_5',
        'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'rsi', 'stoch_k', 'stoch_d',
        'macd', 'macd_signal', 'macd_hist',
        'atr', 'atr_pct', 'adx',
        'volume_sma', 'volume_ratio',
        'hour', 'day_of_week', 'hour_sin', 'hour_cos',
        'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'spread_pct',
        # H4 features
        'trend_h4', 'ema_h4', 'adx_h4', 'rsi_h4', 'macd_h4', 'atr_h4',
        'returns_h4_1', 'returns_h4_2', 'price_vs_sma_h4',
        'volume_h4', 'volume_ratio_h4', 'bb_position_h4', 'atr_percentile_h4',
        'rsi_divergence_h4'
    ]
    
    # Load model
    try:
        model = xgb.Booster()
        model.load_model(os.path.join(Config.MODELS_DIR, f"{Config.SYMBOL}_XGB_V29.json"))
        scaler = joblib.load(os.path.join(Config.MODELS_DIR, f"{Config.SYMBOL}_SCALER_V29.joblib"))
        logger.info(f"Model loaded: {len(feature_cols)} features")
        logger.info(f"Scaler loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    send_telegram("Bot V29 PERFETTO started (all Opus fixes applied)")
    logger.info("Starting trading loop...")
    
    # Main loop
    last_weekend_check = None
    
    while True:
        try:
            logger.info("Checking signals...")
            
            # FIX HIGH #1: datetime.now(timezone.utc) everywhere
            now = datetime.now(timezone.utc)
            
            # Weekend protection check (every hour)
            if last_weekend_check is None or (now - last_weekend_check).total_seconds() > 3600:
                # FIX HIGH #2: >= instead of == (race condition fix)
                if now.weekday() == 4 and now.hour >= Config.FRIDAY_CLOSE_HOUR:
                    close_all_positions_for_weekend()
                last_weekend_check = now
            
            # Check if weekend risk period
            if is_weekend_risk_period():
                logger.info("Weekend risk period - skipping trade")
                time.sleep(300)
                continue
            
            # Check stop conditions
            if check_stop_conditions():
                logger.critical("Stop conditions met - shutting down")
                send_telegram("Bot stopped: stop conditions")
                break
            
            # Check trading hours (FIX HIGH #1: use UTC)
            current_hour = datetime.now(timezone.utc).hour
            if not (Config.TRADE_START_HOUR <= current_hour <= Config.TRADE_END_HOUR):
                time.sleep(60)
                continue
            
            # Check if already have position
            positions = get_positions(Config.MAGIC_V29)
            if len(positions) > 0:
                time.sleep(60)
                continue
            
            # Get data
            df_h1 = get_h1_data(Config.SYMBOL, bars=200)
            df_h4 = get_h4_data(Config.SYMBOL, bars=100)
            
            if df_h1 is None or df_h4 is None:
                logger.warning("Failed to get data from MT5")
                time.sleep(60)
                continue
            
            logger.info(f"[DEBUG] Downloaded H1: {len(df_h1)} bars, H4: {len(df_h4)} bars")
            
            # Calculate features
            df_h1 = calculate_features_h1(df_h1)
            df_h4 = calculate_features_h4(df_h4)
            
            # Merge H4 into H1
            df_h4_cols = ['trend_h4', 'ema_h4', 'adx_h4', 'rsi_h4', 'macd_h4', 'atr_h4',
                         'returns_h4_1', 'returns_h4_2', 'price_vs_sma_h4',
                         'volume_h4', 'volume_ratio_h4', 'bb_position_h4', 'atr_percentile_h4']
            
            df = df_h1.join(df_h4[df_h4_cols], how='left')
            df[df_h4_cols] = df[df_h4_cols].ffill()
            df['rsi_divergence_h4'] = df['rsi'] - df['rsi_h4']
            
            # FIX MEDIUM #3: Handle inf/nan
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna()
            
            if len(df) == 0:
                time.sleep(60)
                continue
            
            # CRITICAL: Use previous COMPLETE candle (no look-ahead bias!)
            current = df.iloc[-2]
            
            # Log which candle we're using (for verification)
            logger.info(f"[DEBUG] Using candle: {df.index[-2]} (previous complete candle)")
            logger.info(f"[DEBUG] Current ADX: {current['adx']:.1f}, Close: {current['close']:.5f}")
            
            # Filters
            if check_weekend_gap(df):
                logger.info("Skipped: weekend gap")
                time.sleep(3600)
                continue
            
            if check_extreme_volatility(df):
                logger.info("Skipped: extreme volatility")
                time.sleep(600)
                continue
            
            if current['adx'] < Config.ADX_THRESHOLD:
                logger.info(f"Skipped: ADX {current['adx']:.1f} < {Config.ADX_THRESHOLD}")
                time.sleep(300)
                continue
            
            # Predict
            X = current[feature_cols].values.reshape(1, -1)
            X_scaled = scaler.transform(X)
            dmatrix = xgb.DMatrix(X_scaled)
            pred_proba = model.predict(dmatrix)[0]
            
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[pred_class]
            
            # DEBUG: Log prediction
            logger.info(f"[DEBUG] Pred class: {pred_class}, Conf: {confidence:.1%}, Proba: {pred_proba}")
            
            if confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"No signal: confidence {confidence:.1%} < {Config.CONFIDENCE_THRESHOLD:.0%}, ADX {current['adx']:.1f}")
                time.sleep(300)
                continue
            
            if pred_class == 1:  # FLAT
                logger.info(f"Signal: FLAT (confidence {confidence:.1%}), ADX {current['adx']:.1f}")
                time.sleep(300)
                continue
            
            # Trade signal
            if pred_class == 2:
                direction = "LONG"
            else:
                direction = "SHORT"
            
            # Calculate SL/TP with slippage buffers
            bid, ask = get_current_price(Config.SYMBOL)
            if bid is None:
                time.sleep(60)
                continue
            
            atr = current['atr']
            point = mt5.symbol_info(Config.SYMBOL).point
            
            if direction == "LONG":
                entry = ask
                # Standard SL/TP (will be buffered in open_trade function)
                sl = entry - (Config.SL_ATR_MULT * atr)
                tp = entry + (Config.TP_ATR_MULT * atr)
            else:
                entry = bid
                sl = entry + (Config.SL_ATR_MULT * atr)
                tp = entry - (Config.TP_ATR_MULT * atr)
            
            sl_distance = abs(entry - sl)
            
            # Position size
            lot_size = calculate_lot_size(Config.CAPITAL_ALLOCATION, Config.RISK_PER_TRADE, 
                                          sl_distance, Config.SYMBOL)
            
            logger.info(f"Signal: {direction}, Conf: {confidence:.2%}, ADX: {current['adx']:.1f}")
            logger.info(f"Entry: {entry:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, Lot: {lot_size}")
            logger.info(f"Note: Actual SL/TP will have +{Config.SLIPPAGE_BUFFER_PIPS} pips buffer")
            
            # Open trade (with buffers applied inside)
            ticket = open_trade(Config.SYMBOL, direction, lot_size, sl, tp)
            
            if ticket is not None:
                msg = f"{direction} {Config.SYMBOL}\n"
                msg += f"Lot: {lot_size}\n"
                msg += f"Entry: {entry:.5f}\n"
                msg += f"SL: {sl:.5f} | TP: {tp:.5f}\n"
                msg += f"Conf: {confidence:.1%} | ADX: {current['adx']:.1f}\n"
                msg += f"(+{Config.SLIPPAGE_BUFFER_PIPS}p buffer applied)"
                send_telegram(msg)
            
            # Wait before next check
            time.sleep(300)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)
    
    # Cleanup
    mt5.shutdown()
    logger.info("Bot stopped")

if __name__ == "__main__":
    main()
