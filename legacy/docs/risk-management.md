# Risk Management in Algorithmic Trading

Comprehensive risk management strategies implemented in V26 and V29 trading systems, covering position sizing, drawdown control, and operational safeguards.

---

## 🎯 Risk Management Philosophy

**Core Principle:** *Capital preservation is paramount. Profits are secondary to survival.*

### Three Pillars of Risk Management

1. **Position Sizing**: Control risk per trade
2. **Portfolio Protection**: Limit total exposure
3. **Operational Safeguards**: Prevent catastrophic failures

---

## 💰 Position Sizing

### Fixed Percentage Risk Model

**Formula:**
```python
risk_amount = account_balance * RISK_PERCENTAGE
position_size = risk_amount / stop_loss_pips / pip_value

# Example:
# Balance: EUR 1000
# Risk: 2% = EUR 20
# Stop Loss: 20 pips
# Pip Value: EUR 0.10 (micro lot EURUSD)
# Position Size: EUR 20 / 20 / 0.10 = 10 micro lots (0.10 lot)
```

**Risk Levels:**

| Profile | Risk % | EUR 1000 Acct | Max Loss/Trade | Trades to Ruin* |
|---------|--------|---------------|----------------|-----------------|
| Conservative | 1% | EUR 10 | EUR 10 | 100+ |
| Moderate | 2% | EUR 20 | EUR 20 | 50+ |
| Aggressive | 3% | EUR 30 | EUR 30 | 33+ |

*Assuming worst-case scenario of consecutive losses

**Implemented Approach:**
```python
class RiskManager:
    def calculate_position_size(self, balance, risk_pct, sl_pips):
        risk_amount = balance * risk_pct
        pip_value = 0.0001 * 100000  # Standard lot EURUSD
        position_size = risk_amount / (sl_pips * pip_value)
        
        # Round to broker minimum (0.01 lot)
        position_size = round(position_size, 2)
        
        # Enforce maximum position size (10% of balance)
        max_position = balance * 0.10 / (pip_value * sl_pips)
        position_size = min(position_size, max_position)
        
        return position_size
```

### Kelly Criterion (Advanced)

**Formula:**
```
f* = (p * b - q) / b

Where:
f* = Fraction of capital to risk
p = Win probability
q = Loss probability (1 - p)
b = Win/loss ratio (avg_win / avg_loss)

# Example:
# Win rate: 55% (p=0.55, q=0.45)
# Avg win: 30 pips
# Avg loss: 20 pips
# b = 30/20 = 1.5

f* = (0.55 * 1.5 - 0.45) / 1.5
f* = (0.825 - 0.45) / 1.5
f* = 0.25 (25% of capital)

# Too aggressive! Use fractional Kelly:
f_practical = f* * 0.5 = 12.5%
# Still too high for most traders
```

**Why We Don't Use Full Kelly:**
- Assumes perfect win rate estimation (overfitting risk)
- Very aggressive position sizes
- High volatility in account balance
- Psychological stress

**Fractional Kelly Approach:**
```python
def fractional_kelly(win_rate, avg_win, avg_loss, fraction=0.25):
    """
    Conservative Kelly: Use 25% of recommended size
    """
    if avg_loss == 0:
        return 0.02  # Fallback to 2%
    
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (p * b - q) / b
    kelly = max(0, min(kelly, 0.25))  # Cap at 25%
    
    return kelly * fraction  # Further reduce
```

---

## 🛡️ Portfolio Protection

### Maximum Drawdown Limits

**Definition:** Maximum peak-to-trough decline in account value.

**Risk Levels:**

| Drawdown | Action | Rationale |
|----------|--------|-----------|
| 10% | Monitor closely | Normal volatility |
| 15% | Reduce risk to 1% | System may be degrading |
| 20% | Stop trading, investigate | Significant issue |
| 25% | Shutdown system | Critical failure |

**Implementation:**
```python
class DrawdownProtection:
    def __init__(self, max_drawdown=0.20):
        self.peak_balance = 0
        self.max_drawdown = max_drawdown
        
    def update(self, current_balance):
        # Track peak
        self.peak_balance = max(self.peak_balance, current_balance)
        
        # Calculate drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
        else:
            drawdown = 0
        
        # Check limit
        if drawdown >= self.max_drawdown:
            self.emergency_shutdown()
            return True
        
        return False
    
    def emergency_shutdown(self):
        # Close all positions
        # Stop opening new trades
        # Send alert
        # Log event
        pass
```

### Daily Loss Limits

**Purpose:** Prevent catastrophic single-day losses.

**Rule:** Stop trading if daily loss exceeds X%.

```python
class DailyLossLimit:
    def __init__(self, max_daily_loss=0.05):  # 5%
        self.start_of_day_balance = 0
        self.max_daily_loss = max_daily_loss
        self.trading_halted = False
        
    def check_daily_loss(self, current_balance):
        if self.start_of_day_balance == 0:
            self.start_of_day_balance = current_balance
            
        daily_loss = (self.start_of_day_balance - current_balance) / self.start_of_day_balance
        
        if daily_loss >= self.max_daily_loss:
            self.trading_halted = True
            self.send_alert(f"Daily loss limit hit: {daily_loss:.2%}")
            
        return self.trading_halted
```

### Maximum Concurrent Positions

**Single Symbol:**
```python
MAX_POSITIONS_PER_SYMBOL = 1  # No overlapping trades
```

**Multi Symbol:**
```python
MAX_TOTAL_POSITIONS = 3  # Max 3 positions across all symbols
MAX_CORRELATED_POSITIONS = 2  # Max 2 in correlated pairs (EUR/GBP)
```

**Correlation Matrix:**
```
         EURUSD  GBPUSD  AUDUSD
EURUSD    1.00    0.65    0.42
GBPUSD    0.65    1.00    0.38
AUDUSD    0.42    0.38    1.00
```

**Risk:** Opening LONG EURUSD + LONG GBPUSD = ~1.65x effective exposure

---

## ⏰ Temporal Risk Controls

### Weekend Protection

**Problem:** Markets gap on weekends (Friday close ≠ Monday open)

**Solution:**
```python
def is_weekend_risk_period(current_time):
    """
    Avoid trading Friday 21:00 UTC to Monday 09:00 UTC
    """
    weekday = current_time.weekday()
    hour = current_time.hour
    
    # Friday after 21:00
    if weekday == 4 and hour >= 21:
        return True
    
    # All day Saturday and Sunday
    if weekday in [5, 6]:
        return True
    
    # Monday before 09:00
    if weekday == 0 and hour < 9:
        return True
    
    return False
```

**Impact:**
- Avoid ~20% of total trading time
- Eliminate weekend gap risk
- Slight reduction in opportunities (acceptable trade-off)

### High-Impact News Protection

**Events to Avoid:**
- NFP (Non-Farm Payrolls)
- FOMC Rate Decisions
- Central Bank Speeches
- GDP Releases
- CPI/Inflation Reports

**Implementation:**
```python
HIGH_IMPACT_EVENTS = {
    'NFP': (1, 'first friday', 14, 30),  # First Friday, 14:30 UTC
    'FOMC': (8, 'scheduled', 19, 0),     # 8 times/year, 19:00 UTC
    # ... etc
}

def is_news_blackout(current_time):
    """
    Avoid trading 1 hour before and 2 hours after high-impact news
    """
    for event in HIGH_IMPACT_EVENTS:
        event_time = get_next_event_time(event)
        time_diff = abs((current_time - event_time).total_seconds() / 3600)
        
        if time_diff < 3:  # Within 3 hours of event
            return True
    
    return False
```

---

## 🔍 Market Condition Filters

### ADX Trend Strength Filter

**Purpose:** Only trade when market is trending (not ranging)

**Logic:**
```python
MIN_ADX = 20  # Minimum trend strength

if adx < MIN_ADX:
    # Market ranging, no trade
    pass
else:
    # Market trending, consider trade
    if signal:
        open_trade()
```

**ADX Interpretation:**
- ADX < 20: Weak trend (avoid)
- ADX 20-25: Moderate trend (trade with caution)
- ADX > 25: Strong trend (optimal)
- ADX > 40: Very strong trend (may be exhausted)

### Volatility Filter

**Purpose:** Avoid trading in extreme volatility

**ATR-Based Filter:**
```python
def check_volatility(current_atr, avg_atr):
    """
    Avoid trading if volatility too high or too low
    """
    volatility_ratio = current_atr / avg_atr
    
    if volatility_ratio < 0.5:
        return 'TOO_LOW'  # Dead market
    elif volatility_ratio > 2.0:
        return 'TOO_HIGH'  # Chaotic market
    else:
        return 'NORMAL'   # OK to trade
```

### Spread Filter

**Purpose:** Avoid trading when spreads widen

**Normal Spreads (EURUSD):**
- London: 0.6-1.0 pips
- New York: 0.8-1.2 pips
- Overlap: 0.6-0.9 pips
- Asian: 1.2-1.8 pips

**Filter:**
```python
MAX_SPREAD = 2.0  # pips

current_spread = (ask - bid) / point
if current_spread > MAX_SPREAD:
    # Too expensive to trade
    skip_trade()
```

---

## 🚨 Operational Safeguards

### Connection Monitoring

```python
class ConnectionMonitor:
    def __init__(self):
        self.last_heartbeat = time.time()
        self.max_silence = 300  # 5 minutes
        
    def check_connection(self):
        if time.time() - self.last_heartbeat > self.max_silence:
            self.reconnect()
            self.send_alert("Connection lost, reconnecting...")
            
    def reconnect(self):
        mt5.shutdown()
        time.sleep(5)
        if not mt5.initialize():
            self.emergency_shutdown()
```

### Order Validation

```python
def validate_order(symbol, order_type, volume, sl, tp):
    """
    Pre-flight checks before sending order
    """
    # Check symbol exists
    if not mt5.symbol_select(symbol):
        return False, "Symbol not available"
    
    # Check volume limits
    symbol_info = mt5.symbol_info(symbol)
    if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
        return False, "Invalid volume"
    
    # Check SL/TP distances
    min_distance = symbol_info.trade_stops_level * symbol_info.point
    current_price = mt5.symbol_info_tick(symbol).ask
    
    if abs(current_price - sl) < min_distance:
        return False, "SL too close"
    if abs(current_price - tp) < min_distance:
        return False, "TP too close"
    
    # Check account balance
    required_margin = calculate_margin(symbol, volume)
    free_margin = mt5.account_info().margin_free
    
    if required_margin > free_margin * 0.8:  # Use max 80% margin
        return False, "Insufficient margin"
    
    return True, "OK"
```

### Error Handling

```python
def safe_trade_execution(func):
    """
    Decorator for robust trade execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except NetworkError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    log_error(f"Trade failed after {max_retries} attempts: {e}")
                    send_alert(f"CRITICAL: Trade execution failed")
                    return None
            except Exception as e:
                log_error(f"Unexpected error: {e}")
                send_alert(f"ERROR: {e}")
                return None
    return wrapper

@safe_trade_execution
def open_trade(symbol, direction, volume, sl, tp):
    # ... trading logic
    pass
```

---

## 📊 Risk Metrics & Monitoring

### Key Metrics to Track

**Daily:**
- Current drawdown vs. peak
- Daily P&L
- Win rate (rolling 30 trades)
- Active positions and exposure

**Weekly:**
- Sharpe ratio (rolling 3 months)
- Profit factor
- Average win/loss
- Max consecutive losses

**Monthly:**
- Total return
- Risk-adjusted return (Sharpe, Sortino)
- Drawdown recovery time
- Model performance degradation

### Alerting System

```python
class AlertManager:
    def __init__(self):
        self.alert_levels = {
            'INFO': self.send_telegram,
            'WARNING': self.send_telegram + self.log_to_file,
            'CRITICAL': self.send_telegram + self.send_email + self.emergency_stop
        }
    
    def alert(self, message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = f"[{level}] {timestamp}: {message}"
        
        for action in self.alert_levels[level]:
            action(formatted_msg)
```

**Alert Triggers:**

| Event | Level | Action |
|-------|-------|--------|
| Trade opened | INFO | Log |
| Trade closed | INFO | Log + Telegram |
| Daily loss >3% | WARNING | Telegram + Log |
| Drawdown >15% | WARNING | Telegram + Email |
| Connection lost | WARNING | Telegram + Reconnect |
| Daily loss >5% | CRITICAL | All + Stop Trading |
| Drawdown >20% | CRITICAL | All + Emergency Shutdown |

---

## 🎓 Risk Management Best Practices

### 1. Start Small
```python
# Phase 1: Demo account (1-3 months)
ACCOUNT_TYPE = 'DEMO'
RISK_PER_TRADE = 0.02

# Phase 2: Micro account (3-6 months)
ACCOUNT_TYPE = 'LIVE'
INITIAL_BALANCE = 500  # Small real money
RISK_PER_TRADE = 0.01  # Conservative

# Phase 3: Scale up gradually
if months_profitable >= 6:
    INITIAL_BALANCE = 2000
    RISK_PER_TRADE = 0.02
```

### 2. Regular Performance Reviews

**Weekly Checklist:**
- [ ] Review all trades (winners and losers)
- [ ] Check win rate vs. expected
- [ ] Verify execution quality (slippage)
- [ ] Monitor model confidence distribution
- [ ] Check for unusual patterns

**Monthly Checklist:**
- [ ] Full performance analysis
- [ ] Compare to backtest expectations
- [ ] Check for model degradation
- [ ] Review risk parameters (still appropriate?)
- [ ] Update market context if needed

### 3. Model Retraining Schedule

**Trigger Retraining When:**
- Performance degrades >20% from backtest
- Win rate drops below 42% (sustained 50+ trades)
- Market regime change detected
- 6+ months since last training

**Retraining Process:**
1. Collect recent data (last 18 months)
2. Retrain with same methodology
3. Backtest new model thoroughly
4. Compare to current model
5. Deploy only if improvement confirmed

### 4. Disaster Recovery Plan

**Scenario: Model Fails Catastrophically**

1. **Immediate:** Close all positions manually
2. **Short-term:** Stop bot, investigate root cause
3. **Medium-term:** Retrain or rollback to previous version
4. **Long-term:** Implement safeguards to prevent recurrence

**Scenario: Broker Issues**

1. Have backup broker account ready
2. Keep models/data synced to cloud
3. Document deployment process
4. Test failover annually

**Scenario: VPS Down**

1. Monitor VPS uptime (UptimeRobot, etc.)
2. Have backup VPS provisioned
3. Automated deployment scripts
4. Can manually trade if needed

---

## 📉 Common Risk Management Mistakes

### ❌ Mistake 1: Overleveraging

**Problem:**
```python
# Too aggressive!
RISK_PER_TRADE = 0.10  # 10% per trade
# One bad streak: 5 losses = -50% account!
```

**Solution:**
```python
# Conservative approach
RISK_PER_TRADE = 0.01  # 1% per trade
# Same streak: 5 losses = -5% account (recoverable)
```

### ❌ Mistake 2: Ignoring Correlation

**Problem:**
```python
# Opening correlated positions
open_long('EURUSD')  # +1.0 EUR exposure
open_long('GBPUSD')  # +0.65 EUR exposure (correlated)
# Effective exposure: ~1.65x risk!
```

**Solution:**
```python
# Check correlation before opening
if correlation(symbol1, symbol2) > 0.6:
    max_combined_positions = 1
```

### ❌ Mistake 3: No Stop Loss

**Problem:**
```python
# "I'll close manually if it goes against me"
open_trade(sl=None)  # Disaster waiting to happen
```

**Solution:**
```python
# Always use hard stop loss
open_trade(sl=entry_price - 20*pip_value)  # Automated protection
```

### ❌ Mistake 4: Revenge Trading

**Problem:**
After 3 losses → Double position size to "win back losses"

**Solution:**
Stick to system, never deviate from risk rules

### ❌ Mistake 5: Ignoring Costs

**Problem:**
```python
# Model assumes 0.5 pip cost
BACKTEST_COST = 0.5

# Reality: spread + commission + slippage
REAL_COST = 1.5 + 0.3 + 0.5 = 2.3 pips
```

**Solution:**
Always use realistic costs in backtest

---

## 🎯 Risk Management Checklist

Before deploying ANY trading system:

- [ ] Position sizing limits implemented
- [ ] Maximum drawdown protection active
- [ ] Daily loss limits configured
- [ ] Weekend protection enabled
- [ ] Stop loss on every trade (no exceptions)
- [ ] Connection monitoring active
- [ ] Alert system tested
- [ ] Correlation limits enforced
- [ ] Spread filter implemented
- [ ] News blackout periods defined
- [ ] Emergency shutdown procedure tested
- [ ] Backup systems in place
- [ ] Performance monitoring automated
- [ ] Disaster recovery plan documented

---

## 📚 Further Reading

- "Risk Management in Trading" by Kenneth Grant
- "The Mathematics of Money Management" by Ralph Vince
- "Quantitative Risk Management" by Alexander J. McNeil
- "Trading Risk: Enhanced Profitability through Risk Control" by Kenneth L. Grant

---

*Risk management is not about avoiding losses—it's about surviving long enough to profit.*
