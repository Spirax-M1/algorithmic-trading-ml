# V26 Hybrid LSTM + XGBoost Trading System

A sophisticated algorithmic trading system combining deep learning (LSTM networks) and ensemble machine learning (XGBoost) for FOREX market prediction and automated execution.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)

---

## 🎯 System Overview

The V26 Hybrid system represents an advanced approach to algorithmic trading by leveraging the complementary strengths of deep learning and gradient boosting:

- **LSTM Networks**: Extract temporal patterns and sequential dependencies from price movements
- **XGBoost Classifier**: Robust decision-making using both raw features and LSTM-derived representations
- **Ensemble Strategy**: Weighted voting mechanism combining predictions from both models
- **Multi-Symbol Support**: Simultaneous trading across major FOREX pairs (EURUSD, GBPUSD, AUDUSD)

The system was designed for production deployment with real-time inference capabilities, achieving consistent performance across different market regimes.

---

## 🏗️ Architecture

### Model Components

#### 1. LSTM Feature Extractor
```
Input (48 timesteps × features) 
    ↓
Bidirectional LSTM (128 units)
    ↓
Batch Normalization + Dropout (0.3)
    ↓
Bidirectional LSTM (64 units)
    ↓
Batch Normalization + Dropout (0.3)
    ↓
Dense (32 units, ReLU)
    ↓
Output: Feature vector (32 dimensions)
```

**Key Features:**
- Bidirectional architecture captures forward and backward temporal dependencies
- Batch normalization for stable training
- Dropout for regularization
- Custom loss function balancing accuracy and magnitude

#### 2. XGBoost Classifier
```
Features:
- Raw technical indicators (50+)
- LSTM-derived feature vector (32 dims)
- Multi-timeframe aggregations (H1, H4, D1)

Model Configuration:
- Objective: multi:softprob (3 classes: LONG/FLAT/SHORT)
- Booster: gbtree
- Max depth: 6-8
- Learning rate: 0.01-0.05
- Early stopping: 50 rounds
```

#### 3. Ensemble Prediction
```python
final_prediction = (
    0.6 * xgboost_prediction + 
    0.4 * lstm_prediction
)
# Weights optimized through validation performance
```

---

## 📊 Feature Engineering

### Technical Indicators (50+ features)
- **Trend**: SMA, EMA, MACD, ADX, Parabolic SAR
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Standard Deviation
- **Volume**: Volume SMA, OBV, VWAP
- **Patterns**: Candlestick patterns, Price action signals

### Multi-Timeframe Features
- H1 (1-hour): Primary trading timeframe
- H4 (4-hour): Trend confirmation
- D1 (Daily): Long-term context

### LSTM-Specific Preprocessing
- Sequence length: 48 bars (lookback window)
- Feature scaling: MinMaxScaler (0-1 range)
- Temporal ordering preserved
- No future data leakage

---

## 🎯 Training Process

### Data Preparation
```bash
# Historical data: 2+ years of OHLCV data
# Training period: 70% of data
# Validation period: 15% of data
# Test period: 15% of data (walk-forward)
```

### Training Pipeline
```bash
python train_hybrid.py
```

**Process Flow:**
1. Download and process historical FOREX data from MT5
2. Engineer 50+ technical indicators across timeframes
3. Create sequences for LSTM training (48-bar windows)
4. Train LSTM feature extractor with early stopping
5. Extract LSTM features from all data
6. Train XGBoost classifier on combined feature set
7. Validate ensemble performance on hold-out data
8. Save models and scaler for deployment

**Training Time:**
- LSTM: 2-4 hours (GPU: NVIDIA RTX 3080)
- XGBoost: 15-30 minutes (CPU/GPU)
- Total: ~3-5 hours

**Output Files:**
- `lstm_model.h5`: Trained LSTM network
- `xgboost_model.json`: Trained XGBoost classifier
- `scaler.pkl`: Feature scaler (MinMaxScaler)
- `training_metrics.json`: Validation performance

---

## 🧪 Backtesting

### Backtest Framework
```bash
python backtest_hybrid.py
```

**Methodology:**
- **Walk-Forward Validation**: Sequential time-based splits, no look-ahead bias
- **Realistic Execution**: Spread (1.5-2 pips), slippage (0.5 pip), commission modeling
- **Risk Management**: 2% risk per trade, 20-pip stop loss, 30-pip take profit
- **Weekend Protection**: No trades Friday 21:00 - Monday 09:00 UTC

**Performance Metrics:**
- Win Rate: 52-58% (above random baseline of 50%)
- Profit Factor: 1.4-1.8 (depending on market regime)
- Sharpe Ratio: 1.2-1.6
- Max Drawdown: 15-22%
- Average trades/month: 25-35 per symbol

**Results Interpretation:**
The system demonstrated consistent edge over random trading, with performance varying by market volatility regime. Best results achieved during trending markets, acceptable performance during ranging conditions.

---

## 🚀 Live Deployment

### Prerequisites
- MetaTrader 5 platform installed
- Broker account with MT5 API access (ActivTrades or compatible)
- Python 3.12+ environment
- Trained models (`lstm_model.h5`, `xgboost_model.json`, `scaler.pkl`)

### Configuration
```python
# Edit bot_hybrid_live.py

# Trading parameters
RISK_PER_TRADE = 0.02  # 2% risk per trade
CONFIDENCE_THRESHOLD = 0.60  # Minimum prediction confidence
MIN_ADX = 20  # Minimum trend strength

# Telegram notifications
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# Symbols to trade
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD"]
```

### Launch Bot
```bash
python bot_hybrid_live.py
```

**Bot Features:**
- Real-time prediction every hour (on new H1 bar)
- Automatic position sizing based on account balance
- Stop-loss and take-profit automated
- Telegram notifications for all trades
- Error handling and automatic reconnection
- Multi-symbol monitoring

**Safety Features:**
- Maximum 1 position per symbol (no overlapping trades)
- Weekend trading protection
- ADX filter (only trade trending markets)
- Confidence threshold filtering
- Account balance protection

---

## 📈 Performance Summary

### Backtested Results (5-month test period)

**EURUSD:**
- Win Rate: 56%
- Profit Factor: 1.7
- Total Trades: 142
- Net Profit: +8.3% (account growth)

**GBPUSD:**
- Win Rate: 53%
- Profit Factor: 1.5
- Total Trades: 128
- Net Profit: +6.1%

**AUDUSD:**
- Win Rate: 51%
- Profit Factor: 1.3
- Total Trades: 105
- Net Profit: +3.2%

**Overall Portfolio:**
- Combined Sharpe Ratio: 1.6
- Max Drawdown: 18%
- Risk-Adjusted Returns: Positive across all pairs

*Note: Results based on backtesting with realistic execution assumptions. Live trading involves additional risks including network latency, broker execution quality, and changing market conditions.*

---

## 🔧 Customization

### Adjusting Risk
```python
# Conservative (1% risk)
RISK_PER_TRADE = 0.01

# Moderate (2% risk, default)
RISK_PER_TRADE = 0.02

# Aggressive (3% risk, not recommended)
RISK_PER_TRADE = 0.03
```

### Confidence Thresholds
```python
# High precision, fewer trades
CONFIDENCE_THRESHOLD = 0.70

# Balanced (default)
CONFIDENCE_THRESHOLD = 0.60

# More trades, lower precision
CONFIDENCE_THRESHOLD = 0.50
```

### Adding New Symbols
```python
# Ensure symbol available in your MT5 broker
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF"]

# Each symbol requires sufficient historical data for features
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. LSTM training very slow**
- Ensure GPU is detected: `tensorflow-gpu` installation
- Reduce sequence length if memory issues
- Use smaller batch size

**2. XGBoost predictions all same class**
- Check class balance in training data
- Adjust profit threshold for label creation
- Increase training data size

**3. Bot not opening trades**
- Verify MT5 connection: `mt5.initialize()`
- Check confidence threshold (may be too high)
- Verify ADX filter not blocking all signals
- Review Telegram logs for error messages

**4. Poor live performance vs backtest**
- Verify spread/commission assumptions match reality
- Check for broker execution delays
- Ensure model files haven't been corrupted
- Consider re-training with more recent data

---

## 📚 Further Development

### Potential Improvements
- **Transformer Architecture**: Replace LSTM with self-attention mechanisms
- **Multi-Horizon Prediction**: Predict multiple future timepoints
- **Regime Detection**: Adaptive strategies for trending/ranging markets
- **Sentiment Integration**: News and social media sentiment features
- **Reinforcement Learning**: RL agent for dynamic strategy adaptation

### Research Directions
- Cross-asset correlation analysis
- Alternative feature engineering (wavelets, Fourier transforms)
- Meta-learning for faster adaptation to new pairs
- Portfolio optimization across multiple strategies

---

## ⚠️ Risk Warning

**Trading foreign exchange carries a high level of risk and may not be suitable for all investors.**

- Past performance is not indicative of future results
- Algorithmic systems can fail due to technical issues, market conditions, or model degradation
- Only trade with capital you can afford to lose
- Start with small position sizes and demo accounts
- Monitor system performance continuously
- Be prepared to intervene manually if needed

**The author assumes no liability for financial losses incurred using this system.**

---

## 📄 License

MIT License - See main repository LICENSE file.

---

## 🤝 Support

For questions about this system:
- Open an issue in the main repository
- Check documentation in `/docs` directory
- Review backtest code for execution logic

---

**Built with TensorFlow, XGBoost, and MetaTrader 5** | *For educational and research purposes only*
