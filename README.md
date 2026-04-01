# Algorithmic Trading System with Machine Learning

A professional algorithmic trading research project featuring advanced machine learning techniques for FOREX market prediction and automated trading execution.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LSTM-green)
![Trading](https://img.shields.io/badge/Platform-MetaTrader%205-orange)
![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen)

## 🎯 Project Overview

This repository contains two production-ready algorithmic trading systems developed for FOREX markets, demonstrating expertise in:

- **Deep Learning**: LSTM networks for temporal pattern recognition
- **Ensemble Methods**: XGBoost gradient boosting for price movement classification
- **Hybrid Architectures**: Combined deep learning and classical ML approaches
- **Production Engineering**: Real-time inference, risk management, and deployment pipelines
- **Quantitative Research**: Backtesting, walk-forward validation, and performance optimization

Both systems were designed for the MetaTrader 5 platform with live trading capabilities, featuring sophisticated risk management, multi-timeframe analysis, and real-time decision making.

---

## 📁 Repository Structure

```
algorithmic-trading-ml/
│
├── v26-hybrid-lstm-xgboost/     # Deep Learning Hybrid System
│   ├── train_hybrid.py          # Training pipeline with LSTM + XGBoost
│   ├── backtest_hybrid.py       # Walk-forward backtesting
│   ├── bot_hybrid_live.py       # Live trading bot
│   ├── requirements.txt         # Dependencies
│   └── README.md                # System documentation
│
├── v29-xgboost-production/      # Production ML System
│   ├── train_v29.py             # XGBoost training with Optuna optimization
│   ├── backtest_v29.py          # Realistic backtesting framework
│   ├── bot_v29_live.py          # Production trading bot
│   ├── requirements.txt         # Dependencies
│   └── README.md                # System documentation
│
└── docs/                        # Technical Documentation
    ├── technical-overview.md    # Architecture and methodology
    ├── risk-management.md       # Risk control strategies
    └── lessons-learned.md       # Development insights
```

---

## 🚀 Key Features

### V26 - Hybrid LSTM + XGBoost System

The hybrid system combines the temporal pattern recognition capabilities of LSTM networks with the robust classification performance of gradient-boosted trees:

- **LSTM Component**: 3-layer bidirectional network for feature extraction from price sequences
- **XGBoost Component**: Gradient boosting classifier using both raw and LSTM-derived features
- **Ensemble Strategy**: Weighted voting mechanism between deep learning and classical ML predictions
- **Multi-Symbol Support**: Simultaneous trading across EURUSD, GBPUSD, AUDUSD
- **Advanced Features**: 50+ technical indicators across multiple timeframes (H1, H4, D1)

**Technical Highlights:**
- TensorFlow/Keras LSTM implementation with attention mechanisms
- GPU-accelerated training on NVIDIA RTX 3080
- Custom loss function balancing directional accuracy and magnitude
- Dropout and batch normalization for regularization

### V29 - Production XGBoost System

A streamlined, production-optimized system focusing on reliability and computational efficiency:

- **Core Model**: XGBoost 3.x with GPU acceleration and Optuna hyperparameter optimization
- **Multi-Timeframe Features**: Hierarchical feature engineering (H1 + H4 data)
- **Smart Forward Fill**: Zero data loss through proper temporal alignment of multi-TF features
- **Optimized Inference**: <100ms prediction time, <500MB RAM usage (VPS-compatible)
- **Confidence Filtering**: Dynamic threshold adjustment based on prediction certainty
- **Risk Management**: Integrated position sizing, stop-loss, and drawdown protection

**Technical Highlights:**
- Temporal cross-validation with walk-forward testing
- Proper handling of multi-timeframe merges (forward fill, not dropna)
- Production-ready deployment pipeline (WSL2 → VPS)
- Comprehensive backtesting with realistic execution simulation

---

## 🔧 Technical Stack

### Core Technologies
- **Python**: 3.12
- **Deep Learning**: TensorFlow 2.18, Keras
- **Machine Learning**: XGBoost 3.0 (GPU), scikit-learn, LightGBM
- **Optimization**: Optuna, Ray Tune
- **Data Processing**: pandas, NumPy, TA-Lib
- **Trading Platform**: MetaTrader 5 (MT5) Python API

### Infrastructure
- **Training Hardware**: NVIDIA RTX 3080, WSL2 Ubuntu, CUDA 12.x
- **Production Deployment**: Windows Server VPS (4-core AMD EPYC, 8GB RAM)
- **Broker Integration**: ActivTrades MT5 real-time data feed

---

## 📊 System Performance

Both systems demonstrated strong technical performance during development and testing phases:

### V26 Hybrid System
- **Validation Accuracy**: 52-58% directional prediction (above random baseline)
- **Backtested Sharpe Ratio**: 1.2-1.8 (depending on market regime)
- **Multi-Symbol Capability**: Consistent performance across major pairs
- **Live Deployment**: Successfully deployed and monitored in real market conditions

### V29 Production System
- **Win Rate**: 44-48% with confidence filtering at 60% threshold
- **Profit Factor**: 0.95-1.05 (near breakeven, conservative approach)
- **Max Drawdown**: 13-18% (controlled risk)
- **Trade Frequency**: 8-12 trades/month (selective entry)
- **Model Stability**: Robust performance across 5-month out-of-sample period

*Note: Performance metrics are based on historical backtesting and limited live testing. Past performance does not guarantee future results. Trading involves substantial risk.*

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
# Python 3.12+
# MetaTrader 5 platform
# CUDA toolkit (for GPU training)
```

### Quick Start

**1. Clone repository:**
```bash
git clone https://github.com/yourusername/algorithmic-trading-ml.git
cd algorithmic-trading-ml
```

**2. Choose a system:**
```bash
# For V26 Hybrid
cd v26-hybrid-lstm-xgboost
pip install -r requirements.txt

# For V29 Production
cd v29-xgboost-production
pip install -r requirements.txt
```

**3. Train models:**
```bash
# V26
python train_hybrid.py

# V29
python train_v29.py
```

**4. Run backtests:**
```bash
# V26
python backtest_hybrid.py

# V29
python backtest_v29.py
```

**5. Deploy (optional):**
```bash
# Configure MT5 connection in bot file
# Update risk parameters
python bot_hybrid_live.py  # or bot_v29_live.py
```

Detailed setup instructions are provided in each system's README.

---

## 📚 Documentation

- **[V26 Hybrid System Documentation](v26-hybrid-lstm-xgboost/README.md)**: Architecture, training process, deployment
- **[V29 Production System Documentation](v29-xgboost-production/README.md)**: Technical details, optimization, production deployment
- **[Technical Overview](docs/technical-overview.md)**: Deep dive into methodology and design decisions
- **[Risk Management](docs/risk-management.md)**: Position sizing, stop-loss strategies, drawdown control
- **[Lessons Learned](docs/lessons-learned.md)**: Critical insights from development process

---

## 🔍 Key Technical Achievements

### Data Engineering
- **Multi-Timeframe Alignment**: Solved complex temporal alignment of H1/H4/D1 data without data loss
- **Forward Fill Strategy**: Proper handling of lower-frequency features in higher-frequency data
- **Feature Engineering**: 50+ technical indicators (momentum, volatility, trend, volume)
- **Anti-Pattern Identification**: Avoided common ML trading pitfalls (look-ahead bias, data leakage)

### Model Development
- **Class Imbalance Handling**: Dynamic profit thresholds to balance LONG/SHORT/FLAT classes
- **GPU Optimization**: Leveraged CUDA for 10-20x training speedup
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization with temporal CV
- **Ensemble Methods**: Weighted voting between LSTM and XGBoost predictions

### Production Engineering
- **Real-Time Inference**: <100ms prediction latency for live trading
- **Memory Optimization**: <500MB RAM usage for VPS deployment
- **Error Handling**: Robust exception handling for network/broker issues
- **Logging & Monitoring**: Comprehensive logging for performance tracking

### Backtesting
- **Walk-Forward Validation**: Temporal train/test splits to prevent look-ahead bias
- **Realistic Execution**: Spread, slippage, and commission modeling
- **Position Tracking**: Proper handling of trade duration to prevent overlapping positions
- **Performance Metrics**: Sharpe ratio, profit factor, max drawdown, win rate

---

## ⚠️ Risk Disclaimer

**This project is for educational and research purposes only.**

- Algorithmic trading carries substantial financial risk
- Past performance does not guarantee future results
- No system can guarantee profits in live trading
- Users should thoroughly test any system before risking real capital
- The author assumes no liability for financial losses

**Recommendation**: Only deploy with risk capital you can afford to lose, use proper position sizing, and continuously monitor performance.

---

## 📈 Future Development

Potential areas for further research and improvement:

- **Reinforcement Learning**: PPO/SAC agents for dynamic strategy optimization
- **Transformer Architectures**: Attention-based models for sequence prediction
- **Multi-Asset Correlation**: Cross-market feature engineering
- **Regime Detection**: Adaptive strategies based on market conditions
- **Alternative Data**: Sentiment analysis, news integration, order flow
- **Portfolio Management**: Multi-strategy ensemble and dynamic allocation

---

## 👨‍💻 About

Developed as a comprehensive machine learning research project in algorithmic trading, demonstrating end-to-end system development from data engineering through model training, backtesting, and production deployment.

**Skills Demonstrated:**
- Deep Learning (LSTM, attention mechanisms)
- Classical ML (XGBoost, ensemble methods)
- Time Series Analysis
- Feature Engineering
- Hyperparameter Optimization
- Production ML Systems
- Risk Management
- Quantitative Research

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contact

For questions, collaboration, or discussions about algorithmic trading and machine learning:

- GitHub: [@Spirax-M1](https://github.com/Spirax-M1)
- Email: smanciagli1@outlook.com

---

**⚡ Built with Python, TensorFlow, XGBoost, and MetaTrader 5**
