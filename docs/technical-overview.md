# Technical Overview: Algorithmic Trading ML Systems

A comprehensive technical analysis of the architecture, methodology, and design decisions behind V26 (Hybrid LSTM+XGBoost) and V29 (Production XGBoost) trading systems.

---

## 📐 System Architecture Comparison

### V26 Hybrid Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                       │
│            Raw OHLCV + 50 Features                  │
└──────────────┬──────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐    ┌──────▼──────┐
│   LSTM    │    │   XGBoost   │
│  Network  │    │  Classifier │
│ (Feature  │    │  (Direct)   │
│Extractor) │    │             │
└─────┬─────┘    └──────┬──────┘
      │                 │
      └────────┬────────┘
               │
        ┌──────▼──────┐
        │  Ensemble   │
        │  Voting     │
        │ (0.4/0.6)   │
        └──────┬──────┘
               │
         ┌─────▼─────┐
         │  Signal   │
         │ Generator │
         └───────────┘
```

**Key Components:**
- **LSTM Branch**: Bidirectional 2-layer network (128→64 units)
- **XGBoost Branch**: Gradient boosting on raw + LSTM features
- **Ensemble**: Weighted voting (60% XGBoost, 40% LSTM)

**Advantages:**
- Captures both temporal patterns (LSTM) and complex interactions (XGBoost)
- Robust to different market regimes
- High capacity for pattern learning

**Disadvantages:**
- Complex deployment (TensorFlow + XGBoost dependencies)
- Higher memory footprint (~1GB)
- Longer inference time (~200-300ms)
- More difficult to debug/interpret

### V29 Production Architecture

```
┌─────────────────────────────────────────────────────┐
│              MULTI-TIMEFRAME INPUT                  │
│         H1 Features + H4 Features (42 dims)         │
└──────────────┬──────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │   XGBoost   │
        │  Classifier │
        │  (Optimized)│
        │             │
        │  - Optuna   │
        │  - GPU      │
        │  - Temporal │
        │    CV       │
        └──────┬──────┘
               │
         ┌─────▼─────┐
         │ Confidence│
         │  Filter   │
         │  (60%)    │
         └─────┬─────┘
               │
         ┌─────▼─────┐
         │    ADX    │
         │  Filter   │
         │  (>20)    │
         └─────┬─────┘
               │
         ┌─────▼─────┐
         │  Signal   │
         │ Generator │
         └───────────┘
```

**Key Components:**
- **Single Model**: XGBoost 3.0 with GPU acceleration
- **Smart Features**: Multi-timeframe with forward fill
- **Dual Filtering**: Confidence + trend strength

**Advantages:**
- Simple, reliable architecture
- Fast inference (<100ms)
- Low memory (<500MB)
- Easy deployment (single dependency)
- Interpretable (feature importance)

**Disadvantages:**
- Less modeling capacity than ensemble
- May underperform in complex patterns
- Requires careful feature engineering

---

## 🔬 Machine Learning Methodology

### Model Selection Rationale

#### Why XGBoost?
- **Robustness**: Handles noisy financial data well
- **Performance**: State-of-art on tabular data
- **Interpretability**: Feature importance, SHAP values
- **Efficiency**: Fast training and inference
- **Production**: Battle-tested in industry

#### Why LSTM? (V26 only)
- **Temporal Patterns**: Natural fit for time series
- **Context Memory**: Captures long-range dependencies
- **Feature Learning**: Automatic representation learning
- **Flexibility**: Adaptable architecture

#### Why Ensemble? (V26)
- **Error Reduction**: Combines complementary strengths
- **Stability**: More robust than single model
- **Performance**: Often outperforms individual models

---

## 📊 Data Engineering

### Multi-Timeframe Feature Alignment

**The Critical Challenge:**
Merging H1 (hourly) and H4 (4-hourly) data without losing information.

**Naive Approach (WRONG):**
```python
df_h1 = get_h1_data()  # 10,000 bars
df_h4 = get_h4_data()  # 2,500 bars

df = df_h1.merge(df_h4, left_index=True, right_index=True, how='inner')
# Result: Only ~2,500 bars remain (75% data loss!)

df = df.dropna()
# Result: Even more loss if any NaN values
```

**Correct Approach (V29):**
```python
df_h1 = get_h1_data()  # 10,000 bars
df_h4 = get_h4_data()  # 2,500 bars

# Left join preserves all H1 bars
df = df_h1.merge(df_h4, left_index=True, right_index=True, 
                 how='left', suffixes=('', '_h4'))

# Forward fill H4 values (trend context changes slowly)
df = df.fillna(method='ffill')

# Result: 9,900+ bars retained (99% data retention)
```

**Why Forward Fill Works:**
- H4 indicators change slowly (every 4 hours)
- Reasonable to assume H4 context stable between updates
- Preserves temporal causality (no look-ahead)
- Industry standard for multi-TF analysis

### Label Creation Strategy

**The Profit Threshold Problem:**

Finding optimal profit threshold for classification labels:

```python
# Label: LONG if price increases MIN_PROFIT_THRESHOLD pips
# Label: SHORT if price decreases MIN_PROFIT_THRESHOLD pips  
# Label: FLAT otherwise

MIN_PROFIT_THRESHOLD = 35  # pips (EURUSD optimized)
```

**Threshold Selection:**
- Too high (50+ pips): Very few LONG/SHORT labels → Model predicts only FLAT
- Too low (20- pips): Noisy labels, hard to distinguish from random walk
- Optimal (30-40 pips): Balanced distribution, meaningful price moves

**Class Distribution Impact:**

```
Threshold 50 pips:
- SHORT: 15%
- FLAT: 70%    ← Model defaulting to FLAT!
- LONG: 15%

Threshold 35 pips:
- SHORT: 49%
- FLAT: 2%     ← Healthy balance
- LONG: 49%
```

### Feature Engineering Philosophy

**Diversity Over Redundancy:**
- Mix of trend, momentum, volatility, volume indicators
- Avoid highly correlated features (multicollinearity)
- Include raw price ratios (close/open, high/low)

**Multi-Timeframe Context:**
- H1: Immediate market conditions
- H4: Broader trend and momentum
- Avoid D1: Too slow for H1 trading

**Normalization:**
- MinMaxScaler (0-1 range) for LSTM
- No scaling for XGBoost (tree-based, scale-invariant)

---

## 🎯 Hyperparameter Optimization

### Optuna Framework (V29)

```python
def objective(trial):
    # Search space
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    
    # Temporal cross-validation
    scores = []
    for train_idx, val_idx in temporal_cv.split(X):
        model.fit(X[train_idx], y[train_idx])
        score = f1_score(y[val_idx], model.predict(X[val_idx]))
        scores.append(score)
    
    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Why Optuna?**
- Bayesian optimization (smarter than grid search)
- Pruning: Early stops unpromising trials
- Parallelization: Multi-GPU support
- Visualization: Built-in plotting

**Key Parameters:**

1. **max_depth**: Tree depth (3-10)
   - Higher → More complex patterns, risk of overfitting
   - Lower → Simpler model, may underfit
   
2. **learning_rate**: Step size (0.01-0.1)
   - Lower → Slower learning, more robust
   - Higher → Faster learning, may overshoot

3. **min_child_weight**: Minimum samples per leaf (1-10)
   - Higher → More regularization
   - Lower → More flexibility

### Temporal Cross-Validation

**Why Temporal CV?**

Standard K-Fold shuffles data → Look-ahead bias!

```python
# WRONG: Shuffled K-Fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)  # Training on future!

# CORRECT: Temporal K-Fold
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)    # Always train on past

# Fold structure:
# Fold 1: Train [0:2000],  Val [2000:2500]
# Fold 2: Train [0:2500],  Val [2500:3000]
# Fold 3: Train [0:3000],  Val [3000:3500]
# ...
```

---

## 🔄 Training Pipeline

### V26 Hybrid Training Flow

1. **Data Collection** (30 min)
   - Download 2+ years OHLCV from MT5
   - Validate data quality (no gaps)

2. **Feature Engineering** (15 min)
   - Calculate 50+ technical indicators
   - Create price ratios and derivatives

3. **Sequence Creation** (10 min)
   - Build 48-bar lookback windows for LSTM
   - Normalize features (MinMaxScaler)

4. **LSTM Training** (2-3 hours)
   - GPU acceleration (RTX 3080)
   - Early stopping on validation loss
   - Save best weights

5. **Feature Extraction** (20 min)
   - Extract LSTM representations
   - Concatenate with raw features

6. **XGBoost Training** (20-30 min)
   - Train on combined features
   - Temporal cross-validation
   - Save final model

**Total Time: ~4-5 hours**

### V29 Production Training Flow

1. **Data Collection** (20 min)
   - Download H1 + H4 data from MT5
   - Minimum 18 months required

2. **Multi-TF Merge** (10 min)
   - Forward fill H4 features into H1
   - Validate 99% data retention

3. **Feature Engineering** (15 min)
   - Calculate 42 technical indicators
   - Quality checks (no NaN, no inf)

4. **Label Creation** (5 min)
   - 35-pip profit threshold
   - 48-bar forward window

5. **Optuna Optimization** (1-2 hours)
   - 100-200 trials
   - Temporal CV (5 folds)
   - GPU accelerated

6. **Final Training** (10 min)
   - Train with best parameters
   - Full training set
   - Export model

**Total Time: ~2-3 hours**

---

## 📈 Performance Evaluation

### Key Metrics

#### Classification Metrics
- **Accuracy**: Overall correctness (target: >50%)
- **F1 Score**: Balanced precision/recall
- **Confusion Matrix**: Per-class performance

#### Trading Metrics
- **Win Rate**: % profitable trades (target: >48%)
- **Profit Factor**: Gross profit / gross loss (target: >1.3)
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.0)
- **Max Drawdown**: Largest peak-to-trough loss (target: <20%)

#### Why Win Rate >48%?

```
Breakeven calculation (EURUSD):
- Total cost per trade: 2.3 pips
- Average trade: 25 pips (SL=20, TP=30, weighted)
- Required win rate: 2.3 / 25 ≈ 9.2% edge
- Breakeven: 50% - 9.2% ≈ 48%

If WR = 48%: Breakeven (no profit, no loss)
If WR = 52%: +4% edge → profitable
If WR = 45%: -3% edge → losing money
```

### Walk-Forward Testing

**Standard Backtest (Flawed):**
```python
train = data[:8000]  # Train on past
test = data[8000:]   # Test on future
# Problem: Single test period, may be lucky/unlucky
```

**Walk-Forward (Robust):**
```python
# Period 1: Train [0:5000], Test [5000:6000]
# Period 2: Train [0:6000], Test [6000:7000]
# Period 3: Train [0:7000], Test [7000:8000]
# Average performance across periods
```

**Benefits:**
- Multiple test periods reduce luck factor
- Simulates real deployment (retrain periodically)
- More realistic performance estimate

---

## 🏭 Production Deployment

### Infrastructure Requirements

**Training (Local):**
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- RAM: 16GB
- OS: Windows 10 + WSL2 Ubuntu
- CUDA: 12.x

**Production (VPS):**
- CPU: 4-core AMD EPYC @ 2.5GHz
- RAM: 8GB
- OS: Windows Server 2019
- No GPU required (CPU inference)

### Deployment Pipeline

```
Local Training (WSL2)
    ↓
Model Export (.json, .pkl)
    ↓
SCP/FTP to VPS
    ↓
VPS Setup (Python, MT5)
    ↓
Bot Launch (systemd/service)
    ↓
Monitoring (Telegram, logs)
```

### Performance Constraints

**Inference Time Budget:**
- Target: <100ms per prediction
- Actual: 50-80ms (V29), 200-300ms (V26)
- Reason: H1 bar close → Must decide within seconds

**Memory Budget:**
- Target: <500MB total
- Actual: ~300MB (V29), ~800MB (V26)
- Reason: VPS has only 8GB shared with MT5

**CPU Budget:**
- Target: <20% average usage
- Actual: <10% (V29), ~15% (V26)
- Reason: Must not interfere with MT5 operation

---

## 🔍 Model Interpretability

### XGBoost Feature Importance

**Top Features (V29):**
1. RSI_h4 (23.5%) - Momentum on H4
2. ADX_h1 (15.2%) - Trend strength
3. MACD_histogram (12.8%) - Momentum change
4. ATR_h1 (9.3%) - Volatility
5. SMA_50_h4 (8.1%) - Medium-term trend

**Interpretation:**
- H4 features dominate (broader context crucial)
- Momentum indicators most predictive
- Volatility matters (ATR high rank)
- Trend strength critical (ADX)

### SHAP Values

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature contribution to individual prediction
shap.force_plot(shap_values[i])
```

**Benefits:**
- Understand individual predictions
- Debug model behavior
- Identify feature interactions
- Build trust in system

---

## 🎓 Lessons Learned

See [lessons-learned.md](lessons-learned.md) for detailed insights from the development process.

---

## 📚 References & Further Reading

### Books
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Algorithmic Trading" by Stefan Jansen
- "Quantitative Trading" by Ernest Chan

### Papers
- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- "LSTM: A Search Space Odyssey" (Greff et al., 2017)
- "The Kelly Criterion in Blackjack Sports Betting" (Thorp, 1997)

### Tools & Libraries
- **XGBoost**: https://xgboost.readthedocs.io/
- **TensorFlow**: https://www.tensorflow.org/
- **Optuna**: https://optuna.readthedocs.io/
- **MetaTrader 5 API**: https://www.mql5.com/en/docs/python_metatrader5

---

*Last Updated: April 2026*
