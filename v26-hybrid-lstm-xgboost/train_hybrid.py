# train_v26_HYBRID.py
# TRAINING STRATEGIA IBRIDA: LSTM + XGBoost
# Hardware: RTX 3080 10GB su WSL2
# Dati: MT5 M5 (Nov 2024 - Gen 2026)
# Output: cervelli_IA_2026/

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# TensorFlow con GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# XGBoost
import xgboost as xgb

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURAZIONE GPU (RTX 3080)
# ============================================
print("=" * 60)
print("🚀 TRAINING V26 HYBRID - LSTM + XGBoost")
print("=" * 60)

# Mixed Precision per Tensor Cores
mixed_precision.set_global_policy('mixed_float16')

# Configurazione GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Rilevata: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   └─ {gpu.name}")
else:
    print("⚠️  ATTENZIONE: GPU non trovata. Training su CPU (lento).")

print("=" * 60)

# ============================================
# PARAMETRI
# ============================================
DATA_DIR = "Dati_Fresh_2026"
OUTPUT_DIR = "cervelli_IA_2026"
SYMBOLS = ["GBPUSD", "AUDUSD", "EURUSD", "USDJPY"]

# Parametri temporali
SEQ_LEN = 24          # 24 candele M5 = 2 ore
PREDICT_FUTURE = 6    # 6 candele = 30 minuti avanti

# Target in pips
TARGET_PIPS = {
    'GBPUSD': 8,
    'AUDUSD': 8,
    'EURUSD': 8,
    'USDJPY': 8
}

# Training
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 512
XGB_ROUNDS = 300

# ============================================
# INDICATORI TECNICI (Identici al Bot Live)
# ============================================
def calculate_adx(df, period=14):
    """ADX - Average Directional Index"""
    df = df.copy()
    
    # Directional Movement
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']
    
    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    
    # Directional Movement
    df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0.0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0.0)
    
    # Smoothed
    df['tr_smooth'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100.0 * (df['plus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    minus_di = 100.0 * (df['minus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    
    # DX e ADX
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def compute_features(df):
    """
    Calcola TUTTE le 23 features che il bot userà live.
    CRITICO: Deve essere identico tra training e produzione.
    """
    df = df.copy()
    
    # Assicura che volume esista
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    elif 'volume' not in df.columns:
        df['volume'] = 0.0
    
    # Conversione numerica
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Medie Mobili
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['SMA_200'] = df['close'].rolling(200).mean()
    
    # 2. RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean() + 1e-9)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 3. ADX
    df['ADX_14'] = calculate_adx(df)
    
    # 4. MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    # 5. Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['BBM_20_2.0_2.0'] = sma20
    df['BBU_20_2.0_2.0'] = sma20 + 2.0 * std20
    df['BBL_20_2.0_2.0'] = sma20 - 2.0 * std20
    df['BBB_20_2.0_2.0'] = df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']
    df['BBP_20_2.0_2.0'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBB_20_2.0_2.0'] + 1e-9)
    
    # 6. ATR relativo
    prev = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev).abs(),
        (df['low'] - prev).abs()
    ], axis=1).max(axis=1)
    df['ATRr_14'] = tr.rolling(14).mean() / (df['close'] + 1e-9)
    
    # 7. Stocastico
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    k = 100 * (df['close'] - low14) / (high14 - low14 + 1e-9)
    df['STOCHk_14_3_3'] = k.rolling(3).mean()
    df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(3).mean()
    df['STOCHh_14_3_3'] = df['STOCHk_14_3_3'] - df['STOCHd_14_3_3']
    
    # 8. Time Features
    df['dayofweek'] = df.index.dayofweek
    df['hourofday'] = df.index.hour
    
    # 9. Spread (sarà aggiunto live dal bot)
    if 'spread' not in df.columns:
        df['spread'] = 0.0
    
    # Rimuovi infiniti e NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

# ============================================
# PREPARAZIONE DATASET
# ============================================
def prepare_dataset(symbol):
    """
    Carica CSV, calcola features, crea target e sequenze
    Returns: X_lstm, X_xgb, y, scaler
    """
    print(f"\n{'='*60}")
    print(f"📊 Preparazione Dataset: {symbol}")
    print(f"{'='*60}")
    
    # Carica CSV
    csv_path = os.path.join(DATA_DIR, f"{symbol}_M5.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ File non trovato: {csv_path}")
        return None, None, None, None
    
    df = pd.read_csv(csv_path)
    print(f"✅ Caricato: {len(df)} righe")
    
    # Parse date (supporta vari formati MT5)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    elif 'datetime' in df.columns:
        df['time'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        print("❌ Colonna time/datetime non trovata")
        return None, None, None, None
    
    df = df.set_index('time')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    print(f"📅 Periodo: {df.index.min()} → {df.index.max()}")
    
    # Calcola Features
    print("🔧 Calcolo indicatori...")
    df_feat = compute_features(df)
    print(f"✅ Features calcolate: {len(df_feat)} righe valide")
    
    # Ordine ESATTO delle colonne (critico per compatibilità)
    feature_cols = [
        'spread', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',
        'ATRr_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHh_14_3_3',
        'SMA_50', 'SMA_200', 'dayofweek', 'hourofday',
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Verifica che ci siano tutte
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        print(f"❌ ERRORE: Colonne mancanti: {missing}")
        return None, None, None, None
    
    data = df_feat[feature_cols].values
    
    # ============================================
    # CREAZIONE TARGET
    # ============================================
    print("🎯 Calcolo target...")
    
    pip_value = 0.01 if "JPY" in symbol else 0.0001
    threshold = TARGET_PIPS[symbol] * pip_value
    
    future_close = df_feat['close'].shift(-PREDICT_FUTURE)
    current_close = df_feat['close']
    
    # 0 = SHORT, 1 = FLAT, 2 = LONG
    targets = []
    for curr, fut in zip(current_close, future_close):
        if np.isnan(fut):
            targets.append(1)  # Flat per ultimi dati
        elif fut > curr + threshold:
            targets.append(2)  # BUY
        elif fut < curr - threshold:
            targets.append(0)  # SELL
        else:
            targets.append(1)  # FLAT
    
    targets = np.array(targets)
    
    # Rimuovi gli ultimi PREDICT_FUTURE
    data = data[:-PREDICT_FUTURE]
    targets = targets[:-PREDICT_FUTURE]
    
    # Distribuzione classi
    unique, counts = np.unique(targets, return_counts=True)
    print(f"📊 Distribuzione target:")
    for cls, cnt in zip(unique, counts):
        label = {0: "SHORT", 1: "FLAT", 2: "LONG"}[cls]
        pct = (cnt / len(targets)) * 100
        print(f"   {label}: {cnt} ({pct:.1f}%)")
    
    # ============================================
    # SCALING
    # ============================================
    print("⚙️  Scaling features...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # ============================================
    # SEQUENZE PER LSTM
    # ============================================
    print(f"🔄 Creazione sequenze LSTM (seq_len={SEQ_LEN})...")
    X_lstm = []
    y_lstm = []
    
    for i in range(SEQ_LEN, len(data_scaled)):
        X_lstm.append(data_scaled[i-SEQ_LEN:i])
        y_lstm.append(targets[i])
    
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    print(f"✅ LSTM: {X_lstm.shape[0]} sequenze di shape {X_lstm.shape[1:]}")
    
    # ============================================
    # DATI PER XGBOOST (Flat, no sequenze)
    # ============================================
    print("🔄 Preparazione dati XGBoost...")
    X_xgb = data_scaled[SEQ_LEN:]  # Allineato con LSTM
    y_xgb = targets[SEQ_LEN:]
    
    print(f"✅ XGBoost: {X_xgb.shape[0]} samples di {X_xgb.shape[1]} features")
    
    return X_lstm, X_xgb, y_lstm, scaler

# ============================================
# TRAINING LSTM
# ============================================
def build_lstm_model(input_shape):
    """Architettura LSTM per trend medio-termine"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(3, activation='softmax', dtype='float32')  # Output float32
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm(symbol, X_train, y_train, X_val, y_val):
    """Training LSTM con early stopping e class weights"""
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING LSTM: {symbol}")
    print(f"{'='*60}")
    
    # Class Weights (bilanciamento FLAT vs BUY/SELL)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"⚖️  Class Weights: {class_weight_dict}")
    
    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    print(f"\n📐 Architettura:")
    model.summary()
    
    # Callbacks
    model_path = os.path.join(OUTPUT_DIR, f"{symbol}_LSTM.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print(f"\n🚀 Avvio training ({LSTM_EPOCHS} epochs max)...")
    print(f"   Batch Size: {LSTM_BATCH_SIZE}")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Valutazione finale
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ Training completato!")
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val Accuracy: {val_acc:.4f}")
    
    return model, history

# ============================================
# TRAINING XGBOOST
# ============================================
def train_xgboost(symbol, X_train, y_train, X_val, y_val):
    """Training XGBoost per pattern veloci"""
    print(f"\n{'='*60}")
    print(f"⚡ TRAINING XGBOOST: {symbol}")
    print(f"{'='*60}")
    
    # Class Weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[int(y)] for y in y_train])
    
    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Parametri ottimizzati
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',  # Veloce su GPU/CPU
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    # GPU se disponibile
    if gpus:
        params['device'] = 'cuda'
        print("🎮 XGBoost: GPU accelerated")
    
    print(f"\n📐 Parametri:")
    for k, v in params.items():
        print(f"   {k}: {v}")
    
    # Training con early stopping
    print(f"\n🚀 Avvio training ({XGB_ROUNDS} rounds max)...")
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_ROUNDS,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Valutazione
    preds = bst.predict(dval)
    pred_labels = np.argmax(preds, axis=1)
    accuracy = np.mean(pred_labels == y_val)
    
    print(f"\n✅ Training completato!")
    print(f"   Val Accuracy: {accuracy:.4f}")
    print(f"   Best Iteration: {bst.best_iteration}")
    
    # Salva
    model_path = os.path.join(OUTPUT_DIR, f"{symbol}_XGB.json")
    bst.save_model(model_path)
    print(f"💾 Salvato: {model_path}")
    
    return bst

# ============================================
# MAIN TRAINING LOOP
# ============================================
def main():
    print(f"\n{'='*60}")
    print(f"📂 Configurazione:")
    print(f"   Dati: {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Simboli: {', '.join(SYMBOLS)}")
    print(f"{'='*60}")
    
    # Crea directory output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Loop per ogni simbolo
    for symbol in SYMBOLS:
        try:
            # ============================================
            # 1. PREPARAZIONE DATI
            # ============================================
            X_lstm, X_xgb, y, scaler = prepare_dataset(symbol)
            
            if X_lstm is None:
                print(f"⏭️  Skip {symbol} (errore dati)")
                continue
            
            # Split Train/Val (90/10)
            split_idx = int(len(X_lstm) * 0.9)
            
            X_lstm_train = X_lstm[:split_idx]
            X_lstm_val = X_lstm[split_idx:]
            
            X_xgb_train = X_xgb[:split_idx]
            X_xgb_val = X_xgb[split_idx:]
            
            y_train = y[:split_idx]
            y_val = y[split_idx:]
            
            # ============================================
            # 2. TRAINING LSTM
            # ============================================
            lstm_model, lstm_history = train_lstm(
                symbol,
                X_lstm_train, y_train,
                X_lstm_val, y_val
            )
            
            # ============================================
            # 3. TRAINING XGBOOST
            # ============================================
            xgb_model = train_xgboost(
                symbol,
                X_xgb_train, y_train,
                X_xgb_val, y_val
            )
            
            # ============================================
            # 4. SALVATAGGIO SCALER
            # ============================================
            scaler_path = os.path.join(OUTPUT_DIR, f"{symbol}_SCALER.joblib")
            joblib.dump(scaler, scaler_path)
            print(f"💾 Scaler salvato: {scaler_path}")
            
            print(f"\n{'='*60}")
            print(f"✅ {symbol} COMPLETATO")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ ERRORE su {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================
    # SUMMARY FINALE
    # ============================================
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETATO!")
    print(f"{'='*60}")
    print(f"\n📂 Modelli salvati in: {OUTPUT_DIR}/")
    print("\nFile generati per ogni simbolo:")
    print("   • {SYMBOL}_LSTM.h5      → Modello trend LSTM")
    print("   • {SYMBOL}_XGB.json     → Modello pattern XGBoost")
    print("   • {SYMBOL}_SCALER.joblib → Scaler features")
    print(f"\n{'='*60}")
    print("🚀 Prossimo step: Deploy su VPS con bot_HYBRID_v26.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
