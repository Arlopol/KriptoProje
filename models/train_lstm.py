import sys
import os
import joblib
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, -1] # Son sütun Target varsayıyoruz
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model():
    print("--- LSTM (Deep Learning) Eğitimi Başlıyor ---")
    
    # 1. Veri Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    print(f"💰 Veri yüklendi: {len(df)} gün")
    
    # 2. Özellik Mühendisliği
    df_processed = prepare_data_for_ml(df)
    
    # LSTM için NaN kabul edilmez
    df_processed.dropna(inplace=True)
    
    # Özellik Seçimi (SADECE FİYAT VE TEKNİK, SENTIMENT YOK)
    features = [col for col in df_processed.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
    
    # X ve y'yi ayır (Target en sona gelecek şekilde düzenle)
    # Scaling için Target'ı ayırıp sonda birleştirmek daha güvenli
    X = df_processed[features].values
    y = df_processed['Target'].values.reshape(-1, 1)
    
    # 3. Scaling (StandardScaler daha iyidir - Outlier'lara karşı)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sequence Length Artırılıyor (10 -> 30 gün)
    SEQ_LENGTH = 30 
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_seq.append(X_scaled[i : i + SEQ_LENGTH])
        y_seq.append(y[i + SEQ_LENGTH]) 
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Tensor Boyutu: {X_seq.shape}")
    
    # 4. Train/Test Split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # 5. Model Mimarisi (Revize Edildi)
    # Daha basit ama güçlü bir yapı
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
        Dropout(0.3),            # Overfitting önlemi
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Learning Rate ayarı (Adam optimizer default 0.001 bazen hızlı olabilir)
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0001) # Daha yavaş ve dikkatli öğren
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # 6. Eğitim
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Eğitim Başlıyor... (Epochs artırıldı, Learning Rate düşürüldü)")
    history = model.fit(
        X_train, y_train,
        epochs=100, # Daha uzun eğitim
        batch_size=16, # Daha küçük batch (daha sık update)
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 7. Değerlendirme & Analiz
    print("\nTest Seti Üzerinde Tahminler...")
    y_pred_probs = model.predict(X_test)
    
    # Olasılık Dağılımını Kontrol Et (Model hep ortalama mı basıyor?)
    print(f"Olasılık İstatistikleri: Min={np.min(y_pred_probs):.4f}, Max={np.max(y_pred_probs):.4f}, Ort={np.mean(y_pred_probs):.4f}")
    
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"📊 LSTM Doğruluk: {acc:.4f}")
    print(f"📊 LSTM F1 Skoru: {f1:.4f}")
    
    # 8. Kaydetme
    os.makedirs("models/saved_models", exist_ok=True)
    model.save("models/saved_models/lstm_btc_v2.keras") # V2 ismi
    print("✅ Model kaydedildi: models/saved_models/lstm_btc_v2.keras")
    
    joblib.dump(scaler, "models/saved_models/lstm_scaler_v2.joblib")
    print("✅ Scaler kaydedildi: models/saved_models/lstm_scaler_v2.joblib")
    
    # Metrikleri JSON yap
    metrics = {
        "model_name": "LSTM_DeepLearning_V2_Refined",
        "features": features,
        "seq_length": SEQ_LENGTH,
        "accuracy": acc,
        "f1": f1,
        "test_period_start": str(df_processed.index[split_idx + SEQ_LENGTH]), 
        "test_period_end": str(df_processed.index[-1])
    }
    # Index datetime değilse düzeltmek gerekebilir, ancak Feature Engineering Date'i index yapmıyor genelde. 
    # Backtest scriptinde kontrol ederiz.
    
    with open("reports/ml_metrics_lstm.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_lstm_model()
