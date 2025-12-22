import sys
import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import add_technical_indicators, prepare_data_for_ml

def train_and_evaluate():
    print("--- ML Model Eğitimi Başlatılıyor ---")
    
    # 1. Veri Yükle
    loader = DataLoader()
    # Eğer önceden inmiş dosya varsa onu kullan, yoksa tekrar indir
    symbol = 'BTC-USD'
    df = loader.fetch_data(symbol, period='5y', interval='1d') # Daha fazla veri (5y)
    
    if df is None or df.empty:
        print("Veri bulunamadı!")
        return

    # 2. Özellik Üretimi
    print("İndikatörler hesaplanıyor...")
    df = add_technical_indicators(df)
    
    # 3. ML Hazırlığı (X ve y)
    print("Veri etiketleniyor...")
    df = prepare_data_for_ml(df)
    
    # Kullanılacak özellikler (Features)
    # OHLCV ve Target hariç hepsi
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Adj Close']]
    
    X = df[features]
    y = df['Target']
    
    print(f"Kullanılan Özellik Sayısı: {len(features)}")
    print(f"Toplam Veri Sayısı: {len(X)}")
    
    # 4. Zaman Serisi Bölümleme (Train/Test Split)
    # Son %20'yi test için ayıralım (Shuffle YOK!)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Eğitim Seti: {len(X_train)} gün")
    print(f"Test Seti:   {len(X_test)} gün")
    
    # 5. Model Eğitimi (Random Forest)
    print("Model eğitiliyor (Random Forest)...")
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_leaf=5, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6. Değerlendirme
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- MODEL PERFORMANSI (Test Seti) ---")
    print(f"Doğruluk (Accuracy):  %{acc*100:.2f}")
    print(f"Keskinlik (Precision):%{prec*100:.2f}")
    print(f"Duyarlılık (Recall):  %{rec*100:.2f}")
    print(f"F1 Skoru:             %{f1*100:.2f}")
    
    # 7. Kaydetme
    # Modeli kaydet
    models_dir = "models/saved_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, "rf_btc_v1.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model kaydedildi: {model_path}")
    
    # Metrikleri kaydet (Dashboard için)
    metrics = {
        "model_name": "Random Forest V1",
        "symbol": symbol,
        "features": features,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_period_start": str(X_test.index[0]),
        "test_period_end": str(X_test.index[-1])
    }
    
    # Reports klasörüne 'ml_metrics_latest.json' olarak kaydedelim
    metrics_path = "reports/ml_metrics_latest.json"
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrikler kaydedildi: {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()
