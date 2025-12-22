import sys
import os
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml

def train_v2_model():
    print("--- ML Model V2 Eğitimi Başlatılıyor ---")
    
    # 1. Veri Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    # 2. V2 Özelliklerini Hazırla (Lagler, Volatilite vs.)
    print("Özellikler (Features) oluşturuluyor...")
    df = prepare_data_for_ml(df)
    
    # Features ve Target
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    X = df[features]
    y = df['Target']
    
    print(f"Kullanılan Özellik Sayısı: {len(features)}")
    
    # 3. Time Series Split ile Train/Test Ayrımı
    # Verinin son %20'sini test olarak ayıralım
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"Eğitim Seti: {len(X_train)} gün")
    print(f"Test Seti:   {len(X_test)} gün")
    
    # 4. Model Eğitimi (Random Forest)
    # Parametreleri biraz daha güçlendirelim
    model = RandomForestClassifier(
        n_estimators=200,     # Daha fazla ağaç
        max_depth=10,         # Aşırı öğrenmeyi (overfitting) engellemek için sınır
        min_samples_split=5,
        random_state=42,
        n_jobs=-1             # Tüm işlemcileri kullan
    )
    
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Değerlendirme
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("\n--- TEST SONUÇLARI (V2) ---")
    print(f"Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"Keskinlik (Precision): %{prec*100:.2f}")
    print(f"Duyarlılık (Recall): %{rec*100:.2f}")
    print(f"F1 Skoru: %{f1*100:.2f}")
    
    # 6. Kaydetme
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = "models/saved_models/rf_btc_v2.joblib"
    joblib.dump(model, model_path)
    
    # Metrikleri JSON olarak kaydet
    metrics = {
        "model_name": "RF_BTC_v2_Advanced",
        "features": features,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_period_start": str(X_test.index[0].date()),
        "test_period_end": str(X_test.index[-1].date())
    }
    
    metrics_path = "reports/ml_metrics_v2.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Model kaydedildi: {model_path}")
    print(f"✅ Metrikler kaydedildi: {metrics_path}")

if __name__ == "__main__":
    train_v2_model()
