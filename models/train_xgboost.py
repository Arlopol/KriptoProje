import sys
import os
import pandas as pd
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml

def train_xgboost():
    print("--- XGBoost Model Eğitimi Başlatılıyor ---")
    
    # 1. Veri Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    # 2. Özellikleri Hazırla
    print("Özellikler (Features) oluşturuluyor...")
    df = prepare_data_for_ml(df)
    
    # Features ve Target
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    X = df[features]
    y = df['Target']
    
    # 3. Train/Test Ayrımı (Son %20 Test)
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # 4. Model Eğitimi (XGBoost)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    print("XGBoost eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Değerlendirme
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("\n--- TEST SONUÇLARI (XGBoost) ---")
    print(f"Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"Keskinlik (Precision): %{prec*100:.2f}")
    print(f"Duyarlılık (Recall): %{rec*100:.2f}")
    print(f"F1 Skoru: %{f1*100:.2f}")
    
    # 6. Kaydetme
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    joblib.dump(model, model_path)
    
    # Metrikleri JSON olarak kaydet
    metrics = {
        "model_name": "XGBoost_BTC_v1",
        "features": features,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_period_start": str(X_test.index[0].date()),
        "test_period_end": str(X_test.index[-1].date())
    }
    
    metrics_path = "reports/ml_metrics_xgboost.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Model kaydedildi: {model_path}")
    print(f"✅ Metrikler kaydedildi: {metrics_path}")

if __name__ == "__main__":
    train_xgboost()
