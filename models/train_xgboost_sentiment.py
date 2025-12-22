import sys
import os
import joblib
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml
from data.sentiment_loader import SentimentLoader

def train_sentiment_model():
    print("--- Sentiment Destekli XGBoost Eğitimi Başlıyor ---")
    
    # 1. Fiyat Verisini Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    df_price = loader.fetch_data(symbol, period='5y', interval='1d')
    print(f"💰 Fiyat verisi yüklendi: {len(df_price)} gün")
    
    # 2. Sentiment Verisini Yükle
    sent_loader = SentimentLoader()
    df_sentiment = sent_loader.fetch_fear_and_greed_data(limit=0)
    
    if df_sentiment is None:
        print("❌ Sentiment verisi alınamadı, işlem iptal.")
        return

    print(f"😱 Sentiment verisi yüklendi: {len(df_sentiment)} gün")
    
    # 3. Verileri Birleştir (Date üzerinden)
    # df_price index'i Date olabilir, kontrol et
    if 'Date' not in df_price.columns:
        df_price = df_price.reset_index()
    
    # Tarih formatlarını eşitle (ns -> date)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    df_price['Date'] = pd.to_datetime(df_price['Date']) # Tekrar datetime objesine
    
    # Merge (Inner join: Sadece ikisinin de olduğu günler)
    df_merged = pd.merge(df_price, df_sentiment, on='Date', how='inner')
    print(f"🔄 Birleştirilmiş veri seti: {len(df_merged)} gün")
    
    # 4. Feature Engineering
    # Mevcut fonksiyonu kullan, FNG özellikleri zaten eklendi
    df_processed = prepare_data_for_ml(df_merged)
    
    # FNG değişimlerini de özellik olarak ekleyelim
    df_processed['FNG_Change'] = df_processed['FNG_Value'].pct_change()
    df_processed['FNG_MA_7'] = df_processed['FNG_Value'].rolling(window=7).mean()
    
    # NaN temizliği
    df_processed.dropna(inplace=True)
    
    # 5. Eğitim Hazırlığı
    features = [col for col in df_processed.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'FNG_Class']]
    
    print(f"🧠 Kullanılan Özellik Sayısı: {len(features)}")
    print(f"Örnek Özellikler: {features[:10]} ... FNG_Value")
    
    X = df_processed[features]
    y = df_processed['Target']
    
    # Train/Test Split (Zaman serisi olduğu için shuffle=False)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Eğitim Seti: {len(X_train)} | Test Seti: {len(X_test)}")
    
    # 6. Model Eğitimi (XGBoost)
    # Yeni model parametreleri
    model = XGBClassifier(
        n_estimators=200,      # Biraz daha fazla ağaç
        learning_rate=0.05,    # Daha yavaş öğrenme (daha hassas)
        max_depth=4,           # Aşırı öğrenmeyi engellemek için düşük derinlik
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 7. Değerlendirme
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\n📊 Model Performansı (Test Seti):")
    print(f"Doğruluk: {acc:.4f}")
    print(f"Keskinlik: {prec:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    
    # 8. Kaydetme
    # Yeni model adı: xgb_sentiment_v1
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = "models/saved_models/xgb_sentiment_v1.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model kaydedildi: {model_path}")
    
    # Metrikleri Kaydet
    metrics = {
        "model_name": "XGBoost_Sentiment_V1",
        "features": features,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_period_start": str(X_test.index[0]), # index artık int olabilir dikkat
        # Tarihi geri almak için df_processed'ın tarih sütununa bakmamız gerekebilir
        # Ancak X dataframe'inde Date yok. df_processed.iloc[split_idx:]['Date'] kullanacağız.
        "test_period_start": str(df_processed.iloc[split_idx]['Date'].date()),
        "test_period_end": str(df_processed.iloc[-1]['Date'].date())
    }
    
    with open("reports/ml_metrics_sentiment.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("✅ Metrikler kaydedildi.")

if __name__ == "__main__":
    train_sentiment_model()
