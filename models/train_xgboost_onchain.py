import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas_ta as ta
import os
import json
import datetime

# --- 1. Veri Yükleme (Fiyat + On-Chain) ---
def load_data():
    # A. Fiyat Verisi (Son 5 Yıl)
    print("Fiyat verisi indiriliyor...", flush=True)
    df_price = yf.download('BTC-USD', period='5y', interval='1d')
    
    # YFinance bazen MultiIndex kolon döndürür ('Close', 'BTC-USD')
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [col[0] for col in df_price.columns]
        
    df_price.reset_index(inplace=True)
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
    df_price.set_index('Date', inplace=True)
    
    # B. On-Chain Verisi (CSV'den)
    csv_path = "data/onchain_data_test.csv"
    if not os.path.exists(csv_path):
        print("HATA: On-Chain verisi bulunamadı. Önce 'data/onchain_loader.py' çalıştırın.")
        return None
        
    print("On-Chain verisi yükleniyor...", flush=True)
    df_onchain = pd.read_csv(csv_path)
    df_onchain['Date'] = pd.to_datetime(df_onchain['Date']).dt.date
    df_onchain.set_index('Date', inplace=True)
    
    # C. Birleştirme (Inner Join)
    # Sadece her iki verinin de olduğu günleri alıyoruz
    df_merged = df_price.join(df_onchain, how='inner')
    
    print(f"Toplam Veri: {len(df_merged)} satır (Fiyat + On-Chain)")
    return df_merged

# --- 2. Feature Engineering (Teknik + On-Chain Özellikler) ---
def add_features(df):
    df = df.copy()
    
    # A. Teknik İndikatörler
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    # B. On-Chain Türevleri (ROC - Rate of Change)
    # Ham değerler yerine değişim oranları daha anlamlıdır
    onchain_cols = ['Hash_Rate', 'Difficulty', 'Transaction_Count', 'Unique_Addresses', 'Avg_Block_Size', 'Miners_Revenue', 'Mempool_Size']
    
    for col in onchain_cols:
        if col in df.columns:
            # 1 günlük ve 7 günlük değişim yüzdeleri
            df[f'{col}_ROC_1'] = df[col].pct_change(1)
            df[f'{col}_ROC_7'] = df[col].pct_change(7)
            
    # C. Hedef Değişken (Yarın artacak mı?)
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Temizlik
    df.dropna(inplace=True)
    return df

# --- 3. Model Eğitimi ---
def train_model():
    # 1. Veri Hazırla
    df = load_data()
    if df is None: return
    
    df = add_features(df)
    
    # Özellik Seçimi (Teknik + On-Chain ROC feature'ları)
    feature_cols = [
        'RSI', 'MACD', 'SMA_50', 'SMA_200', 
        'Hash_Rate_ROC_1', 'Hash_Rate_ROC_7',
        'Difficulty_ROC_1', 'Difficulty_ROC_7',
        'Transaction_Count_ROC_1', 'Transaction_Count_ROC_7',
        'Unique_Addresses_ROC_1', 'Unique_Addresses_ROC_7',
        'Miners_Revenue_ROC_1', 'Miners_Revenue_ROC_7'
    ]
    
    # Mevcut feature'ları kontrol et (bazıları CSV'de olmayabilir)
    valid_features = [f for f in feature_cols if f in df.columns]
    
    X = df[valid_features]
    y = df['Target']
    
    print(f"Kullanılan Özellikler ({len(valid_features)}): {valid_features}")
    
    # 2. Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 3. XGBoost
    model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 4. Değerlendirme
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🔍 Doğruluk (Accuracy): {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 5. Kaydetme
    os.makedirs("models/saved_models", exist_ok=True)
    model.save_model("models/saved_models/xgboost_onchain.json")
    print("✅ Model kaydedildi: models/saved_models/xgboost_onchain.json")
    
    # Özellik Önemleri
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': valid_features, 'Importance': importance}).sort_values('Importance', ascending=False)
    print("\n🏆 En Önemli 5 Özellik:")
    print(feature_imp.head(5))
    
    # Metadata Kaydı
    metrics = {
        "model": "XGBoost_OnChain",
        "accuracy": acc,
        "features": valid_features,
        "date": str(datetime.date.today())
    }
    with open("reports/onchain_metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    train_model()
