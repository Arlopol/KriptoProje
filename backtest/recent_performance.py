
import sys
import os
import pandas as pd
import joblib
from backtesting import Backtest
from datetime import datetime, timedelta

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy_trend import MLStrategyTrend
from data.feature_engineering import prepare_data_for_ml

def analyze_recent_performance():
    print("--- 📅 Son 7 GÜNÜN Performans Analizi ---")
    
    # 1. Veri Yükle
    data_path = "data/raw/BTC-USD_5y_1d.csv"
    if not os.path.exists(data_path):
        print("Hata: Veri dosyası bulunamadı. Önce veri çekilmeli.")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 2. Modeli Yükle
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        print("Hata: Model dosyası bulunamadı.")
        return
        
    print(f"Model Yüklendi: {model_path}")
    model = joblib.load(model_path)
    
    # 3. İndikatörleri Hesapla (Feature Engineering)
    print("İndikatörler hesaplanıyor...")
    df_processed = prepare_data_for_ml(df.copy())
    
    # 4. Tahminleri Yap
    features = [col for col in df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    
    # Modelin beklediği feature sırasını bulmak için JSON metriklerine bakabiliriz ama
    # XGBoost genellikle column name'den eşleştirir, yine de dikkatli olalım.
    # Burada direkt tahmin alıyoruz.
    try:
        df_processed['ML_Signal'] = model.predict(df_processed[features])
        df_processed['ML_Prob'] = model.predict_proba(df_processed[features])[:, 1]
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        # Feature mismatch olabilir, basitçe geçiyorum.
        return

    # 5. Backtest Çalıştır (Sadece Son 30 Günlük Veri Üzerinde Hız İçin, ama SMA200 için geçmiş lazım)
    # Tüm veriyi verelim, Backtest kütüphanesi handle eder.
    bt = Backtest(df_processed, MLStrategyTrend, cash=10000, commission=0.001)
    stats = bt.run()
    trades = stats['_trades']
    
    # 6. Son 7 Günü Filtrele
    last_date = df_processed.index[-1]
    start_date = last_date - timedelta(days=7)
    
    print(f"\nAnaliz Aralığı: {start_date.date()} - {last_date.date()}")
    
    # Trade'leri EntryTime'a göre filtrele
    # Backtesting.py trades array/series dönebilir, datetime'a çevirelim
    trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
    recent_trades = trades[trades['EntryTime'] >= start_date]
    
    if recent_trades.empty:
        print("\n⚠️ Bu hafta hiç işlem açılmamış.")
        print("Sebep (Olası):")
        print("1. Model sinyal üretmedi (Güven < 0.60)")
        print("2. Trend Filtresi (SMA 200) ters yöndeydi.")
        
        # Detaylı Bakış: Son 7 günün sinyallerini göster
        print("\n--- Son 7 Günün Sinyalleri ---")
        recent_data = df_processed.loc[start_date:]
        print(recent_data[['Close', 'SMA_200', 'ML_Signal', 'ML_Prob', 'RSI']].tail(10))
        
    else:
        print(f"\n✅ Toplam {len(recent_trades)} İşlem Bulundu:")
        print(recent_trades[['EntryTime', 'ExitTime', 'Size', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
        
    # Genel Durum
    current_price = df_processed['Close'].iloc[-1]
    sma_200 = df_processed['SMA_200'].iloc[-1]
    trend = "BOĞA (Yükseliş)" if current_price > sma_200 else "AYI (Düşüş)"
    
    print(f"\n--- Piyasa Durumu ({last_date.date()}) ---")
    print(f"Fiyat: ${current_price:,.2f}")
    print(f"Trend (SMA 200): {trend}")

if __name__ == "__main__":
    analyze_recent_performance()
