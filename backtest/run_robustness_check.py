import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from backtesting import Backtest
import joblib
import os
import json
import datetime
import sys

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_engineering import prepare_data_for_ml
from strategies.ml_strategy import MLStrategy

def run_backtest_for_period(df, model, period_name, start_date, end_date):
    print(f"\n--- Test: {period_name} ({start_date} - {end_date}) ---")
    
    # Tarih Aralığı Filtresi
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    period_df = df.loc[mask].copy()
    
    if period_df.empty:
        print(f"UYARI: {period_name} için veri yok!")
        return None

    # Tahmin
    features = [col for col in period_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    X = period_df[features]
    
    print(f"Veri Sayısı: {len(period_df)}")
    
    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return None

    period_df['ML_Signal'] = preds
    period_df['ML_Prob'] = probs
    
    # Backtest
    bt = Backtest(period_df, MLStrategy, cash=1000000, commission=.001)
    stats = bt.run()
    
    # Sonuçları Sözlük Olarak Döndür
    result = {
        "period": period_name,
        "start_date": start_date,
        "end_date": end_date,
        "return": stats['Return [%]'],
        "buy_hold": stats['Buy & Hold Return [%]'],
        "max_drawdown": stats['Max. Drawdown [%]'],
        "trades": stats['# Trades'],
        "win_rate": stats['Win Rate [%]'],
        "sharpe": stats['Sharpe Ratio']
    }
    
    print(f"Sonuç: Getiri %{stats['Return [%]']:.2f} (Al-Tut: %{stats['Buy & Hold Return [%]']:.2f})")
    
    # HTML Raporu da kaydedelim (Opsiyonel)
    report_dir = "reports/robustness"
    os.makedirs(report_dir, exist_ok=True)
    bt.plot(filename=os.path.join(report_dir, f"Robustness_{period_name}.html"), open_browser=False)
    
    return result

def main():
    # 1. Model Yükle
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        print("Model bulunamadı! Önce 'models/train_xgboost.py' çalıştırın.")
        return

    model = joblib.load(model_path)
    print("✅ XGBoost V1 Modeli Yüklendi.")

    # 2. Tüm Veriyi İndir (Uzun dönem)
    print("Veri indiriliyor (2020-Bugün)...")
    df = yf.download('BTC-USD', start='2020-01-01', interval='1d')
    
    # MultiIndex Düzeltme
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Feature Engineering
    df = prepare_data_for_ml(df)
    
    # 3. Dönemleri Tanımla
    periods = [
        {"name": "2021_Bull_Run", "start": "2020-10-01", "end": "2021-11-10"},     # Boğa (Ekim 2020 -> Kasım 2021 Zirvesi)
        {"name": "2022_Bear_Market", "start": "2021-11-11", "end": "2022-12-31"},  # Ayı (Zirveden 2022 Sonuna)
        {"name": "2023_Recovery", "start": "2023-01-01", "end": "2023-12-31"},     # Toparlanma/Yatay
        {"name": "2024_Recent", "start": "2024-01-01", "end": str(datetime.date.today())} # Güncel
    ]
    
    results = []
    
    for p in periods:
        res = run_backtest_for_period(df, model, p['name'], p['start'], p['end'])
        if res:
            results.append(res)
            
    # 4. Genel Raporu Kaydet
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"reports/Robustness_Check_{timestamp}.json"
    
    summary = {
        "model": "XGBoost V1",
        "description": "Farklı piyasa döngülerinde (Bull, Bear, Recovery) sağlamlık testi.",
        "timestamp": timestamp,
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Tablo Olarak Göster
    print("\n" + "="*80)
    print(f"{'DÖNEM':<20} | {'GETİRİ (%)':<12} | {'AL-TUT (%)':<12} | {'DRAWDOWN':<10} | {'İŞLEM':<6}")
    print("="*80)
    for r in results:
        print(f"{r['period']:<20} | {r['return']:>10.2f} % | {r['buy_hold']:>10.2f} % | {r['max_drawdown']:>9.1f}% | {r['trades']:>5}")
    print("="*80)
    print(f"\nRapor Kaydedildi: {filename}")

if __name__ == "__main__":
    main()
