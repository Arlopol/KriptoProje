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
# YENİ STRATEJİ: Trend Filtreli Olanı İçe Aktar
from strategies.ml_strategy_trend import MLStrategyTrend

def run_backtest_for_period(df, model, period_name, start_date, end_date):
    print(f"\n--- Test: {period_name} ({start_date} - {end_date}) ---")
    
    # Tarih Aralığı Filtresi
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    period_df = df.loc[mask].copy()
    
    if period_df.empty:
        print(f"UYARI: {period_name} için veri yok!")
        return None

    # Tahmin
    features = [col for col in period_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date', 'ML_Signal', 'ML_Prob']]
    X = period_df[features]
    
    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return None

    period_df['ML_Signal'] = preds
    period_df['ML_Prob'] = probs
    
    # Backtest - YENİ STRATEJİ SINIFI KULLANILIYOR
    bt = Backtest(period_df, MLStrategyTrend, cash=1000000, commission=.001)
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
    
    print(f"Sonuç: Getiri %{stats['Return [%]']:.2f}")
    
    # HTML Rapor
    report_dir = "reports/robustness_trend"
    os.makedirs(report_dir, exist_ok=True)
    bt.plot(filename=os.path.join(report_dir, f"Robustness_Trend_{period_name}.html"), open_browser=False)
    
    return result

def main():
    # 1. Model Yükle (Aynı model, sadece strateji filtreliyor)
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        print("Model bulunamadı! Önce 'models/train_xgboost.py' çalıştırın.")
        return

    model = joblib.load(model_path)
    print("✅ XGBoost V1 Modeli Yüklendi.")

    # 2. Veri İndir
    print("Veri indiriliyor...")
    df = yf.download('BTC-USD', start='2020-01-01', interval='1d')
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Feature Engineering (SMA_200 burada hesaplanıyor olmalı)
    df = prepare_data_for_ml(df)
    
    # SMA_200 Kontrolü
    if 'SMA_200' not in df.columns:
        print("HATA: SMA_200 sütunu veride yok! Strategies/ml_strategy_trend.py çalışmaz.")
        # Basitçe ekleyelim garanti olsun
        import pandas_ta as ta
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        print("SMA_200 manuel olarak eklendi.")

    # 3. Dönemler
    periods = [
        {"name": "2021_Bull_Run", "start": "2020-10-01", "end": "2021-11-10"},
        {"name": "2022_Bear_Market", "start": "2021-11-11", "end": "2022-12-31"},
        {"name": "2023_Recovery", "start": "2023-01-01", "end": "2023-12-31"},
        {"name": "2024_Recent", "start": "2024-01-01", "end": str(datetime.date.today())}
    ]
    
    results = []
    for p in periods:
        res = run_backtest_for_period(df, model, p['name'], p['start'], p['end'])
        if res:
            results.append(res)
            
    # 4. Raporla
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"reports/Robustness_Trend_{timestamp}.json"
    
    summary = {
        "model": "XGBoost V1 (+Trend Filter)", # İsimde farklılık olsun
        "description": "Boğa piyasasında (Fiyat > SMA200) Short açmayı yasaklayan filtreli test.",
        "timestamp": timestamp,
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nRapor Kaydedildi: {filename}")
    
    # Karşılaştırma kolaylığı için ekrana bas
    print("\n" + "="*80)
    print(f"{'DÖNEM':<20} | {'YENİ GETİRİ (%)':<15} | {'ESKİ (Al-Tut)'}")
    print("="*80)
    for r in results:
        print(f"{r['period']:<20} | {r['return']:>13.2f} % | {r['buy_hold']:>10.2f} %")
    print("="*80)

if __name__ == "__main__":
    main()
