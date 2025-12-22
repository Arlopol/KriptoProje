import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from backtesting import Backtest
import pandas_ta as ta
import os
import json
import datetime
import sys

# Proje kök dizinini ekle (Modül importları için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy import MLStrategy

def prepare_data():
    # 1. Fiyat Verisi
    print("Fiyat verisi hazırlanıyor...", flush=True)
    df_price = yf.download('BTC-USD', period='2y', interval='1d') 
    
    # YFinance MultiIndex düzeltme
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [col[0] for col in df_price.columns]
        
    df_price.reset_index(inplace=True)
    # Timestamp olarak bırak (backtesting.py için önemli)
    df_price['Date'] = pd.to_datetime(df_price['Date']) 
    df_price.set_index('Date', inplace=True)
    
    # 2. On-Chain Verisi
    csv_path = "data/onchain_data_test.csv"
    if not os.path.exists(csv_path):
        print("HATA: On-Chain verisi yok.")
        return None
        
    df_onchain = pd.read_csv(csv_path)
    df_onchain['Date'] = pd.to_datetime(df_onchain['Date'])
    df_onchain.set_index('Date', inplace=True)
    
    # 3. Birleştirme
    df = df_price.join(df_onchain, how='inner')
    
    # 4. Feature Engineering (Eğitimdekiyle AYNI olmalı)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    onchain_cols = ['Hash_Rate', 'Difficulty', 'Transaction_Count', 'Unique_Addresses', 'Miners_Revenue']
    for col in onchain_cols:
        if col in df.columns:
            df[f'{col}_ROC_1'] = df[col].pct_change(1)
            df[f'{col}_ROC_7'] = df[col].pct_change(7)
            
    df.dropna(inplace=True)
    return df

def run_backtest():
    # 1. Model Yükle
    model_path = "models/saved_models/xgboost_onchain.json"
    if not os.path.exists(model_path):
        print("Model bulunamadı! Önce 'models/train_xgboost_onchain.py' çalıştırın.")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("✅ Model yüklendi.", flush=True)
    
    # 2. Veri
    df = prepare_data()
    if df is None: return
    
    # Feature Sütunları (Eğitimdekiyle birebir aynı sıra olmalı)
    feature_cols = [
        'RSI', 'MACD', 'SMA_50', 'SMA_200', 
        'Hash_Rate_ROC_1', 'Hash_Rate_ROC_7',
        'Difficulty_ROC_1', 'Difficulty_ROC_7',
        'Transaction_Count_ROC_1', 'Transaction_Count_ROC_7',
        'Unique_Addresses_ROC_1', 'Unique_Addresses_ROC_7',
        'Miners_Revenue_ROC_1', 'Miners_Revenue_ROC_7'
    ]
    
    # Eksik feature varsa uyar/düzelt (veya drop et, ama model aynı inputu bekler)
    # Burada backtest verisi hazırlarken tüm sütunların oluştuğundan emin olduk.
    # Ancak yine de kontrol:
    valid_features = [f for f in feature_cols if f in df.columns]
    X = df[valid_features]
    
    # 3. Tahmin
    print("Tahminler yapılıyor...", flush=True)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] # Artış ihtimali
    
    # Backtest DataFrame'ine ekle
    df['ML_Signal'] = preds
    df['ML_Prob'] = probs
    
    # 4. Backtest (Son 1 Yıl veya %50)
    split_idx = int(len(df) * 0.5)
    test_df = df.iloc[split_idx:]
    
    print(f"Backtest Dönemi: {test_df.index[0]} - {test_df.index[-1]}")
    
    bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.001)
    stats = bt.run()
    
    print(stats)
    
    # 5. Raporlama
    report_dir = "reports"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename_base = f"XGBoost_OnChain_{timestamp}"
    
    # HTML Kaydet
    html_path = os.path.join(report_dir, f"{filename_base}.html")
    bt.plot(filename=html_path, open_browser=False)
    
    # Model Metriklerini Yükle (Eğitimden)
    model_metrics = {}
    metrics_path = "reports/onchain_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            model_metrics = json.load(f)

    # Detaylı JSON Oluştur
    summary_json = {
        "strategy": "OnChain_XGBoost",
        "description": "On-Chain verileri (Hash Rate, Difficulty, Active Addresses) ile güçlendirilmiş XGBoost modeli.",
        "symbol": "BTC-USD",
        "date": timestamp,
        "initial_capital": 1000000,
        "model_metrics": model_metrics,
        "metrics": {
            "return": stats['Return [%]'],
            "buy_hold_return": stats['Buy & Hold Return [%]'],
            "win_rate": stats['Win Rate [%]'],
            "max_drawdown": stats['Max. Drawdown [%]'],
            "sharpe": stats['Sharpe Ratio'],
            "trades": stats['# Trades'],
            "final_equity": stats['Equity Final [$]']
        },
        "equity_curve": {
            "dates": stats['_equity_curve'].index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "equity": stats['_equity_curve']['Equity'].tolist(),
            "drawdown": stats['_equity_curve']['DrawdownPct'].tolist(),
            # Buy & Hold sütunu backtesting.py default çıktısında olmayabilir, kontrol edelim
            # Genelde yoktur, manuel hesaplanabilir ama şimdilik boş geçelim
             "buy_hold": [] 
        }
    }
    
    # Buy & Hold eğrisini manuel ekleyelim (Yaklaşık)
    # Test verisinin Close fiyatını normalize edip sermaye ile çarpabiliriz
    try:
        if '_equity_curve' in stats:
             # Zaman serisi boyunca Close fiyatlarını bulmak zor olabilir çünkü _equity_curve sadece Equity tutar.
             # Ancak backtesting.py bazen OHLC verisini de saklar.
             pass
    except:
        pass

    with open(os.path.join(report_dir, f"{filename_base}.json"), "w") as f:
        json.dump(summary_json, f, indent=4)
        
    print(f"Rapor Kaydedildi: reports/{filename_base}.json")

if __name__ == "__main__":
    run_backtest()
