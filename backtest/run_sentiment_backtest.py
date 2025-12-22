import sys
import os
import joblib
import pandas as pd
import json
import datetime
import warnings

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest
from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml
from data.sentiment_loader import SentimentLoader
from strategies.ml_strategy import MLStrategy

# Uyarıları gizle
warnings.filterwarnings("ignore")

def run_sentiment_backtest():
    try:
        print("--- Sentiment Destekli Backtest Başlatılıyor ---", flush=True)
        
        # 1. Veri ve Sentiment Yükle
        symbol = 'BTC-USD'
        price_loader = DataLoader()
        df_price = price_loader.fetch_data(symbol, period='5y', interval='1d')
        
        sent_loader = SentimentLoader()
        df_sentiment = sent_loader.fetch_fear_and_greed_data(limit=0)
        
        if df_sentiment is None:
            print("Sentiment verisi yok!")
            return

        # 2. Merge (Date üzerinden)
        if 'Date' not in df_price.columns:
            df_price = df_price.reset_index()
        
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        
        df_merged = pd.merge(df_price, df_sentiment, on='Date', how='inner')
        print(f"Veri Birleştirildi: {len(df_merged)} gün", flush=True)
        
        # 3. Model Yükle
        model_path = "models/saved_models/xgb_sentiment_v1.joblib"
        if not os.path.exists(model_path):
            print("Model dosyası bulunamadı! Önce 'models/train_xgboost_sentiment.py' çalıştırın.", flush=True)
            return
            
        model = joblib.load(model_path)
        print("✅ Sentiment Modeli yüklendi.", flush=True)
        
        # 4. Özellik Mühendisliği (Aynı işlemler)
        df_processed = prepare_data_for_ml(df_merged)
        
        # Ekstra sentimental özellikler (Eğitimde ne yaptıysak aynısı)
        df_processed['FNG_Change'] = df_processed['FNG_Value'].pct_change()
        df_processed['FNG_MA_7'] = df_processed['FNG_Value'].rolling(window=7).mean()
        
        df_processed.dropna(inplace=True)
        
        features = [col for col in df_processed.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'FNG_Class']]
        
        X = df_processed[features]
        
        # 5. Tahminler
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        df_processed['ML_Signal'] = preds
        df_processed['ML_Prob'] = probs
        
        # 6. Backtest (Test Seti - Son %20)
        split_idx = int(len(df_processed) * 0.8)
        test_df = df_processed.iloc[split_idx:]
        
        print(f"Backtest Dönemi: {test_df.iloc[0]['Date']} - {test_df.iloc[-1]['Date']}", flush=True)
        
        # Backtest'te datetime index gerekir
        test_df = test_df.set_index('Date')
        
        # Backtest Başlat (Komisyon %0.1)
        bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.001)
        stats = bt.run()
        
        print(stats, flush=True)
        
        # 7. Raporlama
        report_dir = "reports"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename_base = f"Sentiment_XGBoost_{timestamp}"
        
        # HTML Kaydet
        html_path = os.path.join(report_dir, f"{filename_base}.html")
        bt.plot(filename=html_path, open_browser=False)
        print(f"HTML kaydedildi: {html_path}", flush=True)
        
        # Metrikleri Yükle (Eğitimden gelen)
        model_metrics = {}
        metrics_path = "reports/ml_metrics_sentiment.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)

        # JSON Oluştur
        summary_json = {
            "strategy": "ML_XGBoost_Sentiment",
            "description": "XGBoost + Fear&Greed Index. Sentiment verisi ile piyasa duygusu analizi.",
            "symbol": symbol,
            "date": timestamp,
            "initial_capital": 1000000,
            "model_metrics": model_metrics, 
            "metrics": {
                "return": float(stats['Return [%]']),
                "buy_hold_return": float(stats['Buy & Hold Return [%]']),
                "win_rate": float(stats['Win Rate [%]']),
                "max_drawdown": float(stats['Max. Drawdown [%]']),
                "sharpe": float(stats['Sharpe Ratio']),
                "trades": int(stats['# Trades']),
                "final_equity": float(stats['Equity Final [$]'])
            },
            "equity_curve": {
                "dates": [str(d) for d in stats['_equity_curve'].index],
                "equity": [float(x) for x in stats['_equity_curve']['Equity'].values],
                "buy_hold": (test_df['Close'] / test_df['Close'].iloc[0] * 1000000).tolist()
            },
            "files": {
                "html": os.path.basename(html_path)
            }
        }
        
        json_path = os.path.join(report_dir, f"{filename_base}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=4)
            
        print(f"Rapor Kaydedildi: {json_path}", flush=True)

    except Exception as e:
        print(f"HATA: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sentiment_backtest()
