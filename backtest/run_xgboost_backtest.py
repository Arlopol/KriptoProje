import sys
import os
import pandas as pd
import joblib
import json
import datetime
from backtesting import Backtest

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml
from strategies.ml_strategy import MLStrategy

def run_xgboost_backtest():
    try:
        print("--- XGBoost Backtest Başlatılıyor ---", flush=True)
        
        # 1. Veri ve Model Yükle
        symbol = 'BTC-USD'
        loader = DataLoader()
        df = loader.fetch_data(symbol, period='5y', interval='1d')
        print(f"Veri yüklendi. Satır sayısı: {len(df)}", flush=True)
        
        model_path = "models/saved_models/xgb_btc_v1.joblib"
        if not os.path.exists(model_path):
            print("Model dosyası bulunamadı! Önce 'models/train_xgboost.py' çalıştırın.", flush=True)
            return
            
        model = joblib.load(model_path)
        print("Model yüklendi.", flush=True)
        
        df_processed = prepare_data_for_ml(df.copy())
        print(f"Veri işlendi. Satır sayısı: {len(df_processed)}", flush=True)
        
        print("✅ XGBoost Modeli ve Veri yüklendi.", flush=True)
        
        features = [col for col in df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
        
        # 3. Model Tahminlerini Yap
        print("Tahminler üretiliyor...", flush=True)
        X = df_processed[features]
        
        # XGBoost Olasılıkları
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        print("Tahminler tamam.", flush=True)
        
        df_processed['ML_Signal'] = preds
        df_processed['ML_Prob'] = probs
        
        # 4. Backtest (Test Seti)
        split_idx = int(len(df_processed) * 0.8)
        test_df = df_processed.iloc[split_idx:]
        
        print(f"Backtest Dönemi: {test_df.index[0]} - {test_df.index[-1]}", flush=True)
        
        # Backtest Başlat 
        # DİKKAT: Komisyon oranını binde 1 yaptık (.001)
        bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.001)
        stats = bt.run()
        
        print(stats, flush=True)
        
        # 5. Raporlama
        report_dir = "reports"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename_base = f"Audio_XGBoost_{timestamp}"
        
        # HTML Kaydet
        html_path = os.path.join(report_dir, f"{filename_base}.html")
        bt.plot(filename=html_path, open_browser=False)
        print(f"HTML rapor kaydedildi: {html_path}", flush=True)
        
        # Metrikleri Yükle
        model_metrics = {}
        metrics_path = "reports/ml_metrics_xgboost.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)

        # Buy & Hold
        first_price = test_df['Close'].iloc[0]
        buy_hold_equity = (test_df['Close'] / first_price * 1000000).tolist()
        
        equity_curve = stats['_equity_curve']
        equity_dates = [str(d) for d in equity_curve.index]
        print(f"Equity curve oluşturuldu: {len(equity_curve)} nokta", flush=True)
        
        if len(buy_hold_equity) > len(equity_dates):
            buy_hold_equity = buy_hold_equity[:len(equity_dates)]
            
        summary_json = {
            "strategy": "ML_XGBoost_V1",
            "description": "XGBoost + Prob Threshold + Commission 0.1%. Düşük komisyon ve güçlü model ile test.",
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
                "dates": equity_dates,
                "equity": [float(x) for x in equity_curve['Equity'].values],
                "drawdown": [float(x) for x in equity_curve['DrawdownPct'].values],
                "buy_hold": buy_hold_equity
            },
            "files": {
                "html": os.path.basename(html_path)
            }
        }
        
        json_path = os.path.join(report_dir, f"{filename_base}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=4)
            
        print(f"Kayıt: {json_path}", flush=True)
        
    except Exception as e:
        print("\n***************************", flush=True)
        print(f"HATA OLUŞTU: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("***************************\n", flush=True)

if __name__ == "__main__":
    run_xgboost_backtest()
