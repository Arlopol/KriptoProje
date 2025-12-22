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

def run_ml_backtest_v2():
    print("--- ML Backtest V2 (Gelişmiş Özellikler) Başlatılıyor ---")
    
    # 1. Veri ve Model Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    model_path = "models/saved_models/rf_btc_v2.joblib"
    if not os.path.exists(model_path):
        print("Model V2 dosyası bulunamadı! Önce 'models/train_model_v2.py' çalıştırın.")
        return
        
    model = joblib.load(model_path)
    # Modelin hangi özelliklerle eğitildiğini anlamak için feature engineering çalıştır
    df_processed = prepare_data_for_ml(df.copy())
    
    print("✅ Model V2 ve Veri yüklendi.")
    
    # Features sütunlarını otomatik al (Target hariç)
    features = [col for col in df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    
    # 3. Model Tahminlerini Yap
    print("Tahminler üretiliyor...")
    X = df_processed[features]
    
    # V2 Modeli ile Olasılıkları Al
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    # Orijinal DataFrame'e sinyalleri ekle (İndeks eşleşmesine dikkat)
    # prepare_data_for_ml DropNA yaptığı için satır sayısı azalmıştır.
    df_processed['ML_Signal'] = preds
    df_processed['ML_Prob'] = probs
    
    # 4. Backtest (Test Seti Üzerinde)
    split_idx = int(len(df_processed) * 0.8)
    test_df = df_processed.iloc[split_idx:]
    
    print(f"Backtest Dönemi: {test_df.index[0]} - {test_df.index[-1]}")
    
    # Backtest Başlat
    bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    
    print(stats)
    
    # 5. Raporlama
    report_dir = "reports"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename_base = f"Audio_RandomForest_V2_{timestamp}"
    
    # HTML Kaydet
    html_path = os.path.join(report_dir, f"{filename_base}.html")
    bt.plot(filename=html_path, open_browser=False)
    
    # V2 Metriklerini Yükle
    model_metrics = {}
    metrics_path = "reports/ml_metrics_v2.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            model_metrics = json.load(f)

    # Buy & Hold Curve
    first_price = test_df['Close'].iloc[0]
    buy_hold_equity = (test_df['Close'] / first_price * 1000000).tolist()
    
    # JSON Kaydet
    equity_curve = stats['_equity_curve']
    equity_dates = [str(d) for d in equity_curve.index]
    
    if len(buy_hold_equity) > len(equity_dates):
        buy_hold_equity = buy_hold_equity[:len(equity_dates)]
        
    summary_json = {
        "strategy": "ML_RandomForest_V2",
        "description": "RF V2 (Gelişmiş Özellikler: Lag + Volatilite). Güven Eşiği: %60. Stop Loss %5.",
        "symbol": symbol,
        "date": timestamp,
        "initial_capital": 1000000,
        "model_metrics": model_metrics, # V2 Metrikleri
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
        
    print(f"Kayıt: {json_path}")

if __name__ == "__main__":
    run_ml_backtest_v2()
