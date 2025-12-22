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
from data.feature_engineering import add_technical_indicators, prepare_data_for_ml
from strategies.ml_strategy import MLStrategy

def run_ml_backtest():
    print("--- ML Destekli Backtest Başlatılıyor ---")
    
    # 1. Veri ve Model Yükle
    symbol = 'BTC-USD'
    loader = DataLoader()
    # 5 yıllık veriyi tekrar çekelim (Train sırasında kullanılan veriynin aynısı olmalı)
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    model_path = "models/saved_models/rf_btc_v1.joblib"
    if not os.path.exists(model_path):
        print("Model dosyası bulunamadı! Önce 'models/train_model.py' çalıştırın.")
        return
        
    model = joblib.load(model_path)
    print("✅ Model ve Veri yüklendi.")

    # 2. Özellikleri (Features) Tekrar Üret
    # Model eğitimi sırasındaki mantığın AYNISI olmalı
    df = add_technical_indicators(df)
    
    # Özellik sütunlarını seç
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # 3. Model Tahminlerini Yap (Tüm Veri Seti İçin)
    print("Tahminler üretiliyor...")
    X = df[features]
    
    # Olasılıkları al (Class 1 ihtimali -> Yükseliş)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    # DataFrame'e ekle (Strateji sınıfı bunları okuyacak)
    df['ML_Signal'] = preds
    df['ML_Prob'] = probs
    
    # 4. Backtest (Sadece Test Dönemi İçin mi? Yoksa hepsi mi?)
    # Gerçekçi olması için sadece eğitimin bittiği yerden sonrasını (Test setini) simüle edelim.
    # train_model.py'da %80 train kullanmıştık.
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    print(f"Backtest Dönemi: {test_df.index[0]} - {test_df.index[-1]}")
    
    # Backtest Başlat
    bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    
    print(stats)
    
    # 5. Raporlama (Dashboard İçin Kayıt)
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename_base = f"Audio_RandomForest_{timestamp}" # Dosya adını farklılaştıralım
    
    # HTML Kaydet
    html_path = os.path.join(report_dir, f"{filename_base}.html")
    bt.plot(filename=html_path, open_browser=False)
    
    # JSON Kaydet
    def convert_numpy(obj):
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        elif hasattr(obj, 'item'): return obj.item()
        return str(obj)

    # Equity Curve Verisi
    equity_curve = stats['_equity_curve']
    equity_dates = [str(d) for d in equity_curve.index]
    
    # Al-ve-Tut (Buy & Hold) Eğrisini Hesapla
    # Kapanış fiyatlarının normalize edilmesi ve başlangıç sermayesi ile çarpılması
    first_price = test_df['Close'].iloc[0]
    buy_hold_equity = (test_df['Close'] / first_price * 1000000).tolist()
    
    if len(buy_hold_equity) > len(equity_dates):
        buy_hold_equity = buy_hold_equity[:len(equity_dates)]
    
    # ML Metriklerini Yükle (Eğer varsa)
    model_metrics = {}
    metrics_path = "reports/ml_metrics_latest.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            model_metrics = json.load(f)

    summary_json = {
        "strategy": "ML_RF_Prob_Threshold", 
        "description": "Random Forest + Confidence Thresholds. Sadece model %60+ eminse Long, %60+ düşecek derse Short açar (%40-60 arası nakit).",
        "symbol": symbol,
        "date": timestamp,
        "initial_capital": 1000000,
        "model_metrics": model_metrics, # Metrikleri buraya gömüyoruz
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
        json.dump(summary_json, f, indent=4, default=convert_numpy)
        
    print("\n" + "="*40)
    print(f"🚀 ML BACKTEST TAMAMLANDI")
    print("="*40)
    print(f"💰 Toplam Getiri:      %{stats['Return [%]']:.2f}")
    print(f"📉 Al-ve-Tut:          %{stats['Buy & Hold Return [%]']:.2f}")
    print(f"📊 Fark:               %{stats['Return [%]'] - stats['Buy & Hold Return [%]']:.2f}")
    print("-" * 40)
    print(f"Kayıt: {json_path}")

if __name__ == "__main__":
    run_ml_backtest()
