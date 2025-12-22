import sys
import os
import joblib
import json
import datetime
import numpy as np
import pandas as pd
import warnings
from tensorflow.keras.models import load_model

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest
from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml
from strategies.ml_strategy import MLStrategy

# Uyarıları gizle
warnings.filterwarnings("ignore")

def run_lstm_backtest():
    try:
        print("--- LSTM (Deep Learning) Backtest Başlatılıyor ---", flush=True)
        
        # 1. Veri Yükle
        symbol = 'BTC-USD'
        loader = DataLoader()
        df = loader.fetch_data(symbol, period='5y', interval='1d')
        print(f"Veri yüklendi. Satır sayısı: {len(df)}", flush=True)
        
        # 2. Model ve Scaler Yükle
        model_path = "models/saved_models/lstm_btc_v2.keras"
        scaler_path = "models/saved_models/lstm_scaler_v2.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("Model veya Scaler dosyası yok! Önce 'models/train_lstm.py' çalıştırın.", flush=True)
            return
            
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ LSTM Modeli (V2) ve Scaler yüklendi.", flush=True)
        
        # 3. Veri İşleme
        df_processed = prepare_data_for_ml(df.copy())
        df_processed.dropna(inplace=True)
        
        features = [col for col in df_processed.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]
        
        # 4. LSTM için Veri Hazırlama (Sequence)
        X_values = df_processed[features].values
        
        # Backtest'te SADECE transform yapılır. Fit yapılmaz.
        try:
             X_scaled = scaler.transform(X_values)
        except:
             # Eğer feature sayısı tutmazsa mecburen fit (riskli)
             print("⚠️ Scaler uyumsuzluğu, mecburen fit ediliyor.")
             X_scaled = scaler.fit_transform(X_values)
        
        SEQ_LENGTH = 30 # Eğitimdekiyle aynı olmalı
        
        preds = []
        probs = []
        
        # Batch tahmini yapmak daha hızlı olur
        X_seq_list = []
        valid_indices = []
        
        for i in range(len(X_scaled) - SEQ_LENGTH):
            X_seq_list.append(X_scaled[i : i + SEQ_LENGTH])
            valid_indices.append(df_processed.index[i + SEQ_LENGTH])
            
        X_seq_tensor = np.array(X_seq_list)
        
        print(f"Tahmin edilecek veri boyutu: {X_seq_tensor.shape}", flush=True)
        
        # Tahmin
        y_pred_probs = model.predict(X_seq_tensor, verbose=0)
        probs = y_pred_probs.flatten()
        preds = (probs > 0.5).astype(int)
        
        # Olasılık dağılımını göster
        print(f"Backtest Olasılıkları: Min={np.min(probs):.4f}, Max={np.max(probs):.4f}, Ort={np.mean(probs):.4f}")
        
        # Tahminleri DataFrame'e eşle
        df_preds = pd.DataFrame({
            'ML_Signal': preds, 
            'ML_Prob': probs
        }, index=valid_indices)
        
        # Ana DataFrame ile birleştir (Inner Join)
        df_backtest = df_processed.join(df_preds, how='inner')
        
        # 5. Backtest (Test Seti - Son %20)
        split_idx = int(len(df_backtest) * 0.8)
        test_df = df_backtest.iloc[split_idx:]
        
        print(f"Backtest Dönemi: {test_df.index[0]} - {test_df.index[-1]}", flush=True)
        
        # Standart Stratejiye Dönüş (Model iyiyse buna gerek kalmaz)
        # Ama yine de Threshold'ları class içinde override etmeden MLStrategy kullanırsak
        # Default 0.60/0.40 çalışır.
        
        bt = Backtest(test_df, MLStrategy, cash=1000000, commission=.001)
        stats = bt.run()
        
        print(stats, flush=True)
        
        # 6. Raporlama
        report_dir = "reports"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename_base = f"LSTM_DeepLearning_V2_{timestamp}"
        
        # HTML Kaydet
        html_path = os.path.join(report_dir, f"{filename_base}.html")
        bt.plot(filename=html_path, open_browser=False)
        print(f"HTML kaydedildi: {html_path}", flush=True)
        
        # Metrikleri Yükle
        model_metrics = {}
        metrics_path = "reports/ml_metrics_lstm.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)

        # JSON Oluştur
        summary_json = {
            "strategy": "ML_LSTM_HighRisk",
            "description": "LSTM (Deep Learning) - Yüksek Risk Modu. Güven eşiği %51'e çekildi. Nötr bölge daraltıldı.",
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
    run_lstm_backtest()
