
import sys
import os
import pandas as pd
import joblib
from backtesting import Backtest
import json

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy_logging import MLStrategyLogging
from data.feature_engineering import prepare_data_for_ml

def run_scenario(start_date, end_date, initial_capital=10000):
    """
    Belirli bir tarih aralığı için 'Laboratuvar Testi' çalıştırır.
    """
    
    # 1. Veri Yükle (Cache varsa kullan)
    data_path = "data/raw/BTC-USD_5y_1d.csv"
    if not os.path.exists(data_path):
        return {"error": "Veri dosyası bulunamadı."}

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Tarih Filtreleme
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    scenario_df = df.loc[mask]
    
    if scenario_df.empty or len(scenario_df) < 50: # En az 50 bar olsun (SMA hesabı için önceki veriye de ihtiyaç var aslında)
        # SMA 200 için geçmiş verinin de önünde olması gerekir.
        # Bu yüzden filtrelemeyi feature engineering SONRASI yapmalıyız.
        pass

    # 2. Modeli Yükle
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        return {"error": "Model dosyası bulunamadı."}
        
    model = joblib.load(model_path)
    
    # 3. İndikatörleri Hesapla (Tüm veri üzerinde)
    full_df_processed = prepare_data_for_ml(df.copy())
    
    # 4. Tahminleri Yap
    features = [col for col in full_df_processed.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']]
    full_df_processed['ML_Signal'] = model.predict(full_df_processed[features])
    full_df_processed['ML_Prob'] = model.predict_proba(full_df_processed[features])[:, 1]
    
    # 5. Şimdi Tarihi Filtrele (İndikatörler bozulmasın diye sonda kestik)
    # Başlangıç tarihinden önceki 200 günü de almamız gerekmiyor çünkü indikatörler zaten hesaplandı.
    # Ancak Backtest kütüphanesi ilk barda "position" sıfır başlar.
    
    scenario_data = full_df_processed.loc[start_date:end_date]
    
    if scenario_data.empty:
        return {"error": "Seçilen tarih aralığında veri yok."}
        
    # --- FRACTIONAL TRADING WORKAROUND ---
    # Backtesting kütüphanesi tam sayı (integer) adetlerde işlem yapmaya zorluyor.
    # Bitcoin $60k iken $10k sermaye ile 0.16 BTC alamıyor.
    # Bu yüzden Fiyatları ve SMA'yı 1000'e bölüyoruz (mBTC gibi).
    SCALE_FACTOR = 1000.0
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'SMA_200']
    for c in cols_to_scale:
        if c in scenario_data.columns:
            scenario_data.loc[:, c] = scenario_data[c] / SCALE_FACTOR
            
    # 6. Backtest
    bt = Backtest(scenario_data, MLStrategyLogging, cash=initial_capital, commission=0.001, trade_on_close=True)
    stats = bt.run()
    
    # Strateji instance'ına erişip logları alalım
    strategy_instance = stats['_strategy']
    logs = strategy_instance.decision_logs
    
    # --- DOĞRULUK (ACCURACY) HESABI ---
    # Bu senaryo döneminde model ne kadar başarılıydı?
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Hedef (Target): Ertesi gün yükseldi mi? (1: Evet, 0: Hayır)
    scenario_data['Actual_Target'] = (scenario_data['Close'].shift(-1) > scenario_data['Close']).astype(int)
    
    # Son satırın Target'ı NaN olur (ertesi gün yok), onu çıkaralım
    valid_data = scenario_data.dropna()
    
    acc, prec, rec, f1 = 0, 0, 0, 0
    
    if not valid_data.empty:
        # ML_Signal gerçek tahminlerdir (0 veya 1)
        y_true = valid_data['Actual_Target']
        y_pred = valid_data['ML_Signal']
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
    # Eğitim Metriklerini Oku (Kıyaslama için)
    train_metrics = {}
    try:
        with open("reports/ml_metrics_xgboost.json", "r") as f:
            train_metrics = json.load(f)
    except:
        pass
    
    # Sonuçları hazırla
    result = {
        "start_date": str(scenario_data.index[0].date()),
        "end_date": str(scenario_data.index[-1].date()),
        "duration_days": (scenario_data.index[-1] - scenario_data.index[0]).days,
        "initial_capital": initial_capital,
        "final_equity": stats['Equity Final [$]'],
        "return_pct": stats['Return [%]'],
        "max_drawdown": stats['Max. Drawdown [%]'],
        "win_rate": stats['Win Rate [%]'],
        "total_trades": stats['# Trades'],
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        },
        "train_metrics": train_metrics,
        "logs": logs # Tüm logları gönder
    }
    
    return result

if __name__ == "__main__":
    # Test
    res = run_scenario("2024-01-01", "2024-06-01")
    print(json.dumps(res, indent=2))
