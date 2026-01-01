
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
from data.data_loader import DataLoader

def run_scenario(start_date, end_date, initial_capital=10000, symbol='BTC-USD',
                 buy_threshold=0.60, sell_threshold=0.40, stop_loss=0.05, take_profit=0.15, use_trend=True,
                 use_dynamic_sizing=False):
    """
    Belirli bir tarih aralığı için 'Laboratuvar Testi' çalıştırır.
    """
    
    # 1. Veri Yükle (Dinamik)
    loader = DataLoader()
    # 5 yıllık veri çekelim ki indikatörler hesaplansın
    df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    if df is None or df.empty:
        return {"error": f"{symbol} için veri bulunamadı."}

    # df zaten index=Date olarak geliyor ama kontrol
    if not isinstance(df.index, pd.DatetimeIndex):
         if 'Date' in df.columns:
             df['Date'] = pd.to_datetime(df['Date'])
             df.set_index('Date', inplace=True)
    
    # Tarih Filtreleme (Önce geniş aralık, sonra daraltacağız)
    # Feature engineering için tüm veriye ihtiyaç var.
    
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
    
    # 5. Şimdi Tarihi Filtrele for Backtest
    mask = (full_df_processed.index >= pd.to_datetime(start_date)) & (full_df_processed.index <= pd.to_datetime(end_date))
    scenario_data = full_df_processed.loc[mask]
    
    if scenario_data.empty:
        return {"error": "Seçilen tarih aralığında veri yok."}
        
    # --- FRACTIONAL TRADING WORKAROUND ---
    SCALE_FACTOR = 1000.0
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'SMA_200']
    for c in cols_to_scale:
        if c in scenario_data.columns:
            scenario_data.loc[:, c] = scenario_data[c] / SCALE_FACTOR
            
    # 6. Backtest
    bt = Backtest(scenario_data, MLStrategyLogging, cash=initial_capital, commission=0.001, trade_on_close=True)
    
    stats = bt.run(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        use_trend_filter=use_trend,
        use_dynamic_sizing=use_dynamic_sizing
    )
    
    # Strateji instance'ına erişip logları alalım
    strategy_instance = stats['_strategy']
    logs = strategy_instance.decision_logs
    
    # --- DOĞRULUK (ACCURACY) HESABI ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    scenario_data['Actual_Target'] = (scenario_data['Close'].shift(-1) > scenario_data['Close']).astype(int)
    valid_data = scenario_data.dropna()
    
    acc, prec, rec, f1 = 0, 0, 0, 0
    if not valid_data.empty:
        y_true = valid_data['Actual_Target']
        y_pred = valid_data['ML_Signal']
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
    # Eğitim Metriklerini Oku
    train_metrics = {}
    try:
        with open("reports/ml_metrics_xgboost.json", "r") as f:
            train_metrics = json.load(f)
    except: pass
    
    # Sonuçları hazırla
    result = {
        "start_date": str(scenario_data.index[0].date()),
        "end_date": str(scenario_data.index[-1].date()),
        "duration_days": (scenario_data.index[-1] - scenario_data.index[0]).days,
        "initial_capital": initial_capital,
        "final_equity": stats['Equity Final [$]'],
        "return_pct": stats['Return [%]'],
        "bh_return_pct": stats.get('Buy & Hold Return [%]', 0),
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
        "logs": logs
    }
    
    return result

if __name__ == "__main__":
    res = run_scenario("2024-01-01", "2024-06-01")
    print(json.dumps(res, indent=2))
