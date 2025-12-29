
import sys
import os
import pandas as pd
import joblib
from backtesting import Backtest
import json
import numpy as np

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy_logging import MLStrategyLogging
from data.feature_engineering import prepare_data_for_ml
from data.synthetic_data_generator import generate_synthetic_data

def run_synthetic_test(duration_days=365, volatility=0.03, drift=0.0002, initial_capital=10000, use_regime_switching=True):
    """
    Rastgele (Sentetik) bir piyasa verisi üretir ve modelin bu verideki performansını test eder.
    """
    if duration_days < 250:
        return {"error": "HATA: Simülasyon süresi en az 250 gün olmalıdır. Modelin 'SMA 200' (200 Günlük Ortalama) indikatörünü hesaplayabilmesi için bu veri gereklidir."}
    
    # 1. Sentetik Veri Üret
    # Kullanıcıya "Kaotik", "Boğa", "Ayı" seçimi yaptırabiliriz, ona göre drift/volatilite değişir.
    # Şimdilik parametre olarak alıyoruz.
    df = generate_synthetic_data(duration_days=duration_days, 
                                 start_price=60000, 
                                 volatility=volatility, 
                                 drift=drift,
                                 use_regime_switching=True)
    
    # 2. Özellik Mühendisliği (Feature Engineering)
    # Modelin çalışması için gereken RSI, SMA, Volatilite vb. bu yeni veriye göre sıfırdan hesaplanmalı.
    # UYARI: prepare_data_for_ml içinde 'Target' oluşturulurken shift(-1) kullanılıyor.
    # Bu data leakage değildir, çünkü backtest sırasında .next() geçmiş veriyi kullanır.
    
    # Ancak F&G gibi dış veri kaynaklarına bağımlılık varsa patlar.
    # Bizim prepare_data_for_ml fonksiyonu sadece teknik indikatörler kullanıyor olması lazım.
    # Kontrol edelim: "Fear & Greed" verisi synthetic datada yok.
    # Eğer model F&G isterse hata verir.
    # Modeli F&G'siz eğitmiştik (V1), değil mi? Evet.
    
    # features listesinde F&G yoksa sorun yok.
    
    processed_df = prepare_data_for_ml(df)
    
    # 3. Modeli Yükle ve Tahmin Et
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        return {"error": "Model dosyası bulunamadı."}
        
    model = joblib.load(model_path)
    
    # Feature Sütunlarını Seç (Eğitimdeki sırayla aynı olmalı)
    # Eğitim kodundan kopyalayalım veya modelin feature_names_in_ attribute'una bakalım (varsa)
    
    # Standart feature listesi:
    # ['RSI', 'ROC', 'MACD...', 'SMA...', 'Dist...', 'ATR', 'Bbands...', 'Log_Return', 'Lags...', 'Vols...']
    # prepare_data_for_ml hepsini üretiyor.
    # Sadece OHLC ve Date sütunlarını çıkaracağız.
    
    cols_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']
    features = [col for col in processed_df.columns if col not in cols_to_exclude]
    
    # XGBoost sessizce eksik sütunları handle edebilir veya hata verir.
    
    # Duplicate columns check before assignment
    processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

    try:
        processed_df['ML_Signal'] = model.predict(processed_df[features])
        processed_df['ML_Prob'] = model.predict_proba(processed_df[features])[:, 1]
    except ValueError as e:
        return {"error": f"Model feature uyuşmazlığı: {str(e)}. Sentetik veride F&G veya Onchain data yok."}
    
    
    # 4. Backtest (mBTC Workaround ile)
    scenario_data = processed_df.copy()
    SCALE_FACTOR = 1000.0
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'SMA_200', 'SMA_50', 'BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0']

    # 4. Backtest Prep (Safe Construction)
    final_data = {}
    unique_cols = set()
    
    # Iterate over all columns to preserve them (Features + OHLC)
    for col in scenario_data.columns:
        if col in unique_cols: continue
        unique_cols.add(col)
        
        val = scenario_data[col]
        if isinstance(val, pd.DataFrame): val = val.iloc[:, 0]
        
        if col in cols_to_scale:
            final_data[col] = val.values / SCALE_FACTOR
        else:
            final_data[col] = val.values
            
    final_df = pd.DataFrame(final_data, index=scenario_data.index)
            
    bt = Backtest(final_df, MLStrategyLogging, cash=initial_capital, commission=0.001, trade_on_close=True)
    stats = bt.run()
    
    # 5. Sonuçlar
    strategy_instance = stats['_strategy']
    logs = strategy_instance.decision_logs
    
    result = {
        "duration_days": duration_days,
        "initial_capital": initial_capital,
        "final_equity": stats['Equity Final [$]'],
        "return_pct": stats['Return [%]'],
        "max_drawdown": stats['Max. Drawdown [%]'],
        "win_rate": stats['Win Rate [%]'],
        "total_trades": stats['# Trades'],
        "logs": logs
    }
    
    return result

if __name__ == "__main__":
    res = run_synthetic_test(duration_days=365, volatility=0.05, drift=-0.001) # Ayı piyasası simülasyonu
    print(f"Final Equity: {res.get('final_equity')}")
    print(f"Trades: {res.get('total_trades')}")
