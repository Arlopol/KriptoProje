import sys
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from backtesting import Backtest

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import prepare_data_for_ml
from strategies.ml_strategy_logging import MLStrategyLogging
from data.sentiment_loader import SentimentLoader
from data.onchain_loader import OnChainLoader

def run_walk_forward_analysis(
    symbol='BTC-USD', # Yeni parametre
    train_window_days=180, 
    test_window_days=30, 
    start_date='2023-01-01', 
    initial_capital=10000,
    use_trend_filter=True,
    use_sentiment=False,
    use_onchain=False,
    buy_threshold=0.60,
    sell_threshold=0.40,
    stop_loss_pct=0.10,
    take_profit_pct=0.20,
    use_trailing_stop=False,
    trailing_decay=0.10,
    use_dynamic_sizing=False
    ):
    """
    Rolling Window Walk-Forward Analysis.
    """
    
    # ... (Veri yükleme kısımları aynı) ...
    
    # ... (Önceki kodların devamı) ...

    loader = DataLoader()
    raw_df = loader.fetch_data(symbol, period='5y', interval='1d')
    
    if raw_df.empty:
        return {"error": "Veri indirilemedi."}

    # Tarih formatı düzeltme (Merge için kritik)
    if 'Date' not in raw_df.columns:
        raw_df = raw_df.reset_index()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.date
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])

    # --- 2. SENTIMENT ENTEGRASYONU ---
    if use_sentiment:
        print("🧠 Sentiment Verisi Ekleniyor...")
        sent_loader = SentimentLoader()
        df_sent = sent_loader.fetch_fear_and_greed_data(limit=0)
        
        if df_sent is not None and not df_sent.empty:
            df_sent['Date'] = pd.to_datetime(df_sent['Date'])
            raw_df = pd.merge(raw_df, df_sent, on='Date', how='inner')
            print(f"Sentiment Eklendi. Veri Sayısı: {len(raw_df)}")
            
    # --- 3. ON-CHAIN ENTEGRASYONU ---
    if use_onchain:
        print("🔗 On-Chain Verisi Ekleniyor...")
        oc_loader = OnChainLoader()
        df_oc = oc_loader.get_all_onchain_data()
        
        if df_oc is not None and not df_oc.empty:
            # OnChain datası index olarak Date tutuyor, onu kolona çekelim
            df_oc = df_oc.reset_index()
            df_oc['Date'] = pd.to_datetime(df_oc['Date'])
            
            # Merge
            raw_df = pd.merge(raw_df, df_oc, on='Date', how='inner')
            print(f"On-Chain Eklendi. Veri Sayısı: {len(raw_df)}")
            
    raw_df.set_index('Date', inplace=True)
        
    print("Feature Engineering yapılıyor...")
    df = prepare_data_for_ml(raw_df)

    # Ekstra Feature Engineering (Sentiment & OnChain)
    if use_sentiment and 'FNG_Value' in df.columns:
        df['FNG_Change'] = df['FNG_Value'].pct_change()
        df['FNG_MA_7'] = df['FNG_Value'].rolling(window=7).mean()
        
    if use_onchain and 'Hash_Rate' in df.columns:
        # Onchain verilerinin değişim oranlarını (Momentum) ekle
        for col in ['Hash_Rate', 'Miners_Revenue', 'Transaction_Count', 'Unique_Addresses']:
            if col in df.columns:
                df[f'{col}_Change'] = df[col].pct_change()
    
    df.dropna(inplace=True)
    
    # Feature listesi (Target ve Date hariç her şey)
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date', 'ML_Signal', 'ML_Prob', 'SMA_200', 'FNG_Class']]
    
    # Başlangıç Tarihi İndeksi
    try:
        start_idx = df.index.get_loc(pd.to_datetime(start_date))
    except KeyError:
        # Tam tarih yoksa en yakınını bulmak gerekebilir ama şimdilik basit tutalım
        # veya data'nın başladığı yer + train window kadar ileri gidelim
        start_idx = max(train_window_days, 0)
        
    if start_idx < train_window_days:
        start_idx = train_window_days 
        
    # Walk-Forward Döngüsü
    print(f"Walk-Forward Başlıyor: Start={start_date}, Train={train_window_days} gün, Step={test_window_days} gün")
    
    results = []
    current_idx = start_idx
    total_len = len(df)
    
    model_stats = []
    
    while current_idx < total_len:
        # TEST Penceresinin sonu
        test_end_idx = min(current_idx + test_window_days, total_len)
        
        # Eğer test edilecek veri kalmadıysa çık
        if current_idx >= test_end_idx:
            break
            
        # TRAIN Penceresi (Test başlangıcından geriye doğru)
        train_start_idx = current_idx - train_window_days
        
        # Dilimle
        train_data = df.iloc[train_start_idx:current_idx]
        test_data = df.iloc[current_idx:test_end_idx].copy() # Copy önemli, üzerine yazacağız
        
        if len(train_data) < 50: # Yetersiz veri
            current_idx += test_window_days
            continue
            
        # Modeli Eğit (Hyperparametreler train_xgboost.py'dan alındı)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        X_train = train_data[features]
        y_train = train_data['Target']
        
        # Eğitim
        # print(f"  Eğitiliyor: {train_data.index[0].date()} -> {train_data.index[-1].date()}")
        model.fit(X_train, y_train)
        
        # Test (Prediction) - OUT OF SAMPLE
        X_test = test_data[features]
        
        # Tahminleri Test Verisine Ekle
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        test_data['ML_Signal'] = preds
        test_data['ML_Prob'] = proba
        
        # Modelin o dönemki test başarısı (Accuracy)
        from sklearn.metrics import accuracy_score
        # Target'ı kontrol et (Actual)
        actual_target = (test_data['Close'].shift(-1) > test_data['Close']).astype(int)
        # Shift yüzünden son eleman nan olur, accuracy hesabında düşelim
        mask = actual_target.notna()
        if mask.sum() > 0:
            acc = accuracy_score(actual_target[mask], preds[mask])
            model_stats.append({
                "period": str(test_data.index[0].date()),
                "accuracy": acc
            })
        
        # Sonuçları listeye ekle
        results.append(test_data)
        
        # İlerle
        current_idx += test_window_days
        
    # Tüm parçaları birleştir (Concatenate via Time)
    if not results:
        return {"error": "Yeterli veri aralığı bulunamadı veya işlem yapılamadı."}
        
    final_df = pd.concat(results)
    
    # Tekrarlanan index var mı kontrol (Bazen bindirme olabilir, ama index bazlı yaptık)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    # 2. Backtest Çalıştır (Birleştirilmiş Veri Üzerinde)
    # Scale Factor (mBTC workaround)
    SCALE_FACTOR = 1000.0
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'SMA_200']
    
    # Backtest için kopya
    bt_data = final_df.copy()
    for c in cols_to_scale:
        if c in bt_data.columns:
            bt_data[c] = bt_data[c] / SCALE_FACTOR
            
    # Strateji Seçimi
    # Thresholdları dinamik olarak ayarlamak için sınıfı özelleştiriyoruz
    class OptimizedStrategy(MLStrategyLogging):
        pass
        
    # Class variable update
    OptimizedStrategy.buy_threshold = buy_threshold
    OptimizedStrategy.sell_threshold = sell_threshold
    OptimizedStrategy.stop_loss_pct = stop_loss_pct
    OptimizedStrategy.take_profit_pct = take_profit_pct
    OptimizedStrategy.use_trailing_stop = use_trailing_stop
    OptimizedStrategy.trailing_decay = trailing_decay
    OptimizedStrategy.use_trend_filter = use_trend_filter
    OptimizedStrategy.use_dynamic_sizing = use_dynamic_sizing

    bt = Backtest(bt_data, OptimizedStrategy, cash=initial_capital, commission=0.001, trade_on_close=True)
    stats = bt.run()
    
    # 3. Sonuçları Paketle
    strategy_instance = stats['_strategy']
    logs = strategy_instance.decision_logs
    
    # Equity Curve'ü çıkar
    equity_curve = stats['_equity_curve'] # Bu bir DF
    # Indexi string yapalım JSON için
    equity_dict = {str(k.date()): v for k, v in equity_curve['Equity'].to_dict().items()}
    
    # Buy & Hold Getirisi Hesapla
    first_price = bt_data.Close.iloc[0]
    last_price = bt_data.Close.iloc[-1]
    bh_return = ((last_price - first_price) / first_price) * 100
    
    # Buy & Hold Equity Curve
    bh_series = (bt_data['Close'] / first_price) * initial_capital
    bh_dict = {str(k.date()): v for k, v in bh_series.to_dict().items()}

    # Fiyat Verisi (Grafik için)
    price_dict = {str(k.date()): v for k, v in bt_data['Close'].to_dict().items()}

    res = {
        "symbol": symbol,
        "start_date": str(final_df.index[0].date()),
        "end_date": str(final_df.index[-1].date()),
        "final_equity": stats['Equity Final [$]'],
        "return_pct": stats['Return [%]'],
        "bh_return": bh_return, # Benchmark
        "sharpe_ratio": stats['Sharpe Ratio'],
        "max_drawdown": stats['Max. Drawdown [%]'],
        "win_rate": stats['Win Rate [%]'],
        "total_trades": stats['# Trades'],
        "logs": logs,
        "model_accuracy_history": model_stats,
        "bh_equity_curve": bh_dict,
        "equity_curve": equity_dict,
        "price_data": price_dict
    }
    
    return res

def save_walk_forward_report(res, filename="reports/walk_forward_last_run.json"):
    """
    Sonuç sözlüğünü JSON olarak kaydeder.
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Numpy tiplerini temizle (JSON serialize hatası olmasın)
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        if isinstance(o, np.float32): return float(o)
        return str(o)
        
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(res, f, default=convert, indent=4)
        print(f"✅ Rapor kaydedildi: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Rapor kaydedilemedi: {e}")
        return None

# Fonksiyon sarmalayıcı (Dashboard'dan çağrılan)
def run_walk_forward_and_save(**kwargs):
    res = run_walk_forward_analysis(**kwargs)
    if "error" not in res:
        save_walk_forward_report(res)
    return res

if __name__ == "__main__":
    # Test Modu
    out = run_walk_forward_analysis(train_window_days=180, test_window_days=30, start_date='2024-01-01')
    if "error" in out:
        print(out["error"])
    else:
        print(f"Final Equity: {out['final_equity']}")
        print(f"Accuracy History: {out['model_accuracy_history']}")
