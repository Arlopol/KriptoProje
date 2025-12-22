import sys
import os
import pandas as pd
import numpy as np
from backtesting import Backtest
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ml_strategy import MLStrategy
from data.data_loader import DataLoader

def run_strategy_backtest():
    # 1. Veriyi Yükle
    loader = DataLoader()
    symbol = 'BTC-USD'
    interval = '1h'
    period = '730d' 
    filename = f"{symbol}_{period}_{interval}.csv"
    
    if not os.path.exists(os.path.join(loader.data_dir, filename)):
        print(f"{interval} verisi indiriliyor...")
        df = loader.fetch_data(symbol=symbol, period=period, interval=interval)
    else:
        print(f"Yerel veri yükleniyor: {filename}")
        df = loader.load_local_data(filename)
    
    if df is None:
        print("Veriye ulaşılamadı.")
        return

    # Veri temizliği
    df = df.astype(float)
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    print(f"Toplam Veri: {len(df)} bar")

    # ---------------------------------------------------------
    # 2. Feature Engineering (Gelişmiş)
    # ---------------------------------------------------------
    print("Özellikler türetiliyor (SMA 200, OBV, RSI)...")
    
    # Trend Göstergeleri
    df['SMA_50'] = ta.sma(pd.Series(df['Close']), length=50)
    df['SMA_200'] = ta.sma(pd.Series(df['Close']), length=200)
    
    # Fiyatın SMA'ya uzaklığı (Trend Gücü)
    df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # Momentum
    df['RSI'] = ta.rsi(pd.Series(df['Close']), length=14)
    
    # Hacim Göstergeleri
    df['OBV'] = ta.obv(pd.Series(df['Close']), pd.Series(df['Volume']))
    df['OBV_Slope'] = df['OBV'].diff(5) # 5 barlık OBV değişimi
    
    # Volatilite (ATR)
    df['ATR'] = ta.atr(pd.Series(df['High']), pd.Series(df['Low']), pd.Series(df['Close']), length=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # Lagged Returns
    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_6'] = df['Close'].pct_change(6) # 6 saatlik değişim
    
    # TARGET: Gelecek bar yükseliş mi? (Basit Yön Tahmini)
    # Threshold'u kaldırdık çünkü model öğrenemiyor olabilir.
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # NaN değerleri temizle
    df = df.dropna()

    # ---------------------------------------------------------
    # 3. Model Eğitimi (Train/Test Split)
    # ---------------------------------------------------------
    split_idx = int(len(df) * 0.70)
    
    features = ['RSI', 'Dist_SMA200', 'OBV_Slope', 'ATR_Pct', 'Ret_1', 'Ret_6']
    X = df[features]
    y = df['Target']
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Eğitim Seti: {len(X_train)}, Test Seti: {len(X_test)}")
    print(f"Target Dağılımı (Train): {y_train.value_counts(normalize=True).to_dict()}")

    print("Model eğitiliyor (Random Forest - Relaxed)...")
    # n_estimators=100 yeterli, derinliği biraz sınırlayalım ki ezberlemesin (max_depth=5-10)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds_test = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1] # Olasılık
    
    acc = accuracy_score(y_test, preds_test)
    precision = precision_score(y_test, preds_test)
    print(f"Model Doğruluğu: %{acc * 100:.2f}")
    print(f"Precision (İsabet): %{precision * 100:.2f}")
    
    # Sinyalleri Oluştur: Standart %50 üzeri ise girsin.
    # Güven eşiğini kaldırdık (> 0.5)
    custom_signal = np.where(preds_proba > 0.50, 1, 0)
    
    # Test DataFrame'ine sinyali ekle
    df_test = df.iloc[split_idx:].copy()
    df_test['Signal'] = custom_signal
    
    # Feature Importance (Meraklısı için)
    importances = model.feature_importances_
    feat_imp_dict = dict(zip(features, importances))
    print("Önem Dereceleri:", feat_imp_dict)

    # ---------------------------------------------------------
    # 4. Backtest (Validation)
    # ---------------------------------------------------------
    if df_test.empty:
        print("Test verisi boş!")
        return

    try:
        bt = Backtest(df_test, MLStrategy, cash=1000000, commission=.002)

        print(f"--- Optimized ML Backtest: {symbol} ---")
        stats = bt.run()
        
        print("\n--- SONUÇLAR ---")
        print(stats)
        
        # Raporlama
        print("\nGrafikler oluşturuluyor...")
        import datetime
        report_dir = "reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename_base = f"ML_Optimized_{timestamp}"
        
        html_path = os.path.join(report_dir, f"{filename_base}.html")
        bt.plot(filename=html_path, open_browser=False)
        
        # JSON Çıktısı
        strat_return = stats['Return [%]']
        buy_hold_return = stats['Buy & Hold Return [%]']
        diff = strat_return - buy_hold_return
        
        import json
        json_path = os.path.join(report_dir, f"{filename_base}.json")
        
        def convert_numpy(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif hasattr(obj, 'item'): 
                return obj.item()
            else:
                return str(obj)

        equity_data = stats['_equity_curve'].reset_index()
        aligned_close = df_test['Close'].reindex(stats['_equity_curve'].index)
        initial_price = aligned_close.iloc[0]
        if pd.isna(initial_price):
            initial_price = aligned_close.dropna().iloc[0]
            
        buy_hold_equity = (aligned_close / initial_price) * 1000000
        equity_dates = [str(d) for d in stats['_equity_curve'].index]

        summary_json = {
            "strategy": "ML_Relaxed", 
            "description": f"ML Relaxed (RF, Trend+Vol). Acc: %{acc*100:.1f}. Threshold: >50% (Standard).",
            "symbol": symbol,
            "period": interval,
            "date": timestamp,
            "initial_capital": 1000000,
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
                "equity": [float(x) for x in stats['_equity_curve']['Equity'].values],
                "drawdown": [float(x) for x in stats['_equity_curve']['DrawdownPct'].values],
                "buy_hold": [float(x) if not pd.isna(x) else 0 for x in buy_hold_equity.values]
            },
            "files": {
                "html": os.path.basename(html_path)
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=4, default=convert_numpy)

        print("\n" + "="*40)
        print(f"📊 SONUÇ ÖZETİ ({symbol} - ML Optimized)")
        print("="*40)
        print(f"🔹 Model İsabeti:     %{precision*100:.2f}")
        print(f"🔹 Strateji Getirisi:   %{strat_return:.2f}")
        print(f"🔸 Al-ve-Tut Getirisi:  %{buy_hold_return:.2f}")
        print(f"⚠️  Fark (Başarım):      %{diff:.2f}")
        print("-" * 40)
        print(f"📅 Rapor Dosyası:       {html_path}")
        print(f"📊 Veri Dosyası:        {json_path}")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\nBACKTEST HATASI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_strategy_backtest()
