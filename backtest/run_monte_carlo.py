import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime
import sys
from backtesting import Backtest
import random

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_engineering import prepare_data_for_ml
from strategies.ml_strategy import MLStrategy
from strategies.ml_strategy_trend import MLStrategyTrend

def run_monte_carlo(trades_df, initial_capital=10000, simulations=1000, horizon=100):
    """
    trades_df: Backtest sonucu dönen işlemler listesi (PnL yüzdeleri önemli).
    simulations: Kaç farklı senaryo üretilecek?
    horizon: Her senaryoda kaç işlem olacak?
    """
    if len(trades_df) < 10:
        print("Yetersiz işlem sayısı, simülasyon atlandı.")
        return None, [], []

    # İşlem Getirileri (Yüzdesel)
    returns = trades_df['ReturnPct'].values
    
    final_equities = []
    max_drawdowns = []
    
    # print(f"--- Monte Carlo ({simulations} Senaryo) ---") 
    
    for i in range(simulations):
        sim_returns = np.random.choice(returns, size=horizon, replace=True)
        equity_curve = [initial_capital]
        capital = initial_capital
        peak = initial_capital
        max_dd = 0
        
        for ret in sim_returns:
            capital = capital * (1 + ret)
            equity_curve.append(capital)
            if capital > peak: peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd: max_dd = dd
        
        final_equities.append(capital)
        max_drawdowns.append(max_dd)
        
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    results = {
        "mean_equity": float(np.mean(final_equities)),
        "median_equity": float(np.median(final_equities)),
        "p05_equity": float(np.percentile(final_equities, 5)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "risk_of_ruin_50pct": float(np.sum(max_drawdowns > 0.50) / simulations * 100),
        "risk_of_ruin_90pct": float(np.sum(max_drawdowns > 0.90) / simulations * 100)
    }
    
    return results, final_equities.tolist(), max_drawdowns.tolist()

def run_simulation_for_dashboard(strategy_name="Professional", initial_capital=10000, simulations=1000, horizon=150):
    """
    Dashboard üzerinden çağrılacak ana fonksiyon.
    1. Geçmiş veriyi çeker.
    2. Seçilen stratejiyle Backtest yapar.
    3. Çıkan işlemleri Monte Carlo ile simüle eder.
    """
    
    # 1. Model ve Veri Yükle
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        return {"error": "Model dosyası bulunamadı!"}

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Model yükleme hatası: {e}"}
    
    # Veri İndirme (Cache mekanizması olsa iyi olur ama şimdilik canlı çekelim)
    # Hızlı olsun diye son 3-4 yılı alalım
    try:
        df = yf.download('BTC-USD', start='2020-01-01', interval='1d', auto_adjust=True, progress=False)
        if df.empty:
             return {"error": "Veri indirilemedi."}
             
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if 'Close' not in df.columns: df.columns = [c.capitalize() for c in df.columns]
    except Exception as e:
        return {"error": f"Veri hatası: {e}"}

    # Feature Engineering
    try:
        df = prepare_data_for_ml(df)
        import pandas_ta as ta
        if 'SMA_200' not in df.columns:
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            
        # Tahmin
        features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date', 'ML_Signal', 'ML_Prob', 'ML_Prob', 'SMA_200']]
        
        # Model feature eşleştirme
        if hasattr(model, 'feature_names_in_'):
            X = df[features]
            missing_cols = set(model.feature_names_in_) - set(X.columns)
            if missing_cols:
                for c in missing_cols: X[c] = 0
            X = X[model.feature_names_in_]
            
            df['ML_Signal'] = model.predict(X)
            # Prob her zaman olmayabilir
            try: df['ML_Prob'] = model.predict_proba(X)[:, 1]
            except: df['ML_Prob'] = 0.5
            
            # --- METRİK HESAPLAMA (ÖNEMLİ: Gelecekteki harekete göre) ---
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # --- METRİK HESAPLAMA (Stratejiye Özel) ---
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Target: Ertesi gün Close > Bugün Close ise 1, değilse 0
            actual_target = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Geçerli veri
            valid_data = df.dropna().copy() # Copy önemli
            
            # STRATEJİ FİLTRESİ UYGULA
            # Eğer Profesyonel Strategy ise, Trend Tersi işlemleri "yok saymalıyız" veya "nötr" kabul etmeliyiz.
            # Ancak Accuracy hesaplarken "işlem yapılmayan" günleri nasıl sayacağız?
            # En mantıklısı: Sadece STRATEJİNİN İŞLEM YAPTIĞI (veya yapabileceği) günlerdeki başarısını ölçmek.
            
            if strategy_name == "Professional":
                 # Trend Filtresi: Fiyat > SMA_200 ise LONG (1) serbest, değilse Sinyal gelse bile işlem yok.
                 # Sadece LONG çalışan bir strateji olduğunu varsayarsak (kodlarımız öyle):
                 # Filtrenin geçmediği yerlerdeki sinyalleri değerlendirme dışı bırakalım.
                 
                 # Fiyat > SMA200 OLAN günleri filtrele
                 trend_filter = valid_data['Close'] > valid_data['SMA_200']
                 valid_data = valid_data[trend_filter]
                 
            # Filtre sonrası veri kaldı mı?
            if not valid_data.empty:
                valid_target = (valid_data['Close'].shift(-1) > valid_data['Close']).astype(int)
                
                valid_metrics_df = pd.DataFrame({'Signal': valid_data['ML_Signal'], 'Target': valid_target})
                valid_metrics_df = valid_metrics_df.iloc[:-1] # Son satırı at
                
                if not valid_metrics_df.empty:
                    acc = accuracy_score(valid_metrics_df['Target'], valid_metrics_df['Signal'])
                    prec = precision_score(valid_metrics_df['Target'], valid_metrics_df['Signal'], zero_division=0)
                    rec = recall_score(valid_metrics_df['Target'], valid_metrics_df['Signal'], zero_division=0)
                    f1 = f1_score(valid_metrics_df['Target'], valid_metrics_df['Signal'], zero_division=0)
                else:
                    acc, prec, rec, f1 = 0, 0, 0, 0
            else:
                 acc, prec, rec, f1 = 0, 0, 0, 0
            
    except Exception as e:
        return {"error": f"Feature Engineering hatası: {e}"}

    # 2. STRATEJİ SEÇİMİ
    strat_class = None
    strat_desc = ""
    
    if strategy_name == "Professional":
        strat_class = MLStrategyTrend
        strat_desc = "Profesyonel Model (Trend Filter): SMA 200 Filtresi VAR."
    elif strategy_name == "Adventurous":
        strat_class = MLStrategy
        strat_desc = "Maceracı Model: Filtresiz, agresif işlem."
    else:
        # Default
        strat_class = MLStrategyTrend
        strat_desc = "Varsayılan Strateji"

    # 3. BACKTEST
    try:
        bt = Backtest(df, strat_class, cash=1000000, commission=.001)
        stats = bt.run()
        trades = stats['_trades']
        
        if len(trades) < 10:
            return {"error": "Yetersiz işlem sayısı (Min 10). Monte Carlo yapılamaz."}
            
    except Exception as e:
        return {"error": f"Backtest hatası: {e}"}

    # 4. MONTE CARLO SİMÜLASYONU
    try:
        # Süre Tahmini
        total_days = (df.index[-1] - df.index[0]).days
        avg_days_per_trade = total_days / len(trades)
        sim_duration_days = avg_days_per_trade * horizon
        sim_duration_years = sim_duration_days / 365
        
        sim_stats, sim_equities, sim_dds = run_monte_carlo(trades, initial_capital=initial_capital, simulations=simulations, horizon=horizon)
        
        # ROI ve CAGR
        final_capital = sim_stats['mean_equity']
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
        # CAGR negatif kök hatası vermemesi için abs veya kontrol ekleyelim ama şimdilik basit tutalım
        if final_capital > 0:
            cagr = ((final_capital / initial_capital) ** (1 / max(sim_duration_years, 0.1)) - 1) * 100
        else:
            cagr = -100
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        # Sonuç Objemi
        result_data = {
            "model": strategy_name,
            "is_monte_carlo": True,
            "description": strat_desc,
            "timestamp": timestamp,
            "input_stats": {
                "total_trades": len(trades),
                "win_rate": stats['Win Rate [%]'],
                "avg_trade": float(trades['ReturnPct'].mean())
            },
            "model_metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            },
            "simulation_meta": {
                "initial_capital": initial_capital,
                "simulated_trades": horizon,
                "simulated_duration_years": round(sim_duration_years, 1),
                "mean_roi_pct": round(total_return_pct, 2),
                "cagr_pct": round(cagr, 2)
            },
            "simulation_results": sim_stats,
            "data_samples": {
                 "final_equities": sim_equities[:500], # Boyutu küçültmek için ilk 500
                 "max_drawdowns": sim_dds[:500]
            }
        }
        
        # Raporu Kaydet
        filename = f"reports/Monte_Carlo_{strategy_name}_{timestamp}.json"
        os.makedirs("reports", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(result_data, f, indent=4)
            
        # Dashboard için filename'i de ekle
        result_data['json_filename'] = os.path.basename(filename)
        
        return result_data

    except Exception as e:
        return {"error": f"Simülasyon hesaplama hatası: {e}"}

if __name__ == "__main__":
    # Test
    res = run_simulation_for_dashboard()
    if "error" in res:
        print(res["error"])
    else:
        print("Başarılı:", res['simulation_meta'])
