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

def main():
    # 1. Model ve Veri Yükle
    model_path = "models/saved_models/xgb_btc_v1.joblib"
    if not os.path.exists(model_path):
        print("Model bulunamadı!")
        return

    model = joblib.load(model_path)
    
    df = yf.download('BTC-USD', start='2020-01-01', interval='1d', auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    if 'Close' not in df.columns: df.columns = [c.capitalize() for c in df.columns]

    df = prepare_data_for_ml(df)
    
    import pandas_ta as ta
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = ta.sma(df['Close'], length=200)

    # Tahmin
    features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date', 'ML_Signal', 'ML_Prob', 'ML_Prob', 'SMA_200']]
    X = df[features]
    
    if hasattr(model, 'feature_names_in_'):
        missing_cols = set(model.feature_names_in_) - set(X.columns)
        if missing_cols:
            for c in missing_cols: X[c] = 0
        X = X[model.feature_names_in_]
    
    df['ML_Signal'] = model.predict(X)
    df['ML_Prob'] = model.predict_proba(X)[:, 1]
    
    # 2. STRATEJİLERİ TEST ET
    strategies = [
        {
            "id": "Adventurous",
            "name": "Maceracı Model (XGBoost V1)",
            "class": MLStrategy,
            "desc": "Trend Filtresi YOK. Boğa piyasasında Short açabilir. Yüksek Getiri / Yüksek Risk."
        },
        {
            "id": "Professional",
            "name": "Profesyonel Model (Trend Filter)",
            "class": MLStrategyTrend,
            "desc": "Trend Filtresi VAR (SMA 200). Boğa piyasasında Short yasak. Dengeli ve Güvenli."
        }
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    print("\n" + "="*60)
    print(f"MONTE CARLO SİMÜLASYONU BAŞLIYOR ({timestamp})")
    print("="*60)
    
    for strat in strategies:
        print(f"\n>>> Test Ediliyor: {strat['name']}...")
        
        # Backtest
        bt = Backtest(df, strat['class'], cash=1000000, commission=.001)
        stats = bt.run()
        trades = stats['_trades']
        
        print(f"   İşlem Sayısı: {len(trades)}")
        print(f"   Win Rate: %{stats['Win Rate [%]']:.2f}")
        
        # Simülasyon
        if len(trades) > 10:
            # SÜRE TAHMİNİ
            # Geçmiş verideki işlem sıklığına göre 150 işlem ne kadar sürer?
            total_days = (df.index[-1] - df.index[0]).days
            avg_days_per_trade = total_days / len(trades)
            sim_duration_days = avg_days_per_trade * 150 # Horizon=150
            sim_duration_years = sim_duration_days / 365
            
            sim_stats, sim_equities, sim_dds = run_monte_carlo(trades, initial_capital=10000, simulations=1000, horizon=150)
            
            # ROI ve CAGR Hesabı
            final_capital = sim_stats['mean_equity']
            initial_cap = 10000
            total_return_pct = ((final_capital - initial_cap) / initial_cap) * 100
            cagr = ((final_capital / initial_cap) ** (1 / sim_duration_years) - 1) * 100
            
            # Rapor Kaydet
            filename = f"reports/Monte_Carlo_{strat['id']}_{timestamp}.json"
            report = {
                "model": strat['name'], # Dashboard bunu gösterecek
                "is_monte_carlo": True,
                "description": strat['desc'],
                "timestamp": timestamp,
                "input_stats": {
                    "total_trades": len(trades),
                    "win_rate": stats['Win Rate [%]'],
                    "avg_trade": trades['ReturnPct'].mean()
                },
                "simulation_meta": {
                    "initial_capital": 10000,
                    "simulated_trades": 150,
                    "simulated_duration_years": round(sim_duration_years, 1),
                    "mean_roi_pct": round(total_return_pct, 2),
                    "cagr_pct": round(cagr, 2)
                },
                "simulation_results": sim_stats,
                "data_samples": {
                     "final_equities": sim_equities[:500],
                     "max_drawdowns": sim_dds[:500]
                }
            }
            
            with open(filename, "w") as f:
                json.dump(report, f, indent=4)
                
            print(f"   ✅ Rapor Kaydedildi: {filename}")
            print(f"   🛡️ Batış Riski: %{sim_stats['risk_of_ruin_50pct']:.2f}")
            print(f"   💰 Ort. Sermaye: ${sim_stats['mean_equity']:,.0f} (ROI: %{total_return_pct:.0f})")
            print(f"   ⏳ Tahmini Süre: {sim_duration_years:.1f} Yıl")
        else:
            print("   ❌ Yetersiz işlem verisi.")

if __name__ == "__main__":
    main()
