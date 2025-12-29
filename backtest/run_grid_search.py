
import pandas as pd
import itertools
from backtest.run_walk_forward import run_walk_forward_analysis
import time

def run_grid_search(
    param_grid,
    train_window=365,
    test_window=90,
    use_sentiment=True,
    use_onchain=True,
    progress_callback=None
):
    """
    Verilen parametre kombinasyonlarını dener ve sonuçları raporlar.
    """
    
    # Grid Kombinasyonlarını Oluştur
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Toplam {len(combinations)} kombinasyon test edilecek...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        # Parametreleri sözlüğe çevir (Grid'den gelenler)
        params = dict(zip(keys, combo))
        
        # Grid içinde yoksa varsayılan argümanları kullan, varsa grid'dekini kullan
        current_sentiment = params.get('use_sentiment', use_sentiment)
        current_onchain = params.get('use_onchain', use_onchain)
        
        # İlerleme Bildirimi
        msg = f"[{i+1}/{len(combinations)}] Test Ediliyor... (Sent={current_sentiment}, Grid={params})"
        if progress_callback:
            progress_callback(i + 1, len(combinations), msg)
        else:
            print(msg)
        
        try:
            start_time = time.time()
            res = run_walk_forward_analysis(
                train_window_days=train_window,
                test_window_days=test_window,
                start_date='2023-01-01', # Sabit
                use_trend_filter=True,    # Profesyonel Mod Sabit
                use_sentiment=current_sentiment,
                use_onchain=current_onchain,
                buy_threshold=params['buy_threshold'],
                sell_threshold=params['sell_threshold'],
                stop_loss_pct=params['stop_loss_pct'],
                take_profit_pct=params['take_profit_pct'],
                use_trailing_stop=params.get('use_trailing_stop', False),
                trailing_decay=params.get('trailing_decay', 0.10)
            )
            duration = time.time() - start_time
            
            if "error" in res:
                print(f"  HATA: {res['error']}")
                continue
                
            # Sonuçları Kaydet
            results.append({
                "Sentiment": current_sentiment,
                "OnChain": current_onchain,
                "Buy_Thresh": params['buy_threshold'],
                "Sell_Thresh": params['sell_threshold'],
                "Stop_Loss": params['stop_loss_pct'],
                "Take_Profit": params['take_profit_pct'],
                "Trailing": params.get('use_trailing_stop', False),
                "Trail_Decay": params.get('trailing_decay', "-"),
                "Return_Pct": res['return_pct'],
                "BH_Return": res.get('bh_return', 0.0), # Benchmark
                "Max_DD": res['max_drawdown'],
                "Sharpe": res['sharpe_ratio'],
                "Trades": res['total_trades'],
                "Final_Eq": res['final_equity'],
                "Duration_Sec": round(duration, 1)
            })
            
        except Exception as e:
            print(f"  BEKLENMEDİK HATA: {e}")

    # Sonuçları DataFrame yap
    df_res = pd.DataFrame(results)
    
    # Sıralama (Return'e göre azalan)
    if not df_res.empty:
        df_res = df_res.sort_values(by="Return_Pct", ascending=False)
        
        # CSV Olarak Otomatik Kaydet
        import os
        from datetime import datetime
        os.makedirs("reports/grid_search", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/grid_search/grid_results_{timestamp}.csv"
        df_res.to_csv(filename, index=False)
        print(f"✅ Grid Sonuçları Kaydedildi: {filename}")
        
    return df_res

if __name__ == "__main__":
    # Test Amaçlı Basit Grid
    test_grid = {
        'buy_threshold': [0.60, 0.70],
        'sell_threshold': [0.40],
        'stop_loss_pct': [0.05, 0.10],
        'take_profit_pct': [0.15, 0.30]
    }
    
    df = run_grid_search(test_grid, test_window=180) # Hızlı olsun diye test_window büyük
    print("\n--- SONUÇLAR ---\n")
    print(df.to_string())
