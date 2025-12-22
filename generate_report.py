import json
import os
import random
import datetime

def generate_dummy_report():
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename_base = f"SmaCross_{timestamp}"
    json_path = os.path.join(report_dir, f"{filename_base}.json")
    
    # Generate 100 days of dummy data
    dates = [(datetime.datetime.now() - datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(100, 0, -1)]
    
    initial_capital = 1000000
    equity = [initial_capital]
    buy_hold = [initial_capital]
    
    for _ in range(99):
        # Random daily return between -2% and +2.5% for strategy
        ret = random.uniform(-0.02, 0.025)
        equity.append(equity[-1] * (1 + ret))
        
        # Random daily return for buy & hold (more volatile)
        bh_ret = random.uniform(-0.03, 0.035)
        buy_hold.append(buy_hold[-1] * (1 + bh_ret))
        
    final_equity = equity[-1]
    final_buy_hold = buy_hold[-1]
    
    strat_return = ((final_equity / initial_capital) - 1) * 100
    bh_return = ((final_buy_hold / initial_capital) - 1) * 100
    
    summary_json = {
        "strategy": "SmaCross", 
        "description": "SMA Kesişimi (Kısa: 10, Uzun: 20). Simüle Edilmiş Veri.",
        "symbol": "BTC-USD",
        "date": timestamp,
        "initial_capital": initial_capital,
        "metrics": {
            "return": strat_return,
            "buy_hold_return": bh_return,
            "win_rate": 45.0,
            "max_drawdown": -15.5,
            "sharpe": 1.2,
            "trades": 42,
            "final_equity": final_equity
        },
        "equity_curve": {
            "dates": dates,
            "equity": equity,
            "drawdown": [0] * 100, # Dummy
            "buy_hold": buy_hold
        },
        "files": {
            "html": ""
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4)
        
    print(f"Generated report: {json_path}")

if __name__ == "__main__":
    generate_dummy_report()
