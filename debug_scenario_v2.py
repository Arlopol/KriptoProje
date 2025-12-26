
import pandas as pd
from backtesting import Backtest
from strategies.ml_strategy_logging import MLStrategyLogging
import joblib
from data.feature_engineering import add_technical_indicators

def debug_scenario():
    print("--- DEBUG START ---")
    
    # 1. Veri Yükle
    print("Loading data...")
    df = pd.read_csv('data/dashboard_sentiment.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Rename for Backtesting
    df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    }, inplace=True)
    
    # Drop existing FNG columns to avoid issues with indicators
    cols_to_drop = ['FNG_Class', 'FNG_Value_Text']
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    
    print(f"Data Loaded. Shape: {df.shape}")
    print(df.head())
    
    # 2. Teknik İndikatörler
    print("Adding indicators...")
    df = add_technical_indicators(df)
    
    # 3. Model
    print("Loading model...")
    model = joblib.load('models/saved_models/xgb_btc_v1.joblib')
    features = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume','Date','Target','Signal','year','month','day', 'Fear & Greed Value', 'Fear & Greed Values', 'Fear & Greed Index', 'FNG_Value']]
    
    print(f"Features: {features[:5]}...")
    X = df[features]
    
    print("Predicting...")
    prob = model.predict_proba(X)[:, 1]
    df['ML_Prob'] = prob
    # Dummy signal
    df['ML_Signal'] = (prob > 0.5).astype(int) 
    
    # 4. Slice Data (Last 30 days)
    end_date = df.index[-1]
    start_date = end_date - pd.Timedelta(days=60)
    print(f"Slicing from {start_date} to {end_date}")
    
    scenario_data = df.loc[start_date:end_date].copy()
    
    # Drop non-numeric columns for Backtest
    # Daha güvenli yöntem: Sadece sayısal olanları tut
    import numpy as np
    scenario_data = scenario_data.select_dtypes(include=[np.number])
            
    print(f"Scenario Data Shape: {scenario_data.shape}")
    print("Columns:", scenario_data.columns.tolist())
    
    if scenario_data.empty:
        print("ERROR: Scenario Data is EMPTY!")
        return

    # 5. Run Backtest
    print("Running Backtest...")
    # Use $100 capital
    bt = Backtest(scenario_data, MLStrategyLogging, cash=100, commission=0.001)
    stats = bt.run()
    
    print("--- RESULTS ---")
    print(stats)
    print(f"Trades Count: {stats['# Trades']}")
    
    # Inspect LOGS
    strategy = stats['_strategy']
    logs = strategy.decision_logs
    print(f"Logs Count: {len(logs)}")
    if logs:
        print("First 3 Logs:")
        for l in logs[:3]:
            print(l)

if __name__ == "__main__":
    debug_scenario()
