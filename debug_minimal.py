
import sys
import os
import pandas as pd
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_data_generator import generate_synthetic_data
from data.feature_engineering import prepare_data_for_ml

def test():
    print("Generating data...")
    df = generate_synthetic_data(duration_days=365)
    print("Columns:", df.columns.tolist())
    
    print("Running feature engineering...")
    try:
        processed_df = prepare_data_for_ml(df)
        print("Success!")
        print("Columns processed:", processed_df.columns.tolist())
        
        # PROCESSED_DF DEDUPE
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
        print("Columns after dedupe:", processed_df.columns.tolist())
        
        import joblib
        model_path = "models/saved_models/xgb_btc_v1.joblib"
        if os.path.exists(model_path):
            print("Loading model...")
            model = joblib.load(model_path)
            
            cols_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target', 'Date']
            features = [col for col in processed_df.columns if col not in cols_to_exclude]
            
            print("Features:", features)
            
            print("Predicting...")
            preds = model.predict(processed_df[features])
            print("Prediction shape:", preds.shape)
            
            print("Assigning to ML_Signal...")
            processed_df['ML_Signal'] = preds
            print("Assignment Success!")
            
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test()
