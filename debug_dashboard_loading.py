
import pandas as pd
import json
import os

report_path = "c:/Users/katar/KriptoProje/reports/Monte_Carlo_Adventurous_20251221_1701.json"

try:
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Simulate loading as single object normalize
    df = pd.json_normalize(data)
    
    print("Columns found in dataframe:")
    for col in df.columns:
        if "simulation_meta" in col:
            print(f" - {col}: {df[col].iloc[0]}")
            
    print("\n--- Direct Key Access check ---")
    print(f"simulation_meta.simulated_duration_years: {df.get('simulation_meta.simulated_duration_years', 'NOT FOUND').iloc[0]}")
    
except Exception as e:
    print(f"Error: {e}")
