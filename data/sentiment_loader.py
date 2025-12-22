import requests
import pandas as pd
import time
import os

class SentimentLoader:
    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        
    def fetch_fear_and_greed_data(self, limit=0):
        """
        Alternative.me API'sinden Fear & Greed Index verisini çeker.
        limit=0 tüm geçmişi getirir.
        """
        try:
            print("😱 Fear & Greed verisi indiriliyor...", flush=True)
            params = {
                'limit': limit,
                'format': 'json'
            }
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print("Veri formatı hatalı!")
                return None
                
            records = data['data']
            
            # DataFrame'e çevir
            df = pd.DataFrame(records)
            
            # Tarih formatı (timestamp -> datetime)
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Değerleri sayıya çevir
            df['value'] = pd.to_numeric(df['value'])
            
            # Gereksiz sütunları at
            df = df[['Date', 'value', 'value_classification']]
            df.columns = ['Date', 'FNG_Value', 'FNG_Class']
            
            # Tarihe göre sırala
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Date sütununu datetime (ns) olarak ayarla (merge için önemli)
            df['Date'] = pd.to_datetime(df['Date'].dt.date) 
            
            print(f"✅ Toplam {len(df)} günlük Sentiment verisi çekildi.", flush=True)
            return df
            
        except Exception as e:
            print(f"Sentiment verisi çekilirken hata: {e}")
            return None

if __name__ == "__main__":
    loader = SentimentLoader()
    df = loader.fetch_fear_and_greed_data()
    if df is not None:
        print(df.tail())
        # Test amaçlı kaydet
        df.to_csv("data/sentiment_test.csv", index=False)
