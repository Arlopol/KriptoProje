
import sys
import os
import pandas as pd

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.sentiment_loader import SentimentLoader

def prepare_dashboard_data():
    print("--- Dashboard Verisi Hazırlanıyor ---")
    
    # 1. Fiyat Verisi (Son 2-3 yıl yeterli, sentiment verisi zaten sınırlı)
    loader = DataLoader()
    # Sentiment verisi genelde 2018'den başlıyor. 5 yıllık alalım.
    price_df = loader.fetch_data('BTC-USD', period='5y', interval='1d')
    
    if price_df is None or price_df.empty:
        print("❌ Fiyat verisi alınamadı.")
        return

    # Date sütununu ayarla (Index'ten çıkar)
    price_df = price_df.reset_index()
    price_df['Date'] = pd.to_datetime(price_df['Date'].dt.date)
    
    # 2. Sentiment Verisi
    sent_loader = SentimentLoader()
    sent_df = sent_loader.fetch_fear_and_greed_data(limit=0)
    
    if sent_df is None or sent_df.empty:
        print("❌ Sentiment verisi alınamadı.")
        return
        
    # Date zaten datetime.date formatında gelmeli (loader'da ayarlamıştık)
    # Kontrol edelim
    sent_df['Date'] = pd.to_datetime(sent_df['Date'])
    
    # 3. Birleştirme (Merge)
    # Left join: Fiyat verisi ana eksen olsun.
    merged_df = pd.merge(price_df, sent_df, on='Date', how='left')
    
    # Eksik sentiment verilerini doldur (Öncekini kullan - ffill)
    merged_df['FNG_Value'] = merged_df['FNG_Value'].fillna(method='ffill')
    merged_df['FNG_Class'] = merged_df['FNG_Class'].fillna(method='ffill')
    
    # 4. Kaydet
    output_path = "data/dashboard_sentiment.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Dashboard verisi hazırlandı: {output_path}")
    print(merged_df[['Date', 'Close', 'FNG_Value']].tail())

if __name__ == "__main__":
    prepare_dashboard_data()
