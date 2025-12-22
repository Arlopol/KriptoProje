import requests
import pandas as pd
import datetime
import time

class OnChainLoader:
    def __init__(self):
        self.base_url = "https://api.blockchain.info/charts"
        self.metrics = {
            "hash-rate": "Hash_Rate",
            "difficulty": "Difficulty",
            "n-transactions": "Transaction_Count",
            "n-unique-addresses": "Unique_Addresses",
            "avg-block-size": "Avg_Block_Size",
            "miners-revenue": "Miners_Revenue",
            "mempool-size": "Mempool_Size"
        }
    
    def fetch_metric(self, metric_endpoint, metric_name):
        """
        Blockchain.info API'den belirli bir metriği çeker.
        """
        url = f"{self.base_url}/{metric_endpoint}?timespan=5years&format=json&sampled=true"
        print(f"Veri çekiliyor: {metric_name}...", flush=True)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # 'values' listesi içinde x (timestamp) ve y (değer) var
            values = data.get('values', [])
            
            df = pd.DataFrame(values)
            df.columns = ['Timestamp', metric_name]
            
            # Timestamp'i Date'e çevir
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
            df.drop('Timestamp', axis=1, inplace=True)
            df.set_index('Date', inplace=True)
            
            # Aynı güne denk gelen birden fazla veri varsa ortalamasını al
            df = df.groupby(df.index).mean()
            
            return df
            
        except Exception as e:
            print(f"Hata ({metric_name}): {e}")
            return pd.DataFrame()

    def get_all_onchain_data(self):
        """
        Tüm metrikleri çeker ve tek bir DataFrame'de birleştirir.
        """
        main_df = pd.DataFrame()
        
        for endpoint, name in self.metrics.items():
            df_metric = self.fetch_metric(endpoint, name)
            
            if df_metric.empty:
                continue
                
            if main_df.empty:
                main_df = df_metric
            else:
                main_df = main_df.join(df_metric, how='outer')
            
            time.sleep(1) # API limitine takılmamak için bekleme
            
        # Boş değerleri doldur (Önceki değerle)
        main_df.fillna(method='ffill', inplace=True)
        main_df.dropna(inplace=True)
        
        print("\n✅ On-Chain Verileri Hazır.")
        print(main_df.tail())
        return main_df

if __name__ == "__main__":
    loader = OnChainLoader()
    df = loader.get_all_onchain_data()
    # Test için kaydet
    df.to_csv("data/onchain_data_test.csv")
    print(f"Test verisi kaydedildi: data/onchain_data_test.csv (Satır: {len(df)})")
