import yfinance as yf
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_data(self, symbol='BTC-USD', period='2y', interval='1d'):
        """
        Yahoo Finance'ten veri çeker ve CSV olarak kaydeder.
        
        Args:
            symbol (str): Ticker sembolü (örn. BTC-USD, ETH-USD)
            period (str): Veri periyodu (örn. 1mo, 1y, 2y, max)
            interval (str): Veri aralığı (örn. 1d, 1h, 15m)
        
        Returns:
            pd.DataFrame: OHLCV verisi
        """
        print(f"Veri indiriliyor: {symbol} (Periyot: {period}, Aralık: {interval})...")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if df.empty:
                print(f"UYARI: {symbol} için veri bulunamadı.")
                return None
            
            # Sütunları düzeltme (Yahoo Finance bazen MultiIndex döndürür)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # CSV'ye kaydet
            filename = f"{symbol}_{period}_{interval}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            print(f"Veri kaydedildi: {filepath}")
            print(f"Toplam Satır: {len(df)}")
            return df
            
        except Exception as e:
            print(f"HATA: Veri indirilirken bir sorun oluştu: {e}")
            return None

    def load_local_data(self, filename):
        """Yerel CSV dosyasından veriyi yükler."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Dosya bulunamadı: {filepath}")
            return None
            
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

if __name__ == "__main__":
    # Test
    loader = DataLoader()
    df = loader.fetch_data(symbol='BTC-USD', period='1y', interval='1d')
    print(df.head())
