
import numpy as np
import pandas as pd

def generate_synthetic_data(duration_days=365, start_price=60000, volatility=0.02, drift=0.0005, random_seed=None):
    """
    Geometric Brownian Motion (GBM) kullanarak yapay bir Bitcoin grafiği (OHLC) üretir.
    
    Argümanlar:
        duration_days (int): Kaç günlük veri üretilecek.
        start_price (float): Başlangıç fiyatı.
        volatility (float): Günlük volatilite (Standart Sapma). Örn: 0.02 = %2.
        drift (float): Günlük beklenen getiri (Trend). Pozitifse yukarı, negatifse aşağı eğilim.
        random_seed (int): Tekrarlanabilirlik için tohum değeri. None ise rastgele.
        
    Döndürür:
        pd.DataFrame: Open, High, Low, Close, Volume, Date sütunları.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Tarih İndeksi
    start_date = pd.to_datetime('2025-01-01') # Gelecek bir tarih veriyoruz
    dates = pd.date_range(start=start_date, periods=duration_days, freq='D')
    
    # Kapanış Fiyatlarını Hesapla (GBM)
    dt = 1 # Zaman adımı (1 gün)
    
    # Rastgele şoklar (Wiener Process)
    shock = np.random.normal(0, 1, duration_days)
    
    # Fiyat Yolu
    prices = [start_price]
    for i in range(1, duration_days):
        # S_t = S_{t-1} * exp( (mu - 0.5 * sigma^2)*dt + sigma * shock * sqrt(dt) )
        drift_comp = (drift - 0.5 * volatility**2) * dt
        shock_comp = volatility * shock[i] * np.sqrt(dt)
        change = np.exp(drift_comp + shock_comp)
        
        new_price = prices[-1] * change
        prices.append(new_price)
        
    close_prices = np.array(prices)
    
    # OHLC Türetme (Basit bir yaklaşımla)
    # High = Close ve Open'dan yüksek rastgele bir değer
    # Low = Close ve Open'dan düşük rastgele bir değer
    # Open = Bir önceki günün Close'u (Gapli açılışlar için biraz gürültü eklenebilir)
    
    opens = np.zeros(duration_days)
    highs = np.zeros(duration_days)
    lows = np.zeros(duration_days)
    volumes = np.zeros(duration_days)
    
    opens[0] = start_price
    # İlk günden sonraki Open'lar bir önceki Close olsun (hafif kayma ile)
    for i in range(1, duration_days):
        opens[i] = close_prices[i-1] * np.random.uniform(0.999, 1.001)
        
    for i in range(duration_days):
        daily_vol = max(opens[i], close_prices[i]) * (volatility * np.random.uniform(0.5, 1.5))
        
        high_val = max(opens[i], close_prices[i]) + daily_vol * np.random.uniform(0, 1)
        low_val = min(opens[i], close_prices[i]) - daily_vol * np.random.uniform(0, 1)
        
        highs[i] = high_val
        lows[i] = low_val
        
        # Hacim de rastgele ( fiyat değişimi ile korele olabilir ama şimdilik random)
        volumes[i] = np.random.randint(1000, 100000) * (1 + abs(shock[i]))

    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    })
    
    df.set_index('Date', inplace=True)
    return df

if __name__ == "__main__":
    # Test
    df_sync = generate_synthetic_data(duration_days=365)
    print(df_sync.head())
    print("\nSon Fiyat:", df_sync.Close[-1])
