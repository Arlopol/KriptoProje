
import numpy as np
import pandas as pd

def generate_synthetic_data(duration_days=365, start_price=60000, volatility=0.02, drift=0.0005, random_seed=None, use_regime_switching=True):
    """
    Gelişmiş Sentetik Veri Üreticisi:
    - Regime Switching (Boğa, Ayı, Yatay)
    - Volatilite Kümelenmesi (GARCH benzeri etki)
    - Ani Şoklar (Jumps)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    start_date = pd.to_datetime('2025-01-01')
    dates = pd.date_range(start=start_date, periods=duration_days, freq='D')
    
    # Parametrelerin "Baz" değerleri (Kullanıcıdan gelenler)
    base_drift = drift
    base_vol = volatility
    
    # REJİMLER: [Drift Çarpanı, Volatilite Çarpanı, Rejim Adı]
    regimes = {
        0: [1.0, 1.0, "Normal"],       # Kullanıcının seçtiği
        1: [5.0, 0.8, "Boğa Rallisi"], # Güçlü Trend, Düşük Volatilite
        2: [-5.0, 2.0, "Ayı Çöküşü"],  # Güçlü Düşüş, Yüksek Volatilite (Korku)
        3: [0.0, 0.5, "Yatay/Sıkıcı"]  # Trend Yok, Çok Düşük Volatilite
    }
    
    current_regime = 0
    regime_duration = 0
    prices = [start_price]
    regime_history = []
    
    dt = 1
    
    for i in range(1, duration_days):
        # 1. Rejim Değişimi Kontrolü (Markov Chain Basitleştirilmiş)
        # Her gün %2 ihtimalle rejim değişir (Ortalama 50 günde bir)
        if use_regime_switching:
            if regime_duration > 14 and np.random.random() < 0.05: # En az 2 hafta sürsün
                # Yeni rejim seç (Mevcut hariç)
                possible = list(regimes.keys())
                possible.remove(current_regime)
                current_regime = np.random.choice(possible)
                regime_duration = 0
            else:
                regime_duration += 1
        
        # 2. Seçili Rejimin Parametrelerini Al
        r_params = regimes[current_regime]
        curr_drift = base_drift if r_params[2] == "Normal" else base_vol * 0.1 * r_params[0] # Drift'i vol cinsinden türetelim ki ölçekli olsun
        # Boğa rallisi için: Vol'un %10'u kadar pozitif günlük getiri çok agresif olur.
        # Basitleştirme:
        if current_regime == 0: curr_drift = base_drift
        elif current_regime == 1: curr_drift = 0.003 # Günlük %0.3 artış
        elif current_regime == 2: curr_drift = -0.004 # Günlük %0.4 düşüş
        elif current_regime == 3: curr_drift = 0.0
        
        curr_vol = base_vol * r_params[1]
        
        regime_history.append(r_params[2])

        # 3. Fiyat Hareketi (GBM + Jump)
        shock = np.random.normal(0, 1)
        
        # Black Swan / Ani Şok İhtimali (%0.5)
        jump = 0
        if np.random.random() < 0.005:
            jump = np.random.normal(0, 5) * curr_vol # 5 sigma hareket
        
        drift_comp = (curr_drift - 0.5 * curr_vol**2) * dt
        shock_comp = curr_vol * shock * np.sqrt(dt)
        change = np.exp(drift_comp + shock_comp + jump)
        
        new_price = prices[-1] * change
        
        # Negatif fiyat koruması
        if new_price < 1000: new_price = 1000 
        
        prices.append(new_price)

    close_prices = np.array(prices)
    
    # OHLC Üretimi (Daha gerçekçi mumlar)
    opens = np.zeros(duration_days)
    highs = np.zeros(duration_days)
    lows = np.zeros(duration_days)
    volumes = np.zeros(duration_days)
    
    opens[0] = start_price
    for i in range(1, duration_days):
        # Open genellikle önceki Close'a yakındır ama gün içi volatilitenin ufak bir kısmı kadar gap olabilir
        opens[i] = close_prices[i-1] * np.random.uniform(0.9995, 1.0005)
        
    for i in range(duration_days):
        # Mumun gövdesi
        body_max = max(opens[i], close_prices[i])
        body_min = min(opens[i], close_prices[i])
        
        # Fitiller (Shadows)
        # Volatiliteye bağlı olarak fitil boyu
        daily_movement = abs(close_prices[i] - opens[i])
        avg_price = (close_prices[i] + opens[i]) / 2
        
        # Rejime göre fitil karakteri (Ayı piyasasında aşağı fitil uzun olabilir vs. ama şimdilik rastgele)
        regime_idx = 0 # Default
        if use_regime_switching and i > 0 and i-1 < len(regime_history):
             # Bu index mantığı biraz kayık olabilir ama approx yeterli
             pass 

        # High, Body Max'tan yukarıda olmalı
        # Low, Body Min'den aşağıda olmalı
        # Günlük değişimden bağımsız bir "gün içi volatilite" faktörü ekleyelim
        # Yaklaşık günlük vol kadar range olsun
        day_range = avg_price * base_vol * np.random.uniform(0.8, 1.2)
        
        # Eğer mum gövdesi range'den büyükse range'i büyüt
        if daily_movement > day_range: day_range = daily_movement * 1.1
        
        # Range'in ne kadarı yukarıda ne kadarı aşağıda? Rastgele
        up_shadow_ratio = np.random.random()
        down_shadow_ratio = 1 - up_shadow_ratio
        
        # Kalan range (Gövde dışındaki kısım)
        excess_range = max(0, day_range - daily_movement)
        
        highs[i] = body_max + excess_range * up_shadow_ratio
        lows[i] = body_min - excess_range * down_shadow_ratio
        
        # Hacim: Trend dönüşlerinde ve yüksek volatilitede artar
        # Basit model: Fiyat değişimi yüzdesi ile hacim korele olsun
        price_change_pct = abs(close_prices[i] - opens[i]) / opens[i]
        base_volume = 10000
        vol_multiplier = 1 + (price_change_pct * 100) # %1 değişimde hacim 2 katına çıkar
        
        volumes[i] = base_volume * vol_multiplier * np.random.uniform(0.8, 1.2)

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
