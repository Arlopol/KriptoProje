import pandas as pd
import pandas_ta as ta
import numpy as np

def add_technical_indicators(df):
    """
    Verilen DataFrame'e (OHLCV) teknik analiz indikatörleri ekler.
    ML modeli için özellik (feature) üretir.
    """
    df = df.copy()
    
    # 1. Momentum
    df['RSI'] = df.ta.rsi(length=14)
    df['ROC'] = df.ta.roc(length=10) # 10 günlük değişim hızı
    
    # 2. Trend
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    # pandas_ta macd sonuçlarını farklı sütunlarda döndürür, bunları df'e ekleyelim
    df = pd.concat([df, macd], axis=1)
    
    df['SMA_50'] = df.ta.sma(length=50)
    df['SMA_200'] = df.ta.sma(length=200)
    
    # Fiyatın ortalamalara uzaklığı (Normalize edilmiş)
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # 3. Volatilite
    df['ATR'] = df.ta.atr(length=14)
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Bollinger Band Genişliği (Bbandwidth) otomatik hesaplanmış olabilir, kontrol edelim
    # Genelde BBB_20_2.0 sütunu gelir. Biz manuel de hesaplayabiliriz garanti olsun.
    # (Upper - Lower) / Middle
    if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns:
        df['BB_Width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    
    # 4. Getiriler (Returns)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # NaN temizliği (İndikatörlerin hesaplanması için gereken ilk dönemlerin silinmesi)
    df.dropna(inplace=True)
    
    return df

def add_lag_features(df, lags=[1, 2, 3, 5, 7]):
    """
    Geçmiş getirileri (Lagged Returns) ekler.
    Modelin momentumu ve kısa vadeli trendleri öğrenmesini sağlar.
    """
    for lag in lags:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
    return df

def add_rolling_volatility(df, windows=[7, 30]):
    """
    Hareketli volatilite (Standart Sapma) ekler.
    Piyasa rejimi (Durgun/Kaotik) tespiti için önemlidir.
    """
    for window in windows:
        # Log getirilerin standart sapması (Daha sağlıklı volatilite ölçümü)
        df[f'Volatility_{window}'] = df['Log_Return'].rolling(window=window).std()
    return df

def prepare_data_for_ml(df, target_col='Target', shift=-1):
    """
    ML eğitimi için X (Özellikler) ve y (Hedef) hazırlar.
    shift=-1 : Hedef yarının kapanışının bugünden yüksek olup olmadığıdır.
    """
    # 1. Temel İndikatörler
    df = add_technical_indicators(df)
    
    # 2. Yeni Gelişmiş Özellikler
    df = add_lag_features(df)
    df = add_rolling_volatility(df)
    
    # Hedef Oluşturma: Yarın > Bugün ise 1, değilse 0
    # shift(-1) yarının verisini bugünün satırına getirir.
    future_close = df['Close'].shift(shift)
    df[target_col] = (future_close > df['Close']).astype(int)
    
    # NaN temizliği (Lagler ve Target oluştururken ortaya çıkan boşluklar)
    df.dropna(inplace=True)
    
    return df
