from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import os

# Veri Yükle
try:
    df = pd.read_csv('data/raw/BTC-USD_2y_1d.csv', index_col=0, parse_dates=True)
    df = df.astype(float)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()
    df = df.sort_index()
    print("Veri yüklendi. Shape:", df.shape)
    print(df.head())
except Exception as e:
    print("Veri okuma hatası:", e)
    exit()

# Strateji
class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), price)
        self.ma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), price)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position.close()

# Backtest
try:
    bt = Backtest(df, SmaCross, cash=1000000, commission=.002)
    stats = bt.run()
    print("\n--- SONUÇLAR ---")
    print(stats)
    print("İşlem Sayısı:", stats['# Trades'])
except Exception as e:
    print("Backtest Hatası:", e)
    import traceback
    traceback.print_exc()
