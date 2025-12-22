from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class SmaCross(Strategy):
    n1 = 10   # Kısa vadeli SMA
    n2 = 20   # Uzun vadeli SMA

    def init(self):
        # SMA Göstergelerini oluştur
        # self.data.Close backtesting.py'da bir numpy array'dir, ancak pd.Series'e çevirip işlem yapabiliriz
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        # Eğer kısa vade, uzun vadeyi yukarı keserse AL
        if crossover(self.sma1, self.sma2):
            self.buy()

        # Eğer kısa vade, uzun vadeyi aşağı keserse SAT
        elif crossover(self.sma2, self.sma1):
            self.position.close()
