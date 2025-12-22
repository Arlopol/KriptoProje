from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover
import pandas as pd

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(lambda x: pd.Series(x).rolling(10).mean(), price)
        self.ma2 = self.I(lambda x: pd.Series(x).rolling(20).mean(), price)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

print("Veri Seti (GOOG):")
print(GOOG.head())

bt = Backtest(GOOG, SmaCross, cash=10000, commission=.002)
stats = bt.run()
print(stats)
