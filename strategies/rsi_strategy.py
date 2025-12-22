from backtesting import Strategy
import pandas as pd
import pandas_ta as ta

def RSI(values, n=14):
    """
    Return RSI of `values` using pandas-ta logic wrapper or simple calculation.
    Since backtesting.py expects a simple function, we can use pandas_ta or manual calculation.
    """
    # Using pandas_ta for robust RSI calculation
    # values is a numpy array from backtesting.py, convert to Series
    s = pd.Series(values)
    return ta.rsi(s, length=n)

class RsiStrategy(Strategy):
    rsi_period = 14
    oversold = 30
    overbought = 70
    
    def init(self):
        # Calculate RSI
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), self.rsi_period)
        
    def next(self):
        # If RSI < oversold (30), BUY
        if self.rsi[-1] < self.oversold:
            if not self.position.is_long:
                self.buy()
                
        # If RSI > overbought (70), SELL (Close position)
        elif self.rsi[-1] > self.overbought:
            if self.position.is_long:
                self.position.close()
