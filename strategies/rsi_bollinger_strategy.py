from backtesting import Strategy
import pandas as pd
import pandas_ta as ta

class RsiBollingerStrategy(Strategy):
    # Parameters
    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    oversold = 30
    overbought = 70
    
    def init(self):
        # Calculate RSI
        # data.Close is a numpy array in backtesting.py, so we convert to Series for pandas-ta
        close_series = pd.Series(self.data.Close)
        self.rsi = self.I(ta.rsi, close_series, self.rsi_period)
        
        # Calculate Bollinger Bands
        # using standard pandas rolling to avoid pandas-ta conflicts
        def get_bbands(values, length, std):
            # values is numpy array, convert to Series
            s = pd.Series(values)
            ma = s.rolling(length).mean()
            sd = s.rolling(length).std()
            
            upper = ma + std * sd
            lower = ma - std * sd
            
            # Backfill NaNs to avoid issues, or let backtesting handle it (usually NaNs are fine at start)
            # conversion to numpy array
            return lower.values, upper.values

        # self.I can handle functions returning multiple arrays
        self.lower_band, self.upper_band = self.I(get_bbands, self.data.Close, self.bb_period, self.bb_std)
        
    def next(self):
        price = self.data.Close[-1]
        rsi_val = self.rsi[-1]
        lower_bb = self.lower_band[-1]
        upper_bb = self.upper_band[-1]
        
        # BUY SIGNAL: Price < Lower Band AND RSI < Oversold
        if price < lower_bb and rsi_val < self.oversold:
            if not self.position.is_long:
                self.buy()
                
        # SELL SIGNAL: Price > Upper Band AND RSI > Overbought
        elif price > upper_bb and rsi_val > self.overbought:
            if self.position.is_long:
                self.position.close()
