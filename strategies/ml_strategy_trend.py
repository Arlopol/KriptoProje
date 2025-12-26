from backtesting import Strategy

class MLStrategyTrend(Strategy):
    """
    ML Stratejisi + Trend Filtresi (SMA 200).
    
    Mantık:
    - Model "Short" dese bile, eğer Fiyat > SMA 200 ise (Boğa Piyasası), Short açma.
    - Bu sayede güçlü yükseliş trendlerinde terste kalmayı engelle.
    - Ayı piyasasında (Fiyat < SMA 200) hem Long hem Short serbest (veya sadece Short?).
      Şimdilik sadece "Boğada Short Yasak" kuralını uyguluyoruz.
    """
    
    buy_threshold = 0.60 
    sell_threshold = 0.40
    
    def init(self):
        self.signal = self.I(lambda x: x, self.data.ML_Signal)
        # ML_Prob sütunu şart
        self.prob = self.I(lambda x: x, self.data.ML_Prob)
        # SMA_200 Feature Mühendisliğinden geliyor olmalı
        # Eğer yoksa hata verir, robustness scriptinde kontrol edeceğiz.
        self.sma200 = self.I(lambda x: x, self.data.SMA_200)

    def next(self):
        # print("DEBUG: MLStrategyTrend.next called")
        # 1. Pozisyon Yönetimi (Standart)
        if self.position:
            pl_pct = self.position.pl_pct
            if pl_pct < -0.05: # Stop Loss %5
                self.position.close()
                return
            if pl_pct > 0.15: # Take Profit %15
                self.position.close()
                return

        # 2. Trend Analizi
        price = self.data.Close[-1]
        sma = self.sma200[-1]
        
        # Trend Filtresi: Fiyat SMA'nın üzerindeyse BOĞA'dır.
        is_bull_market = price > sma
        
        prob_up = self.prob[-1]
        
        # --- LONG GİRİŞ ---
        # Model %60+ AL diyor
        if prob_up > self.buy_threshold:
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=0.95)
                
        # --- SHORT GİRİŞ ---
        # Model %60+ SAT diyor (prob_up < 0.40)
        elif prob_up < self.sell_threshold:
            # KRİTİK FİLTRE: Eğer Boğa Piyasasıysa SHORT AÇMA!
            if is_bull_market:
                # Sadece var olan Long'u kapat (Nakit'e geç), ama Short açma.
                if self.position.is_long:
                    self.position.close()
                # PASS: Short açmıyoruz.
            else:
                # Ayı piyasası veya Yatay: Short serbest.
                if self.position.is_long:
                    self.position.close()
                if not self.position.is_short:
                    self.sell(size=0.95)
                    
        # --- NÖTR ---
        else:
            if self.position:
                self.position.close()
