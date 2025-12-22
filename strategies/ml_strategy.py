from backtesting import Strategy

class MLStrategy(Strategy):
    """
    Makine Öğrenmesi Modeli Stratejisi.
    
    Bu strateji, DataFrame içinde önceden hesaplanmış 'ML_Signal' ve 'ML_Prob' 
    sütunlarını kullanır. Model karmaşıklığı dışarıda (run_ml_backtest.py) halledilir.
    """
    
    # Olasılık Eşiği (Threshold)
    buy_threshold = 0.60 
    sell_threshold = 0.40
    
    def init(self):
        # Sütunları kolay erişim için tanımla
        self.signal = self.I(lambda x: x, self.data.ML_Signal)
        # Olasılıkları kullanmak için (Sütun adı run_ml_backtest.py içinde 'ML_Prob' olarak ayarlanmalı)
        self.prob = self.I(lambda x: x, self.data.ML_Prob)

    def next(self):
        # 1. Pozisyon Yönetimi (Stop Loss / Take Profit) - %5 Stop, %15 TP (Biraz genişletelim)
        if self.position:
            pl_pct = self.position.pl_pct
            if pl_pct < -0.05:
                self.position.close()
                return
            if pl_pct > 0.15:
                self.position.close()
                return

        # 2. Model Sinyalleri (Kademeli / Olasılık Bazlı)
        # Modelin 'Yükseliş' (1) deme ihtimali
        prob_up = self.prob[-1]
        
        # --- LONG GİRİŞ ---
        # Model %60'dan fazla eminse AL
        if prob_up > self.buy_threshold:
            # Short varsa kapat
            if self.position.is_short:
                self.position.close()
            # Long yoksa aç
            if not self.position.is_long:
                self.buy()
                
        # --- SHORT GİRİŞ ---
        # Model %40'tan az "Artacak" diyorsa (Yani %60 "Düşecek" diyorsa)
        elif prob_up < self.sell_threshold:
            # Long varsa kapat
            if self.position.is_long:
                self.position.close()
            # Short yoksa aç
            if not self.position.is_short:
                self.sell()
                
        # --- ÇIKIŞ (NÖTR BÖLGE) ---
        # Eğer olasılık %40-%60 arasındaysa (Model kararsızsa) risk alma, nakite geç.
        else:
            if self.position:
                self.position.close()
