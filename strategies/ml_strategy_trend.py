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
    stop_loss_pct = 0.05
    take_profit_pct = 0.15
    
    # Trend Filtresi (Açık/Kapalı)
    use_trend_filter = True
    
    # Trailing Stop Ayarları
    use_trailing_stop = False
    trailing_decay = 0.10 # Zirveden %10 düşerse sat
    
    def init(self):
        self.signal = self.I(lambda x: x, self.data.ML_Signal)
        # ML_Prob sütunu şart
        self.prob = self.I(lambda x: x, self.data.ML_Prob)
        # SMA_200 Feature Mühendisliğinden geliyor olmalı
        self.sma200 = self.I(lambda x: x, self.data.SMA_200)
        
        # Trailing Stop için Takip
        self.peak_pl = 0

    def next(self):
        # 1. Pozisyon Yönetimi (Standart + Trailing)
        if self.position:
            pl_pct = self.position.pl_pct
            
            # Trailing Stop Logic
            if self.use_trailing_stop:
                # Zirve Karı Güncelle
                self.peak_pl = max(self.peak_pl, pl_pct)
                
                # Zirveden geri çekilme kontrolü (Sadece kardaysak veya belirli eşik?)
                # Genelde trailing stop kara geçince aktif olur ama basitlik için zirveden düşüşe bakalım.
                # Eğer Zirve %20 ise ve Decay %10 ise, %10'a düşünce satar.
                if (self.peak_pl - pl_pct) > self.trailing_decay:
                    self.position.close()
                    self.peak_pl = 0
                    return
            
            # Sabit Stop Loss (Her zaman aktif - Sigorta)
            if pl_pct < -self.stop_loss_pct:
                self.position.close()
                self.peak_pl = 0
                return
                
            # Sabit Take Profit İPTAL EDİLDİ
            # Boğa piyasasında erken çıkmayı engellemek için TP'yi kaldırıyoruz.
            # Sadece Trailing Stop veya Trend Dönüşü (Sell Sinyali) ile çıkılacak.
            # if not self.use_trailing_stop and pl_pct > self.take_profit_pct:
            #    self.position.close()
            #    self.peak_pl = 0
            #    return
        else:
            self.peak_pl = 0 # Pozisyon yoksa sıfırla

        # 2. Trend Analizi
        price = self.data.Close[-1]
        sma = self.sma200[-1]
        
        # Trend Filtresi: Fiyat SMA'nın üzerindeyse BOĞA'dır.
        is_bull_market = price > sma
        
        prob_up = self.prob[-1]
        
        # EĞER BOĞA PİYASASINDAYSAK (Fiyat > SMA) VE Filtre Açıksa, Alım Eşiğini Düşür!
        # Çünkü genel rüzgar arkamızda, modelin azıcık olumlu olması yeterli.
        current_buy_thresh = self.buy_threshold
        if self.use_trend_filter and is_bull_market:
            current_buy_thresh = 0.50 # %50 üstü her olumlu sinyalde al
        
        # --- LONG GİRİŞ ---
        # Model AL diyor
        if prob_up > current_buy_thresh:
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=0.95)
                
        # --- SHORT GİRİŞ ---
        # Model %60+ SAT diyor (prob_up < 0.40)
        elif prob_up < self.sell_threshold:
            # KRİTİK FİLTRE: Eğer Boğa Piyasasıysa ve FİLTRE AÇIKSA SHORT AÇMA!
            if self.use_trend_filter and is_bull_market:
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
            # Model kararsızsa (%40 - %60 arası) POZİSYONU KAPATMA!
            # Trend stratejisinde "Bırak koşsun" mantığı işler.
            # Sadece Sell Sinyali veya Stop Loss ile çık.
            pass
