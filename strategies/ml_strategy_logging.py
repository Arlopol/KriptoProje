
from strategies.ml_strategy_trend import MLStrategyTrend
import pandas as pd

class MLStrategyLogging(MLStrategyTrend):
    """
    Laboratuvar modu için 'Konuşkan' Strateji.
    Her adımda ne düşündüğünü (Logs) kaydeder.
    """
    
    def init(self):
        super().init()
        # Günlük tutacağımız liste
        self.decision_logs = []

    def next(self):
        # 1. Verileri Topla
        date = self.data.index[-1]
        price = self.data.Close[-1]
        sma = self.sma200[-1]
        prob = self.prob[-1]
        
        # Trend Durumu
        is_bull = price > sma
        trend_str = "BOĞA (Yükseliş)" if is_bull else "AYI (Düşüş)"
        
        action = "BEKLE"
        reason = "Analiz ediliyor..."
        
        # --- KARAR MEKANİZMASI (Logging için) ---
        # Burada aslında MLStrategyTrend'in mantığını simüle ediyoruz
        # Çünkü super().next() çağırsak bile emrin gerçekleşip gerçekleşmediğini o an anlayamayız.
        # Bu yüzden "Niyetimizi" logluyoruz.

        # Mevcut Durum
        is_long = self.position.is_long
        is_short = self.position.is_short
        
        # 1. Stop Loss / Take Profit Kontrolü (Bizim kontrolümüzde değil, Backtesting.py otomatik yapmazsa burası yapar)
        # MLStrategyTrend stop-loss mantığı:
        pl_pct = self.position.pl_pct
        closed_reason = ""
        
        if self.position:
            if pl_pct < -0.05:
                closed_reason = "Stop Loss (%5 Zarar) Tetiklendi. Pozisyon Kapatılıyor."
            elif pl_pct > 0.15:
                closed_reason = "Take Profit (%15 Kar) Tetiklendi. Pozisyon Kapatılıyor."
        
        # 2. Sinyal Kontrolü
        if closed_reason:
            action = "🚫 POZİSYON KAPAT"
            reason = closed_reason
            
        else:
            # --- LONG SİNYALİ ---
            # Threshold'ları elle yazıyoruz (Inheritance sorunu riskine karşı)
            THRESHOLD_BUY = 0.60
            THRESHOLD_SELL = 0.40
            
            if prob > THRESHOLD_BUY: # 0.60
                if is_short:
                    action = "🔄 TERS İŞLEM (SHORT -> LONG)"
                    reason = f"Model fikrini değiştirdi (Güven: %{prob*100:.0f}). Short kapatıp Long açılıyor."
                elif is_long:
                    action = "POZİSYONU KORU (LONG)"
                    reason = f"Yükseliş beklentisi devam ediyor (Güven: %{prob*100:.0f})."
                else: # Nakit
                    action = "🟢 ALIM (LONG)"
                    reason = f"Güçlü Yükseliş Sinyali (Güven: %{prob*100:.0f}). Trend: {trend_str}."
                    
            # --- SHORT SİNYALİ ---
            elif prob < THRESHOLD_SELL: # 0.40
                if is_bull: # Trend Boğa ise Short YASAK
                    if is_long:
                        action = "🚫 POZİSYON KAPAT (NAKİTE GEÇ)"
                        reason = f"Model düşüş bekliyor (Güven: %{(1-prob)*100:.0f}) FAKAT Trend Boğa olduğu için Short açılmıyor, sadece Long kapatılıyor."
                    elif is_short:
                         # Bu durum teorik olarak olmamalı (Boğada short tutmamalıydık) ama olduysa kapat.
                        action = "🚫 POZİSYON KAPAT"
                        reason = "Boğa piyasasındayız, Short pozisyon kapatılıyor."
                    else: # Nakit
                        action = "✋ PAS GEÇ (SHORT YOK)"
                        reason = f"Model düşüş bekliyor (Güven: %{(1-prob)*100:.0f}). ANCAK Fiyat > SMA200 (Boğa) olduğu için Short işlem açılması riskli bulundu ve engellendi."
                
                else: # Trend AYI (Short SERBEST)
                    if is_long:
                         action = "🔄 TERS İŞLEM (LONG -> SHORT)"
                         reason = f"Model düşüş bekliyor. Ayı piyasası teyitli. Long kapat, Short aç."
                    elif is_short:
                        action = "POZİSYONU KORU (SHORT)"
                        reason = f"Düşüş beklentisi devam ediyor (Güven: %{(1-prob)*100:.0f})."
                    else: # Nakit
                        action = "🔴 SATIŞ (SHORT)"
                        reason = f"Güçlü Düşüş Sinyali (Güven: %{(1-prob)*100:.0f}). Trend: Ayı."
            
            # --- NÖTR SİNYAL ---
            else:
                if is_long or is_short:
                    action = "🚫 POZİSYON KAPAT (NÖTR)"
                    reason = f"Model kararsız (%{prob*100:.0f}). Riski azaltmak için pozisyon kapatılıyor."
                else:
                    action = "BEKLE"
                    reason = f"Model kararsız (%{prob*100:.0f}). Güvenli limanda (Nakit) bekleniyor."

        # Log Kaydı
        # Fiyatı 1000 ile çarpıyoruz (Workaround yüzünden)
        real_price = float(price) * 1000.0
        
        log_entry = {
            "Date": str(date),
            "Price": real_price, # Native float
            "Trend": str(trend_str),
            "Model_Conf": float(prob),
            "Action": str(action),
            "Reason": str(reason),
            "Balance": float(self.equity)
        }
        self.decision_logs.append(log_entry)
        
        # Gerçek İşlemi Yapması için Stratejiyi Çalıştır
        super().next()
