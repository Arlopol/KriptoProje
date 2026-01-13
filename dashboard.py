import streamlit as st
import pandas as pd
import json
import os
import glob
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Sayfa Ayarları
st.set_page_config(
    page_title="Kripto Proje Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode CSS (Streamlit zaten dark mode destekler ama özelleştirme için)
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-title { color: #888; font-size: 14px; }
    .metric-value { color: #fff; font-size: 24px; font-weight: bold; }
    .positive { color: #00ff00; }
    .negative { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Başlık
st.title("🚀 Kripto Proje Sonuç Analizi")
st.markdown("Geçmişte yapılan tüm strateji testlerini ve sonuçlarını buradan inceleyebilirsiniz.")

# Verileri Yükle
REPORT_DIR = "reports"
files = glob.glob(os.path.join(REPORT_DIR, "*.json"))

data = []
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as json_file:
            content = json.load(json_file)
            # Dosya adını benzersiz anahtar olarak ekle
            content['json_filename'] = os.path.basename(f)
            data.append(content)
    except Exception as e:
        st.error(f"Dosya okuma hatası ({f}): {e}")

if not data:
    st.warning("Henüz hiç rapor bulunmamaktadır. Lütfen önce `run_backtest.py` çalıştırın.")
    st.stop()

# DataFrame'e Çevir
# pd.json_normalize kullanarak iç içe yapıları düzleştiriyoruz
df = pd.json_normalize(data)

# Kolon isimlerini eşitle (Robustness raporlarında farklı anahtarlar olabiliyor)
if 'date' not in df.columns and 'timestamp' in df.columns:
    df['date'] = df['timestamp']
elif 'date' in df.columns and 'timestamp' in df.columns:
    df['date'] = df['date'].fillna(df['timestamp'])

if 'strategy' not in df.columns and 'model' in df.columns:
    df['strategy'] = df['model']
elif 'strategy' in df.columns and 'model' in df.columns:
    df['strategy'] = df['strategy'].fillna(df['model'])

if 'date' not in df.columns:
    st.error("Raporlarda 'date' veya 'timestamp' alanı bulunamadı!")
    st.stop()

# Sıralama
df = df.sort_values(by='date', ascending=False)

# Kategorilere Ayırma
df['is_robustness'] = df.apply(lambda x: isinstance(x.get('results'), list), axis=1)
df['is_monte_carlo'] = df.apply(lambda x: pd.notna(x.get('simulation_results.mean_equity')), axis=1)

# Sidebar - Kenar Çubuğu
st.sidebar.header("📂 Rapor Gezgini")

# 1. Kategori Seçimi
category = st.sidebar.radio("Kategori:", ["📈 Model Sonuçları", "🛡️ Sağlamlık Testleri", "🎲 Simülasyon Testleri", "🧪 Laboratuvar (Canlı Test)"])

selected_filename = None

if category == "🧪 Laboratuvar (Canlı Test)":
    st.header("🧪 Laboratuvar: Geçmişi Yeniden Yaşa")
    st.info("Bu modda, yapay zekayı belirli bir tarih aralığında çalıştırıp **'Neden?'** sorusuna cevap arayabilirsiniz.")
    
    # Girdiler
    c1, c2, c3 = st.columns(3)
    with c1:
        lab_symbol = st.selectbox("Sembol (Coin)", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"])
    with c2:
        # Varsayılan: Son 1 yıl
        default_start = datetime.now() - timedelta(days=365)
        start_date = st.date_input("Başlangıç Tarihi", value=default_start)
    with c3:
        end_date = st.date_input("Bitiş Tarihi", value=datetime.now())
        
    initial_capital = st.number_input("Başlangıç Sermayesi ($)", value=10000, step=1000)
    
    with st.expander("🛠️ Gelişmiş Ayarlar (Risk & Strateji)", expanded=False):
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            lab_buy_thresh = st.slider("Alış Eşiği (Güven %)", 0.50, 0.90, 0.60, 0.01)
            lab_sell_thresh = st.slider("Satış Eşiği (Güven %)", 0.10, 0.50, 0.40, 0.01)
        with c_adv2:
            lab_sl = st.slider("Stop Loss (%)", 0.01, 0.20, 0.05, 0.01)
            lab_tp = st.slider("Take Profit (%)", 0.05, 0.50, 0.15, 0.01)
            lab_use_trend = st.checkbox("Trend Filtresi (SMA 200)", value=True, help="Boğa piyasasında Short açmayı engeller.")
            lab_use_dynamic = st.checkbox("🧠 Akıllı Sermaye (Güvene Göre)", value=False, help="Düşük güven varsa az para yatırır.")
    
    if st.button("🚀 Senaryoyu Çalıştır", type="primary"):
        with st.spinner(f"{lab_symbol} için Yapay Zeka Düşünüyor..."):
            from backtest.run_scenario import run_scenario
            results = run_scenario(str(start_date), str(end_date), initial_capital, symbol=lab_symbol,
                                   buy_threshold=lab_buy_thresh, sell_threshold=lab_sell_thresh, 
                                   stop_loss=lab_sl, take_profit=lab_tp, use_trend=lab_use_trend,
                                   use_dynamic_sizing=lab_use_dynamic)
            
            if "error" in results:
                st.error(results["error"])
            else:
                # Sonuçları Göster
                st.subheader("📊 Test Sonuçları")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Son Sermaye", f"${results['final_equity']:,.0f}", f"%{results['return_pct']:.2f}")
                
                bh_ret = results.get('bh_return_pct', 0)
                alpha = results['return_pct'] - bh_ret
                m2.metric("Al-Tut Getirisi", f"%{bh_ret:.2f}", f"Fark: %{alpha:.2f}")

                m3.metric("İşlem Sayısı", results['total_trades'])
                m4.metric("Max Drawdown", f"%{results['max_drawdown']:.2f}")
                m5.metric("Kazanma Oranı", f"%{results['win_rate']:.1f}")
                
                # Model Performansı (Genel Accuracy)
                metrics = results.get('metrics', {})
                train_metrics = results.get('train_metrics', {})
                acc_val = metrics.get('accuracy', 0) * 100
                train_acc = train_metrics.get('accuracy', 0) * 100
                m6.metric("Model Doğruluğu", f"%{acc_val:.1f}", f"Eğitim: %{train_acc:.1f}", delta_color="normal")
                
                # --- DETAYLI METRİKLER ---
                with st.expander("📈 Detaylı Model Performansı (Accuracy, Precision, Recall)", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Doğruluk (Accuracy)", f"%{metrics.get('accuracy',0)*100:.1f}", help="Doğru tahmin oranı")
                    c2.metric("Keskinlik (Precision)", f"%{metrics.get('precision',0)*100:.1f}", help="Al dediğinde ne kadar haklıydı?")
                    c3.metric("Duyarlılık (Recall)", f"%{metrics.get('recall',0)*100:.1f}", help="Yükselişlerin ne kadarını yakaladı?")
                    c4.metric("F1 Skoru", f"%{metrics.get('f1',0)*100:.1f}", help="Denge puanı")
                
                # --- İNTERAKTİF GRAFİK (Log Visualization) ---
                st.markdown("---")
                st.subheader("🧠 Yapay Zeka Günlüğü (Görsel Analiz)")
                st.info("Grafik üzerindeki noktalara gelerek yapay zekanın **neden** o kararı verdiğini okuyabilirsiniz.")
                
                logs = results.get('logs', [])
                if logs:
                    df_logs = pd.DataFrame(logs)
                    df_logs['Date'] = pd.to_datetime(df_logs['Date'])
                    
                    fig_log = go.Figure()
                    
                    # 1. Fiyat Çizgisi
                    fig_log.add_trace(go.Scatter(
                        x=df_logs['Date'], 
                        y=df_logs['Price'],
                        mode='lines',
                        name='Fiyat',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # 2. İşlem Noktaları (Renkli Markerlar)
                    # Alım, Satım ve Bekle için ayrı renkler ve hover textler
                    
                    # Renk Haritası
                    color_map = {
                        "ALIM": "green",
                        "SATIŞ": "red",
                        "PAS GEÇ": "orange",
                        "BEKLE": "gray",
                        "POZİSYONU KORU": "blue",
                        "TERS İŞLEM": "purple",
                        "POZİSYON KAPAT": "black"
                    }
                    
                    # Hover Template
                    df_logs['Color'] = df_logs['Action'].apply(lambda x: next((v for k, v in color_map.items() if k in x), "gray"))
                    
                    # Marker Boyutu (Önemli aksiyonlar büyük)
                    df_logs['Size'] = df_logs['Action'].apply(lambda x: 12 if "ALIM" in x or "SATIŞ" in x or "KAPAT" in x else 6)
                    
                    fig_log.add_trace(go.Scatter(
                        x=df_logs['Date'],
                        y=df_logs['Price'],
                        mode='markers',
                        name='Kararlar',
                        marker=dict(
                            color=df_logs['Color'],
                            size=df_logs['Size'],
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=df_logs['Action'], # Hover başlığı
                        customdata=df_logs['Reason'], # Hover detay
                        hovertemplate="<b>%{text}</b><br>Fiyat: $%{y:,.0f}<br>💭 <i>%{customdata}</i><extra></extra>"
                    ))
                    
                    fig_log.update_layout(
                        title="Yapay Zeka Karar Haritası",
                        xaxis_title="Tarih",
                        yaxis_title="Fiyat ($)",
                        height=600,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_log, use_container_width=True)
                    
                    # Tablo Gösterimi (İsteğe Bağlı)
                    with st.expander("📜 Detaylı Log Listesi (Tablo)"):
                        st.dataframe(df_logs[['Date', 'Action', 'Price', 'Trend', 'Reason']])

    # Laboratuvar modu seçiliyse aşağısındaki standart raporu gösterme
    st.stop()



elif category == "🎲 Simülasyon Testleri":
    # Son yapılan analizin sembolünü bulmaya çalış
    wf_last_path = "reports/walk_forward_last_run.json"
    wf_symbol_display = ""
    if os.path.exists(wf_last_path):
        try:
            with open(wf_last_path, 'r', encoding='utf-8') as f:
                last_res = json.load(f)
                sym = last_res.get('symbol', 'Bilinmiyor')
                wf_symbol_display = f" - (Analiz Kaynağı: {sym})"
        except: pass

    st.header(f"🎲 Monte Carlo Simülasyonu{wf_symbol_display}")
    
    # İki Alt Mod: Yeni Simülasyon veya Rapor Görüntüle
    mc_mode = st.radio("Seçiminiz:", ["🔍 Geçmiş Raporları İncele", "⚡ Yeni Simülasyon Başlat"], horizontal=True)
    
    if mc_mode == "⚡ Yeni Simülasyon Başlat":
        st.info("**Monte Carlo Mantığı:** Geçmişteki gerçek işlemlerinizin sırası rastgele değiştirilerek (Reshuffling) 1000'lerce 'Alternatif Senaryo' üretilir. Amaç, şans faktörünü ölçmek ve 'En kötü durumda ne olurdu?' sorusuna yanıt bulmaktır.\n\n*Not: Bu test sentetik fiyat üretmez, gerçek işlemlerinizi kullanır.*")
        
        cc1, cc2 = st.columns(2)
        with cc1:
            strat_choice = st.selectbox("Strateji Seçimi", ["Professional", "Adventurous"])
            mc_capital = st.number_input("Başlangıç Sermayesi ($)", value=10000, step=1000, key="mc_cap")
        with cc2:
            sim_count = st.slider("Simülasyon Sayısı (Adet)", 100, 5000, 1000, step=100)
            horizon_count = st.slider("İşlem Derinliği (Adet)", 50, 500, 150, help="Her simülasyonda kaç adet işlem yapılacak?")
            
        if st.button("🎲 Zarları At (Simülasyonu Başlat)", type="primary"):
            with st.spinner("Binlerce paralel evren simüle ediliyor..."):
                try:
                    from backtest.run_monte_carlo import run_simulation_for_dashboard
                    res_mc = run_simulation_for_dashboard(strategy_name=strat_choice, initial_capital=mc_capital, simulations=sim_count, horizon=horizon_count)
                    
                    if "error" in res_mc:
                        st.error(res_mc["error"])
                    else:
                        st.success("✅ Simülasyon Tamamlandı!")
                        # Sonuçları ekrana basmak yerine reports listesine yönlendirmek daha kolay olabilir
                        # Ama kullanıcı anlık görmek ister.
                        # Buradaki variable ismini 'run_data' yaparsak aşağıdaki kod otomatik gösterir mi?
                        # run_data aşağıda tanımlanıyor. Biz burada direkt run_data'yı set edelim.
                        run_data = pd.Series(res_mc) # Dict to Series
                        selected_filename = "CANLI_TEST" # Dummy
                        
                        # Aşağıdaki kod bloğu 'run_data' üzerinden çalıştığı için
                        # Buradan sonrasını manipüle edebiliriz.
                        # Ancak kodun akışı 'elif' bloklarından çıkıp aşağıya gidiyor.
                        # O yüzden burada 'run_data'yı global scope'a çıkarmamız lazım veya
                        # aşağıya 'goto' yapamadığımız için kodu kopyalamak veya yapılandırmak lazım.
                        
                        # ÇÖZÜM: Sonuçları session state'e atıp rerun() diyebiliriz ya da
                        # Direkt kodun geri kalanını kullanmak için selected_filename'i set edip
                        # df'ye bu yeni raporu ekleyebiliriz (Karmaşık).
                        
                        # En temizi: Sonuçları burada gösterelim ve st.stop() diyelim.
                        
                        # --- SONUÇ GÖSTERİMİ (Copy-Paste from below with modifications) ---
                        st.divider()
                        st.subheader(f"📊 Sonuçlar: {res_mc['model']}")
                        
                        sim_res = res_mc['simulation_results']
                        sim_meta = res_mc['simulation_meta']
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Ortalama Sermaye", f"${sim_res['mean_equity']:,.0f}", f"ROI: %{sim_meta['mean_roi_pct']}")
                        c2.metric("Kötü Senaryo (%5)", f"${sim_res['p05_equity']:,.0f}", delta="Risk", delta_color="inverse")
                        c3.metric("Batış Riski", f"%{sim_res['risk_of_ruin_50pct']:.2f}")
                        c4.metric("Tahmini Süre", f"{sim_meta['simulated_duration_years']} Yıl")
                        
                        # --- ML METRİKLERİ ---
                        st.divider()
                        st.subheader("🤖 Model Performansı (Tüm Dönem)")
                        mm = res_mc.get('model_metrics', {})
                        if mm:
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Doğruluk (Acc)", f"%{mm.get('accuracy', 0)*100:.1f}")
                            m2.metric("Keskinlik (Prec)", f"%{mm.get('precision', 0)*100:.1f}")
                            m3.metric("Duyarlılık (Rec)", f"%{mm.get('recall', 0)*100:.1f}")
                            m4.metric("F1 Skoru", f"%{mm.get('f1', 0)*100:.1f}")
                        
                        # Histogram
                        dist_data = res_mc['data_samples']['final_equities']
                        import plotly.express as px
                        fig = px.histogram(x=dist_data, nbins=50, title="Olası Sonuç Dağılımı", color_discrete_sequence=['#00CC96'])
                        fig.add_vline(x=mc_capital, line_dash="dash", line_color="white", annotation_text="Başlangıç")
                        fig.add_vline(x=sim_res['p05_equity'], line_dash="dot", line_color="red", annotation_text="Kötü Senaryo")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(f"Detaylı rapor kaydedildi: {res_mc['json_filename']}")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.stop()
        
    # MEVCUT RAPORLARI GÖSTERME (Eski Kodun Devamı)
    df_display = df[df['is_monte_carlo'] == True].sort_values(by='date', ascending=False)
    if df_display.empty:
        st.sidebar.warning("Henüz simülasyon testi raporu yok.")
    else:
        selected_filename = st.sidebar.radio(
            "Test Seçiniz:",
            df_display['json_filename'].tolist(),
            format_func=lambda x: f"{df_display[df_display['json_filename']==x]['date'].values[0]} | {df_display[df_display['json_filename']==x]['model'].values[0] if 'model' in df_display.columns else 'Monte Carlo'}"
        )
        st.sidebar.caption(f"Dosya: {selected_filename}")

elif category == "🛡️ Sağlamlık Testleri":
    st.header("🛡️ Sağlamlık (Robustness) Testleri")
    
    # Alt Mod Seçimi: Yeni Test vs Raporlar
    # Eğer henüz hiç rapor yoksa direkt yeni teste yönlendir
    rb_mode = st.radio("Seçiminiz:", ["🔍 Geçmiş Raporları İncele", "⚡ Yeni Test Başlat"], horizontal=True, index=1)
    
    if rb_mode == "⚡ Yeni Test Başlat":
        test_type = st.radio("Test Türü:", ["🧪 Optimizasyon (Grid Search)", "🔴 Yürüyen Analiz (Walk-Forward)"], horizontal=True)
        
        if test_type == "🧪 Optimizasyon (Grid Search)":
            st.info("Bu mod, en iyi parametreleri bulmak için çoklu testler yapar.")
            
            use_ultra = st.toggle("🔥 ULTRA MOD (Tüm Kombinasyonları Dene)", value=False)
            
            if use_ultra:
                st.warning("⚠️ Bu mod Sentiment ve On-Chain verilerinin OLAN ve OLMAYAN tüm hallerini dener. Süre uzayabilir!")
                # Otomatik Grid
                grid_buy = [0.60, 0.70]
                grid_sl = [0.05, 0.10]
                grid_tp = [0.15, 0.30]
                # Veri Kaynakları da Grid'e dahil
                grid_sent = [False, True]
                grid_oc = [False, True]
                # Trailing de test et
                grid_trail_use = [False, True]
                grid_trail_decay = [0.10]
                
                st.write("Ultra Mod Ayarları Otomatik Yüklendi ✅")
            else:
                # Session State Başlatma (İlk kez çalışıyorsa)
                if 'grid_buy' not in st.session_state: st.session_state.grid_buy = [0.60, 0.75]
                if 'grid_sl' not in st.session_state: st.session_state.grid_sl = [0.05, 0.10]
                if 'grid_tp' not in st.session_state: st.session_state.grid_tp = [0.15, 0.30]

                # Tümünü Seç Butonları
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("✅ Tüm Alım Eşiklerini Seç"):
                    st.session_state.grid_buy = [0.55, 0.60, 0.65, 0.70, 0.75]
                if col_btn2.button("✅ Tüm Risk Ayarlarını Seç"):
                    st.session_state.grid_sl = [0.05, 0.10, 0.15]
                    st.session_state.grid_tp = [0.15, 0.25, 0.30, 0.35, 0.50]

                # Manuel Seçim
                c1, c2 = st.columns(2)
                grid_buy = c1.multiselect("Alım Eşiği (Buy Thresholds)", [0.55, 0.60, 0.65, 0.70, 0.75], key='grid_buy')
                grid_sl = c2.multiselect("Stop Loss (Zarar Kes)", [0.05, 0.10, 0.15], key='grid_sl')
                
                c3, c4 = st.columns(2)
                grid_tp = c3.multiselect("Take Profit (Kar Al)", [0.15, 0.25, 0.30, 0.35, 0.50], key='grid_tp')
                
                # Trailing için Grid
                grid_trail_use = [False]
                grid_trail_decay = [0.10]
                if st.checkbox("İz Süren Stop (Trailing) Kombinasyonlarını da Dene?", value=False):
                    grid_trail_use = [False, True]
                    grid_trail_decay = c4.multiselect("Trailing Decay Ayarları", [0.05, 0.10, 0.15], [0.10])

                grid_sent = [True] # Varsayılan: Hepsi açık olsun manuelde
                grid_oc = [True]
            
            grid_sell = [0.40] # Sabit
            
            total_tests = len(grid_buy) * len(grid_sl) * len(grid_tp) * len(grid_sent) * len(grid_oc) * len(grid_trail_use)
            st.write(f"Tahmini Test Sayısı: {total_tests}")
            
            if st.button("⚡ Optimizasyonu Başlat (Sabır Gerekir)", type="primary"):
                from backtest.run_grid_search import run_grid_search
                
                param_grid = {
                    'buy_threshold': grid_buy,
                    'sell_threshold': grid_sell,
                    'stop_loss_pct': grid_sl,
                    'take_profit_pct': grid_tp,
                    'use_sentiment': grid_sent,
                    'use_onchain': grid_oc,
                    'use_trailing_stop': grid_trail_use,
                    'trailing_decay': grid_trail_decay
                }
                
                status_test = st.empty()
                prog_bar = st.progress(0)
                
                def update_progress(current, total, message):
                    percent = int((current / total) * 100)
                    prog_bar.progress(percent)
                    status_test.text(f"⏳ {message}")
                
                try:
                    # 365 gün train, 90 gün step (Hız için)
                    df_results = run_grid_search(
                        param_grid, 
                        train_window=365, 
                        test_window=90, 
                        use_sentiment=True, 
                        use_onchain=True,
                        progress_callback=update_progress
                    )
                    
                    status_test.text("✅ Optimizasyon Tamamlandı!")
                    st.success("En İyi Sonuçlar:")
                    st.dataframe(df_results.style.highlight_max(axis=0, subset=['Return_Pct', 'Sharpe']))
                    
                    if not df_results.empty:
                        best = df_results.iloc[0]
                        st.json({
                            "ÖNERİLEN AYARLAR": {
                                "Buy Threshold": best['Buy_Thresh'],
                                "Stop Loss": best['Stop_Loss'],
                                "Take Profit": best['Take_Profit'],
                                "Beklenen Getiri": f"%{best['Return_Pct']:.2f}"
                            }
                        })
                        
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

        elif test_type == "🔴 Yürüyen Analiz (Walk-Forward)":
            st.info("**Yürüyen Analiz (Walk-Forward):** Modelin adaptasyon yeteneğini ölçer. Geçmişten bugüne gelirken, her ay modeli **yeni verilerle yeniden eğitiriz (Re-training).** Böylece modelin 'ezberci' mi yoksa 'öğrenen' mi olduğunu anlarız.\n\n*Not: Mevcut 'Başarılı Modelinizi' bozmaz, geçici modeller eğitir.*")
            
            st.subheader("⚙️ Parametreler")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                wf_symbol = st.selectbox("Sembol", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"], index=0)
            with c2:
                wf_strat_type = st.selectbox("Strateji Tipi", ["Profesyonel (Filtreli)", "Maceracı (Risk Seven)"])
            with c3:
                wf_train_window = st.number_input("Eğitim Penceresi (Gün)", 90, 720, 365, help="Model her seferinde geçmiş kaç günü öğrensin?")
            with c4:
                wf_step = st.selectbox("Yeniden Eğitim Sıklığı", [30, 60, 90], index=0, format_func=lambda x: f"Her {x} Günde Bir")
                
            use_filt = True if wf_strat_type == "Profesyonel (Filtreli)" else False
            use_sent = st.checkbox("🧠 Sentiment (Korku & Açgözlülük) Verisini Dahil Et", value=False, help="Model piyasa duygusunu da öğrensin mi?")
            use_oc = st.checkbox("🔗 On-Chain (Zincir Üstü) Verisini Dahil Et", value=False, help="Model madenci geliri, işlem sayısı vb. ağ verilerini de öğrensin mi?")
            
            import datetime
            wf_start_date = st.date_input("Analiz Başlangıç Tarihi", datetime.date(2023, 1, 1))
            
            # Model Seçimi
            wf_model_type = st.selectbox(
                "🤖 Yapay Zeka Modeli", 
                ["XGBoost", "RandomForest", "LSTM"],
                help="Kullanılacak algoritmayı seçin. LSTM biraz yavaş çalışabilir."
            )
                
                
            # Eşik Değer Ayarları (Advanced)
            with st.expander("⚙️ Gelişmiş Ayarlar (Risk Toleransı)", expanded=False):
                c1, c2 = st.columns(2)
                wf_buy_thresh = c1.slider("Alım Eşiği (Buy Threshold)", 0.50, 0.90, 0.60, 0.05, help="Model ne kadar emin olunca alsın? Yüksek değer = Daha az ama öz işlem.")
                wf_sell_thresh = c2.slider("Satış Eşiği (Sell Threshold)", 0.10, 0.50, 0.40, 0.05, help="Model ne kadar emin olunca satsın? Düşük değer = Daha kolay sat.")
                
                c3, c4 = st.columns(2)
                wf_stop_loss = c3.slider("Stop Loss (Zarar Kes %)", 0.01, 0.20, 0.10, 0.01)
                wf_take_profit = c4.slider("Kar Al (Take Profit %)", 0.05, 0.50, 0.20, 0.05)
                
                # Trailing Stop Seçeneği
                st.markdown("---")
                use_trail = st.checkbox("🏃‍♂️ İz Süren Stop (Trailing Stop) Kullan", value=False, help="Kar belli bir seviyeye gelince satmaz, zirveden dönüşü bekler.")
                trail_decay = 0.10
                if use_trail:
                    trail_decay = st.slider("İz Süren Stop Eşiği (Trailing Decay)", 0.05, 0.30, 0.10, 0.01, help="Fiyat zirveden ne kadar düşünce satılsın?")
                    
                wf_use_dynamic = st.checkbox("🧠 Akıllı Sermaye (Güvene Göre)", value=False, help="Düşük güven varsa az para yatırır.")

            # Butonlar Yan Yana
            c_btn1, c_btn2 = st.columns([1, 1])
            start_single = c_btn1.button("🚀 Tekil Analizi Başlat", type="primary")
            start_compare = c_btn2.button("⚔️ Modelleri Yarıştır (XGB vs RF vs LSTM)", type="secondary")

            if start_compare:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Yarış Başlıyor...")
                
                # Import
                from backtest.run_walk_forward import run_walk_forward_and_save
                
                models_to_test = ["XGBoost", "RandomForest", "LSTM"]
                results_list = []
                
                for idx, model_name in enumerate(models_to_test):
                    status_text.text(f"⏳ Çalışıyor: {model_name}...")
                    
                    with st.spinner(f"{model_name} eğitiliyor ve test ediliyor..."):
                        res = run_walk_forward_and_save(
                            symbol=wf_symbol,
                            model_type=model_name,
                            train_window_days=wf_train_window,
                            test_window_days=wf_step,
                            start_date=str(wf_start_date),
                            use_trend_filter=use_filt,
                            use_sentiment=use_sent,
                            use_onchain=use_oc,
                            buy_threshold=wf_buy_thresh,
                            sell_threshold=wf_sell_thresh,
                            stop_loss_pct=wf_stop_loss,
                            take_profit_pct=wf_take_profit,
                            use_trailing_stop=use_trail,
                            trailing_decay=trail_decay,
                            use_dynamic_sizing=wf_use_dynamic
                        )
                        
                        if "error" in res:
                            st.error(f"{model_name} Hatası: {res['error']}")
                            continue
                            
                        # Metrikleri Hazırla
                        # Ortalama ML Metrikleri
                        acc_hist = pd.DataFrame(res.get('model_accuracy_history', []))
                        avg_acc = acc_hist['accuracy'].mean() if not acc_hist.empty else 0
                        avg_f1 = acc_hist.get('f1_score', pd.Series([0])).mean() if not acc_hist.empty else 0
                        avg_recall = acc_hist.get('recall', pd.Series([0])).mean() if not acc_hist.empty else 0
                        avg_auc = acc_hist.get('auc', pd.Series([0])).mean() if not acc_hist.empty else 0
                        
                        row = {
                            "Model": model_name,
                            "Final Sermaye ($)": int(res['final_equity']),
                            "Net Getiri (%)": round(res['return_pct'], 2),
                            "Al-Tut Farkı (%)": round(res['return_pct'] - res.get('bh_return', 0), 2),
                            "Sharpe": round(res['sharpe_ratio'], 2),
                            "Drawdown (%)": round(res['max_drawdown'], 2),
                            "Kar Faktörü": round(res.get('profit_factor', 0), 2),
                            "İşlem Sayısı": res['total_trades'],
                            "Ort. Accuracy": round(avg_acc, 2),
                            "Ort. F1": round(avg_f1, 2),
                            "Ort. Recall": round(avg_recall, 2),
                            "Ort. AUC": round(avg_auc, 2),
                            "P-Value": round(res.get('p_value', 1.0), 4)
                        }
                        results_list.append(row)
                    
                    progress_bar.progress((idx + 1) / len(models_to_test))
                
                status_text.text("✅ Yarış Tamamlandı!")
                if results_list:
                    st.divider()
                    st.subheader("🏆 Büyük Karşılaştırma Sonucu")
                    df_compare = pd.DataFrame(results_list)
                    st.dataframe(df_compare.style.highlight_max(axis=0, color='darkgreen'), use_container_width=True)
                    
                    # Grafiksel Tablo 1: Finansal Performans
                    import plotly.graph_objects as go
                    
                    # Kolonları İkiye Böl
                    cols_fin = ["Model", "Final Sermaye ($)", "Net Getiri (%)", "Al-Tut Farkı (%)", "Sharpe", "Drawdown (%)", "Kar Faktörü"]
                    cols_ml = ["Model", "Ort. Accuracy", "Ort. F1", "Ort. Recall", "Ort. AUC", "P-Value"]
                    
                    df_fin = df_compare[cols_fin]
                    df_ml = df_compare[cols_ml]

                    # Tablo 1
                    fig_fin = go.Figure(data=[go.Table(
                        header=dict(values=cols_fin, fill_color='paleturquoise', align='left', font=dict(size=12, color='black')),
                        cells=dict(values=[df_fin[k].tolist() for k in cols_fin], fill_color='lavender', align='left', font=dict(size=11, color='black'))
                    )])
                    fig_fin.update_layout(title="Tablo 1: Modellerin Finansal Performans Karşılaştırması", margin=dict(l=0, r=0, t=30, b=0), height=200)
                    st.plotly_chart(fig_fin, use_container_width=True)

                    # Tablo 2
                    fig_ml = go.Figure(data=[go.Table(
                        header=dict(values=cols_ml, fill_color='peachpuff', align='left', font=dict(size=12, color='black')),
                        cells=dict(values=[df_ml[k].tolist() for k in cols_ml], fill_color='papayawhip', align='left', font=dict(size=11, color='black'))
                    )])
                    fig_ml.update_layout(title="Tablo 2: Modellerin Yapay Zeka ve İstatistik Performansı", margin=dict(l=0, r=0, t=30, b=0), height=200)
                    st.plotly_chart(fig_ml, use_container_width=True)

                    st.success("👆 Tablolar bölündü! Artık A4 kağıdına rahatça sığacaktır.")

            if start_single:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Analiz Hazırlanıyor...")
                    # Wrapper fonksiyonu import et (veya dosyadaki yeni fonksiyonu kullan)
                    from backtest.run_walk_forward import run_walk_forward_and_save
                    
                    with st.spinner(f"{wf_symbol} için model zaman yolculuğuna çıktı... Her adımda yeniden eğitiliyor..."):
                        wf_res = run_walk_forward_and_save(
                            symbol=wf_symbol,
                            model_type=wf_model_type,
                            train_window_days=wf_train_window,
                            test_window_days=wf_step,
                            start_date=str(wf_start_date),
                            use_trend_filter=use_filt,
                            use_sentiment=use_sent,
                            use_onchain=use_oc,
                            buy_threshold=wf_buy_thresh,
                            sell_threshold=wf_sell_thresh,
                            stop_loss_pct=wf_stop_loss,
                            take_profit_pct=wf_take_profit,
                            use_trailing_stop=use_trail,
                            trailing_decay=trail_decay,
                            use_dynamic_sizing=wf_use_dynamic
                        )
                    
                    if "error" in wf_res:
                        st.error(wf_res["error"])
                    else:
                        progress_bar.progress(100)
                        st.success("✅ Analiz Tamamlandı!")
                        
                        # Sonuç özet (Artık 6 Kolon: Profit Factor ekliyoruz)
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        
                        bh_return = wf_res.get('bh_return', 0.0)
                        strat_return = wf_res['return_pct']
                        diff_return = strat_return - bh_return
                        
                        # Profit Factor (Eski versiyonlarda yoksa 0 ata)
                        profit_factor = wf_res.get('profit_factor', 0.0)

                        c1.metric("Final Sermaye", f"${wf_res['final_equity']:,.0f}", f"Net: %{strat_return:.1f}")
                        c2.metric("Al-Tut Farkı", f"%{diff_return:.1f}", delta=f"{diff_return:.1f}%")
                        c3.metric("Sharpe Oranı", f"{wf_res['sharpe_ratio']:.2f}")
                        c4.metric("Max Drawdown", f"%{wf_res['max_drawdown']:.2f}")
                        c5.metric("Kar Faktörü", f"{profit_factor:.2f}") 
                        c6.metric("İşlem Sayısı", wf_res['total_trades'])

                        # İstatistiksel Anlamlılık (P-Value & Confidence Interval)
                        st.divider()
                        st.subheader("🧪 İstatistiksel Testler (Hypothesis Testing)")
                        
                        c_stat1, c_stat2, c_stat3 = st.columns(3)
                        
                        pval = wf_res.get('p_value', 1.0)
                        sharpe_pval = wf_res.get('sharpe_p_value', 1.0)
                        conf_low = wf_res.get('conf_interval_low', 0.0)
                        conf_high = wf_res.get('conf_interval_high', 0.0)
                        
                        # 1. T-Test (Returns)
                        if pval < 0.05:
                            c_stat1.success(f"**Getiri T-Testi:**\nP={pval:.4f} (Anlamlı)")
                        else:
                            c_stat1.warning(f"**Getiri T-Testi:**\nP={pval:.4f} (>0.05)")
                            
                        # 2. Bootstrap (Sharpe)
                        if sharpe_pval < 0.05:
                            c_stat2.success(f"**Sharpe Bootstrap:**\nP={sharpe_pval:.4f} (Anlamlı!)")
                        else:
                            c_stat2.warning(f"**Sharpe Bootstrap:**\nP={sharpe_pval:.4f} (>0.05)")

                        # 3. Conf Interval
                        c_stat3.info(f"**%95 Güven Aralığı:**\nAralık: [{conf_low:.4f}, {conf_high:.4f}]")
                        
                        st.caption("*Not: Getiri T-Testi piyasa korelasyonunu (aynı yönü), Bootstrap testi ise Risk/Getiri kalitesini (kalite farkını) ölçer.*")
                        
                        # Grafik 1: Equity Curve
                        st.divider()
                        st.subheader("📈 Gerçekçi (Adapte Olan) Performans Eğrisi")
                        
                        eq_data = pd.DataFrame.from_dict(wf_res['equity_curve'], orient='index', columns=['Strateji'])
                        if 'bh_equity_curve' in wf_res:
                            bh_data = pd.DataFrame.from_dict(wf_res['bh_equity_curve'], orient='index', columns=['Buy & Hold'])
                            eq_data = pd.concat([eq_data, bh_data], axis=1)
                        
                        st.line_chart(eq_data)
                        
                        # Grafik 1.5: İşlem Yerleri (Fiyat Grafiği)
                        if 'price_data' in wf_res and 'logs' in wf_res:
                            st.subheader("📍 İşlem Noktaları (Boncuklar)")
                            price_series = pd.Series(wf_res['price_data'])
                            price_df = pd.DataFrame({'Close': price_series})
                            price_df.index = pd.to_datetime(price_df.index)
                            price_df = price_df.sort_index()
                            
                            import plotly.graph_objects as go
                            fig_trade = go.Figure()
                            
                            # Fiyat Çizgisi
                            fig_trade.add_trace(go.Scatter(
                                x=price_df.index, 
                                y=price_df['Close'], 
                                mode='lines', 
                                name='BTC Fiyatı',
                                line=dict(color='gray', width=1)
                            ))
                            
                            # İşlemleri Parse Et
                            buy_dates = []
                            buy_prices = []
                            buy_reasons = [] # HOVER TEXT İÇİN
                            
                            sell_dates = []
                            sell_prices = []
                            sell_reasons = [] # HOVER TEXT İÇİN
                            
                            for log in wf_res['logs']:
                                # log format: {'Date': '...', 'Action': '...', 'Price': ...}
                                # Action bazen "AL (Score: 0.65)" şeklinde olabilir, startswith kullanalım.
                                act = log.get('Action', '').upper()
                                date_str = log.get('Date')
                                if not date_str: continue
                                
                                # Emoji olduğu için startswith çalışmayabilir, IN kullanalım
                                if "ALIM" in act or "BUY" in act:
                                    buy_dates.append(date_str)
                                    buy_prices.append(log.get('Price'))
                                    buy_reasons.append(log.get('Reason', 'Nedeni Bilinmiyor'))
                                elif "SATIŞ" in act or "SELL" in act or "SHORT" in act:
                                    sell_dates.append(date_str)
                                    sell_prices.append(log.get('Price'))
                                    sell_reasons.append(log.get('Reason', 'Nedeni Bilinmiyor'))
                                    
                            # Alım Boncukları (Yeşil Üçgen)
                            fig_trade.add_trace(go.Scatter(
                                x=buy_dates, 
                                y=buy_prices, 
                                mode='markers', 
                                name='Alım',
                                text=buy_reasons, # HOVER BURADA
                                hoverinfo='text+y+x', # Sadece metin, tarih, fiyat göster
                                marker=dict(symbol='triangle-up', size=12, color='#00CC96')
                            ))
                            
                            # Satım Boncukları (Kırmızı Üçgen)
                            fig_trade.add_trace(go.Scatter(
                                x=sell_dates, 
                                y=sell_prices, 
                                mode='markers', 
                                name='Satım',
                                text=sell_reasons, # HOVER BURADA
                                hoverinfo='text+y+x',
                                marker=dict(symbol='triangle-down', size=12, color='#EF553B')
                            ))
                            
                            fig_trade.update_layout(title="Alım-Satım Noktaları (Üzerine Gel)", hovermode="closest")
                            st.plotly_chart(fig_trade, use_container_width=True)

                        
                        # Grafik 2: Modelin Başarım Tarihçesi (Accuracy, F1, vb.)
                        st.subheader("🤖 Model Zekası (Detaylı Anlık Performans)")
                        acc_hist = pd.DataFrame(wf_res['model_accuracy_history'])
                        if not acc_hist.empty:
                            acc_hist['period'] = pd.to_datetime(acc_hist['period'])
                            acc_hist.set_index('period', inplace=True)
                            
                            # Ortalama Metrikleri Göster
                            m1, m2, m3, m4 = st.columns(4)
                            avg_acc = acc_hist['accuracy'].mean()
                            avg_f1 = acc_hist.get('f1_score', pd.Series([0])).mean()
                            avg_recall = acc_hist.get('recall', pd.Series([0])).mean()
                            avg_auc = acc_hist.get('auc', pd.Series([0])).mean()
                            
                            m1.metric("Ort. Accuracy", f"%{avg_acc*100:.1f}")
                            m2.metric("Ort. F1 Score", f"{avg_f1:.2f}")
                            m3.metric("Ort. Recall", f"{avg_recall:.2f}")
                            m4.metric("Ort. ROC AUC", f"{avg_auc:.2f}")
                            
                            st.caption("Not: ROC AUC 0.5 çıkarsa o dönemde model hep tek yöne (sadece artış/azalış) tahmin yapmış olabilir.")
                            # Chart'ta da F1 skorunu gösterelim
                            if 'f1_score' in acc_hist.columns:
                                st.line_chart(acc_hist[['accuracy', 'f1_score']] * 100)
                            else:
                                st.line_chart(acc_hist['accuracy'] * 100)
                        else:
                            st.warning("Yeterli doğruluk verisi toplanamadı.")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            st.stop()
            
        elif robust_type == "Gürültü Testi (Noise Test)":
            st.write("Bu bölüm, modelin gürültülü veriye karşı ne kadar dayanıklı olduğunu test eder.")
            noise_level = st.slider("Gürültü Seviyesi (%)", 0.0, 5.0, 1.0, 0.1)
            if st.button("Gürültü Testini Başlat"):
                 st.info("Bu özellik demo aşamasındadır.")
            st.stop()

    # Rapor Görüntüleme Modu
    else: # rb_mode == "🔍 Geçmiş Raporları İncele"
        df_display = df[df['is_robustness'] == True].sort_values(by='date', ascending=False)
        if df_display.empty:
            st.sidebar.warning("Henüz sağlamlık testi raporu yok.")
        else:
            selected_filename = st.sidebar.radio(
                "Test Seçiniz:",
                df_display['json_filename'].tolist(),
                format_func=lambda x: f"{df_display[df_display['json_filename']==x]['date'].values[0]} | {df_display[df_display['json_filename']==x]['strategy'].values[0] if 'strategy' in df_display.columns else 'Robustness'}"
            )
            st.sidebar.caption(f"Dosya: {selected_filename}")

else: # Model Sonuçları
    # Monte Carlo ve Robustness olmayanlar
    df_models = df[(df['is_robustness'] == False) & (df['is_monte_carlo'] == False)]
    
    # Strateji Filtresi
    strategies = ["Tümü"] + list(df_models['strategy'].unique())
    sel_strat = st.sidebar.selectbox("Strateji Filtrele:", strategies)
    
    if sel_strat != "Tümü":
        df_display = df_models[df_models['strategy'] == sel_strat]
    else:
        df_display = df_models
        
    if df_display.empty:
        st.sidebar.warning("Bu filtreye uygun rapor yok.")
    else:
        selected_filename = st.sidebar.radio(
            "Rapor Seçiniz:",
            df_display['json_filename'].tolist(),
            format_func=lambda x: f"{df_display[df_display['json_filename']==x]['date'].values[0]} | {df_display[df_display['json_filename']==x]['strategy'].values[0]}"
        )

# Seçim yapılmadıysa dur
if not selected_filename:
    st.info("Lütfen sol menüden bir rapor seçin.")
    st.stop()

# Seçili Rapor Detayları
run_data = df[df['json_filename'] == selected_filename].iloc[0]

# --- MONTE CARLO RAPORU KONTROLÜ ---
if run_data.get('is_monte_carlo'):
    st.markdown("### 🎲 Monte Carlo Simülasyon Sonuçları")
    st.info(f"**Açıklama:** {run_data.get('description', '')}")
    
    # Özet Metrikler
    # pd.json_normalize genellikle her şeyi düzleştirir (simulation_results.mean_equity gibi)
    # Bu yüzden önce düzleşmiş sütunlara bakalım.
    
    mean_eq = run_data.get('simulation_results.mean_equity')
    if pd.isna(mean_eq): # Eğer düz sütun yoksa veya NaN ise, belki nested dict vardır
        s_res = run_data.get('simulation_results', {})
        if isinstance(s_res, dict):
            mean_eq = s_res.get('mean_equity', 0)
            p05_eq = s_res.get('p05_equity', 0)
            risk_50 = s_res.get('risk_of_ruin_50pct', 0)
            risk_90 = s_res.get('risk_of_ruin_90pct', 0)
        else:
            mean_eq = 0
            p05_eq = 0
            risk_50 = 0
            risk_90 = 0
    else:
        # Düzleşmiş sütunlardan al
        mean_eq = run_data.get('simulation_results.mean_equity', 0)
        p05_eq = run_data.get('simulation_results.p05_equity', 0)
        risk_50 = run_data.get('simulation_results.risk_of_ruin_50pct', 0)
        risk_90 = run_data.get('simulation_results.risk_of_ruin_90pct', 0)
    
    # Ek Metrikler (ROI, CAGR, Süre)
    # Flattened kontrolü
    sim_duration = run_data.get('simulation_meta.simulated_duration_years')
    if pd.isna(sim_duration):
        meta = run_data.get('simulation_meta', {})
        if isinstance(meta, dict):
            sim_duration = meta.get('simulated_duration_years', '?')
            roi = meta.get('mean_roi_pct', 0)
            cagr = meta.get('cagr_pct', 0)
        else:
            sim_duration = '?'
            roi = 0
            cagr = 0
    else:
        # Düzleşmiş veriden al
        roi = run_data.get('simulation_meta.mean_roi_pct', 0)
        cagr = run_data.get('simulation_meta.cagr_pct', 0)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ortalama Sermaye Beklentisi", f"${mean_eq:,.0f}", delta=f"ROI: %{roi:,.0f}")
    with c2:
        st.metric("En Kötü Senaryo (%5)", f"${p05_eq:,.0f}", delta="Risk Altında", delta_color="inverse")
    with c3:
        st.metric("Batış Riski (%50 Kayıp)", f"%{risk_50:.2f}", delta="-Risk" if risk_50 > 0 else "Güvenli")
    with c4:
        st.metric("Tahmini Süre", f"{sim_duration} Yıl", help="Simüle edilen 150 işlemin ortalama gerçekleşme süresi.")
        
    st.caption(f"📈 **Yıllık Bileşik Getiri (CAGR):** %{cagr:.2f} | **Başlangıç:** $10,000 | **İşlem Sıklığı:** Her ~{round((1.2*365)/150) if sim_duration != '?' else '?'} günde bir işlem")

    # --- ML METRİKLERİ (Kayıtlı Rapordan) ---
    mm = run_data.get('model_metrics', {})
    # Nested dict değilse (flatten edilmişse)
    if not mm and 'model_metrics.accuracy' in run_data:
        mm = {
            'accuracy': run_data.get('model_metrics.accuracy'),
            'precision': run_data.get('model_metrics.precision'),
            'recall': run_data.get('model_metrics.recall'),
            'f1': run_data.get('model_metrics.f1')
        }
    
    if mm:
        st.divider()
        st.subheader("🤖 Model Performansı (Tüm Dönem)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Doğruluk (Acc)", f"%{mm.get('accuracy', 0)*100:.1f}")
        m2.metric("Keskinlik (Prec)", f"%{mm.get('precision', 0)*100:.1f}")
        m3.metric("Duyarlılık (Rec)", f"%{mm.get('recall', 0)*100:.1f}")
        m4.metric("F1 Skoru", f"%{mm.get('f1', 0)*100:.1f}")

    # Histogram (Dağılım)
    st.subheader("📊 Olası Sonuç Dağılımı")
    
    # Histogram verisini al (data_samples.final_equities)
    # Json normalize yüzünden data_samples.final_equities şeklinde olabilir veya dict içinde
    dist_data = run_data.get('data_samples.final_equities')
    if not isinstance(dist_data, list):
        # Belki 'data_samples' dict olarak duruyordur
        ds = run_data.get('data_samples')
        if isinstance(ds, dict):
            dist_data = ds.get('final_equities')
    
    if dist_data:
        import plotly.express as px
        fig = px.histogram(x=dist_data, nbins=50, title="Simüle Edilmiş 1000 Portföyün Son Değerleri",
                           labels={'x': 'Portföy Değeri ($)', 'y': 'Frekans'},
                           color_discrete_sequence=['#00CC96'])
        
        fig.add_vline(x=10000, line_dash="dash", line_color="white", annotation_text="Başlangıç ($10k)")
        fig.add_vline(x=p05_eq, line_dash="dot", line_color="red", annotation_text="Kötü Senaryo")
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dağılım verisi bulunamadı.")
        
    st.stop() # Monte Carlo ise burada bitir.

# --- ROBUSTNESS (SAĞLAMLIK) RAPORU KONTROLÜ ---
# Eğer 'results' alanı varsa bu bir Robustness raporudur
if 'results' in run_data and isinstance(run_data['results'], list):
    st.markdown("### 🛡️ Sağlamlık Testi (Robustness Check) Sonuçları")
    st.info(f"**Açıklama:** {run_data.get('description', '')}")
    
    # Listeyi DataFrame'e çevir
    df_res = pd.DataFrame(run_data['results'])
    
    # Kolon İsimlerini Düzenle
    df_res.rename(columns={
        'period': 'Dönem',
        'return': 'Getiri (%)',
        'buy_hold': 'Al-Tut (%)',
        'max_drawdown': 'Max. Drawdown (%)',
        'trades': 'İşlem Sayısı',
        'win_rate': 'Kazanma Oranı (%)',
        'sharpe': 'Sharpe Oranı',
        'start_date': 'Başlangıç',
        'end_date': 'Bitiş'
    }, inplace=True)
    
    # Tabloyu Göster
    st.dataframe(
        df_res.style.format({
            'Getiri (%)': '{:.2f}',
            'Al-Tut (%)': '{:.2f}',
            'Max. Drawdown (%)': '{:.2f}',
            'Kazanma Oranı (%)': '{:.2f}',
            'Sharpe Oranı': '{:.2f}'
        }).background_gradient(subset=['Getiri (%)'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Grafiksel Karşılaştırma
    st.subheader("📊 Dönemlere Göre Getiri Karşılaştırması")
    import plotly.express as px
    
    # Veriyi Long Format'a çevir (Bar chart için)
    df_melt = df_res.melt(id_vars=['Dönem'], value_vars=['Getiri (%)', 'Al-Tut (%)'], var_name='Strateji', value_name='Getiri')
    
    fig = px.bar(df_melt, x='Dönem', y='Getiri', color='Strateji', barmode='group',
                 color_discrete_map={'Getiri (%)': '#00FF7F', 'Al-Tut (%)': '#FFA500'})
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.stop() # Robustness raporuysa aşağıdakileri (standart metrikleri) gösterme ve burada dur.

# --- ÜST METRİKLER (Standart Raporlar İçin) ---
st.markdown("### 📊 Performans Özeti")
col1, col2, col3, col4 = st.columns(4)

initial_capital = run_data.get('initial_capital', 1000000) # Varsayılan 1M
# json_normalize sonrası metrics.final_equity column adına dikkat
final_capital = run_data.get('metrics.final_equity', 0)
profit = final_capital - initial_capital

metrics_return = run_data.get('metrics.return', 0)
metrics_buy_hold = run_data.get('metrics.buy_hold_return', 0)

with col1:
    st.metric(
        label="Toplam Getiri", 
        value=f"%{metrics_return:.2f}", 
        delta=f"{metrics_return - metrics_buy_hold:.2f}% vs Al-Tut"
    )
with col2:
    st.metric(
        label="Net Kâr/Zarar", 
        value=f"${profit:,.0f}",
        delta=f"{((final_capital/initial_capital)-1)*100:.1f}%"
    )
with col3:
    st.metric("Başlangıç Sermayesi", f"${initial_capital:,.0f}")
with col4:
    st.metric("Son Sermaye", f"${final_capital:,.0f}")

# --- STRATEJİ DETAYI ---
st.info(f"**Strateji Açıklaması:** {run_data.get('description', 'Açıklama bulunamadı.')}")

st.divider()

# --- SERMAYE GRAFİĞİ (Modern & Plotly) ---
st.subheader("📈 Sermaye Büyüme Grafiği")

# Flatten sonrası equity_curve.dates ve equity_curve.equity sütunları oluşur
if 'equity_curve.dates' in run_data and isinstance(run_data['equity_curve.dates'], list):
    try:
        import plotly.graph_objects as go
        
        dates = run_data['equity_curve.dates']
        equities = run_data.get('equity_curve.equity', [])
        
        # Veri Temizliği: NaN değerleri None ile değiştir (Plotly için)
        # JSON'dan gelen 'nan' stringleri veya float('nan') olabilir.
        if isinstance(equities, list):
            import math
            cleaned_equities = []
            for e in equities:
                if isinstance(e, str) and e.lower() == 'nan':
                    cleaned_equities.append(None)
                elif isinstance(e, float) and math.isnan(e):
                    cleaned_equities.append(None)
                else:
                    cleaned_equities.append(e)
            equities = cleaned_equities
        else:
            equities = []

        # Eğer veri varsa çiz
        if dates and equities:
            fig = go.Figure()
            
            # Sermaye Çizgisi
            fig.add_trace(go.Scatter(
                x=dates, 
                y=equities, 
                mode='lines', 
                name='Strateji Sermayesi',
                line=dict(color='#00FF7F', width=2),
                fill='tozeroy', # Altını doldur
                fillcolor='rgba(0, 255, 127, 0.1)' # Hafif yeşil dolgu
            ))
            
            # Al-ve-Tut Çizgisi (Varsa)
            buy_hold_values = run_data.get('equity_curve.buy_hold')
            
            # Veri Temizliği: NaN değerleri None ile değiştir
            if isinstance(buy_hold_values, list):
                cleaned_bh = []
                for v in buy_hold_values:
                    if isinstance(v, str) and v.lower() == 'nan':
                        cleaned_bh.append(None)
                    elif isinstance(v, float) and math.isnan(v):
                        cleaned_bh.append(None)
                    else:
                        cleaned_bh.append(v)
                buy_hold_values = cleaned_bh
            else:
                buy_hold_values = []

            if buy_hold_values:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=buy_hold_values,
                    mode='lines',
                    name='Al-ve-Tut (Buy & Hold)',
                    line=dict(color='#FFA500', width=2, dash='dash'), # Turuncu kesikli çizgi
                    opacity=0.8
                ))
            
            # Başlangıç Çizgisi
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="white", annotation_text="Başlangıç")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title="Zaman İçindeki Portföy Değeri",
                xaxis_title="Tarih",
                yaxis_title="Dolar ($)",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Grafik verisi eksik veya bozuk.")
    except ImportError:
        st.error("Plotly kütüphanesi eksik.")
        
    # --- AYLIK GETİRİ ISI HARİTASI (HEATMAP) ---
    st.subheader("🗓️ Aylık Getiri Isı Haritası")
    if dates and equities:
        try:
            # Pandas Serisine çevir
            df_eq = pd.DataFrame({'Date': pd.to_datetime(dates), 'Equity': equities})
            df_eq.set_index('Date', inplace=True)
            
            # Günlük Getiri yerine Aylık örnekleme (Resample)
            # Her ayın son bakiyesi
            monthly_equity = df_eq['Equity'].resample('M').last()
            
            # Her ayın getirisi = (Bu Ay Sonu - Geçen Ay Sonu) / Geçen Ay Sonu
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Yüzdeye çevir
            monthly_returns_pct = monthly_returns * 100
            
            # Pivot Tablo (Yıl x Ay)
            heatmap_data = pd.DataFrame({
                'Year': monthly_returns_pct.index.year,
                'Month': monthly_returns_pct.index.month,
                'Return': monthly_returns_pct.values
            })
            
            pivot_table = heatmap_data.pivot(index='Year', columns='Month', values='Return')
            
            # Eksik aylar için None doldurabiliriz ama heatmap otomatik halleder
            # Ay isimleri
            month_names = {1:'Oca', 2:'Şub', 3:'Mar', 4:'Nis', 5:'May', 6:'Haz', 
                           7:'Tem', 8:'Ağu', 9:'Eyl', 10:'Eki', 11:'Kas', 12:'Ara'}
            
            x_labels = [month_names[i] for i in range(1, 13)]
            y_labels = list(pivot_table.index)
            
            # Z değerlerini matrise çevir (Her yıl için 12 ayın değerleri)
            z_values = []
            text_values = []
            
            for year in y_labels:
                row = []
                txt_row = []
                for month in range(1, 13):
                    if month in pivot_table.columns and year in pivot_table.index:
                        val = pivot_table.loc[year, month]
                        if pd.notna(val):
                            row.append(val)
                            txt_row.append(f"%{val:.1f}")
                        else:
                            row.append(None)
                            txt_row.append("")
                    else:
                        row.append(None)
                        txt_row.append("")
                z_values.append(row)
                text_values.append(txt_row)
                
            # Heatmap Çiz
            import plotly.graph_objects as go
            
            fig_hm = go.Figure(data=go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                text=text_values,
                texttemplate="%{text}",
                colorscale='RdYlGn', # Kırmızı (Zarar) -> Sarı -> Yeşil (Kâr)
                zmid=0, # 0 noktası nötr renk olsun
                showscale=True,
                xgap=2, # Kutular arası boşluk
                ygap=2
            ))
            
            fig_hm.update_layout(
                title="Aylık Performans (%)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Ay",
                yaxis_title="Yıl",
                height=300
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            
        except Exception as e:
            st.error(f"Heatmap oluşturulurken hata: {e}")

else:
    st.info("Bu test için detaylı grafik verisi kaydedilmemiş (Eski versiyon olabilir).")

# --- ML MODEL PERFORMANSI ---
# --- ML MODEL PERFORMANSI ---
# --- ML MODEL PERFORMANSI ---
# Hem 'model_metrics' (dict) hem de 'model_metrics.accuracy' (flat) ihtimallerini kontrol edelim
has_metrics = False
metrics_data = {}

# Durum 1: Flatten edilmiş (model_metrics.accuracy sütunu var ve boş değil)
if 'model_metrics.accuracy' in run_data and pd.notna(run_data['model_metrics.accuracy']):
    has_metrics = True
    metrics_data = {
        'accuracy': run_data.get('model_metrics.accuracy'),
        'precision': run_data.get('model_metrics.precision'),
        'recall': run_data.get('model_metrics.recall'),
        'f1': run_data.get('model_metrics.f1'),
        'model_name': run_data.get('model_metrics.model_name', 'Bilinmiyor'),
        'test_period_start': run_data.get('model_metrics.test_period_start', '?'),
        'test_period_end': run_data.get('model_metrics.test_period_end', '?')
    }

# Durum 2: Nested Dict (model_metrics sütunu var ve içinde dict var)
elif 'model_metrics' in run_data and isinstance(run_data['model_metrics'], dict) and run_data['model_metrics']:
    has_metrics = True
    m = run_data['model_metrics']
    metrics_data = {
        'accuracy': m.get('accuracy'),
        'precision': m.get('precision'),
        'recall': m.get('recall'),
        'f1': m.get('f1'),
        'model_name': m.get('model_name', 'Bilinmiyor'),
        'test_period_start': m.get('test_period_start', '?'),
        'test_period_end': m.get('test_period_end', '?')
    }

if has_metrics:
    st.divider()
    st.subheader("🤖 Makine Öğrenmesi (ML) Model Performansı")
    
    try:
        m1, m2, m3, m4 = st.columns(4)
        
        acc = metrics_data.get('accuracy', 0)
        prec = metrics_data.get('precision', 0)
        rec = metrics_data.get('recall', 0)
        f1 = metrics_data.get('f1', 0)
        
        with m1:
            st.metric("Model Doğruluğu (Accuracy)", f"%{acc*100:.2f}")
        with m2:
            st.metric("Keskinlik (Precision)", f"%{prec*100:.2f}", help="Model 'Yükselecek' dediğinde ne kadar haklı?")
        with m3:
            st.metric("Duyarlılık (Recall)", f"%{rec*100:.2f}", help="Gerçek yükselişlerin ne kadarını yakaladık?")
        with m4:
            st.metric("F1 Skoru", f"%{f1*100:.2f}")
            
        # Süre Hesaplama
        start_date_str = metrics_data.get('test_period_start', '?')
        end_date_str = metrics_data.get('test_period_end', '?')
        duration_str = "Bilinmiyor"
        
        try:
            if start_date_str != '?' and end_date_str != '?':
                sd = pd.to_datetime(start_date_str)
                ed = pd.to_datetime(end_date_str)
                diff = ed - sd
                days = diff.days
                months = days // 30
                duration_str = f"{days} Gün (~{months} Ay)"
                
        except:
            pass
            
        st.caption(f"📅 **Test Süresi:** {start_date_str} - {end_date_str} ({duration_str}) | Model: {metrics_data['model_name']}")
        
        # --- DETAYLI MODEL & STRATEJİ KARTI ---
        st.markdown("---")
        st.subheader("🧠 Strateji ve Model Detayları")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.info("🎯 **Strateji Mantığı**")
            
            strat_name = run_data.get('strategy', '')
            
            if 'XGBoost' in strat_name and 'OnChain' not in strat_name:
                st.markdown("""
                - **Model:** 🚀 XGBoost (Teknik)
                - **Özellikler:** Hafıza (Lag) + Volatilite + Teknik.
                - **Güven Eşiği:** Model **%60+** eminse işlem açar.
                - **Yön:** 🟢 Long ve 🔴 Short.
                """)
            elif 'OnChain_XGBoost' in strat_name:
                st.markdown("""
                - **Model:** 🔗 XGBoost + On-Chain (Stanford)
                - **Ek Veriler:** Hash Rate, Difficulty, Active Address vb.
                - **Amaç:** Ağ sağlığını (Network Health) fiyata yansıtmak.
                - **Not:** Akademik makaleden esinlenildi.
                """)
            elif 'RandomForest_V2' in strat_name or 'Advanced' in strat_name:
                st.markdown("""
                - **Model:** 🌲 Random Forest V2 (Gelişmiş)
                - **Özellikler:** Hafıza (Lag) + Volatilite eklendi.
                - **Güven Eşiği:** %60. Stop Loss %5.
                - **Farkı:** Eski versiyona göre daha akıllı ama XGBoost kadar hızlı değil.
                """)
            elif 'RandomForest' in strat_name:
                st.markdown("""
                - **Model:** 🌲 Random Forest (Temel Versiyon)
                - **Risk:** Basit Al/Sat sinyali. Sadece teknik indikatörler.
                - **Güven Eşiği:** %60. Stop Loss %5.
                - **Farkı:** Eski versiyona göre daha akıllı ama XGBoost kadar hızlı değil.
                """)
            elif 'SmaCross' in strat_name:
                st.markdown("""
                - **Tip:** Teknik Analiz (Trend Takibi)
                - **Kural:** Kısa vade (SMA 10/50), Uzun vadeyi (SMA 20/200) yukarı keserse AL.
                - **Risk:** Ters kesişim olana kadar TUT.
                """)
            elif 'Rsi' in strat_name:
                st.markdown("""
                - **Tip:** Teknik Analiz (Momentum)
                - **Kural:** RSI < 30 (Aşırı Satım) ise AL, RSI > 70 ise SAT.
                """)
            else:
                # Bilinmeyen strateji ise JSON'daki açıklamayı göster
                desc = run_data.get('description', 'Detay bulunamadı.')
                st.markdown(f"- **Açıklama:** {desc}")
            
        with c2:
            st.success("📊 **Modelin İncelediği Veriler (Features)**")
            
            # Özellikleri kategorize et
            features_list = run_data.get('model_metrics', {}).get('features', [])
            if not features_list:
                # Flattened yapıda olabilir
                features_list = run_data.get('model_metrics.features', [])
                
            if isinstance(features_list, str):
                 # Bazen string olarak gelebilir, parse et
                 import ast
                 try: features_list = ast.literal_eval(features_list)
                 except: pass

            if features_list:
                # Kategoriler
                cats = {
                    "On-Chain (Temel)": [f for f in features_list if "Hash" in f or "Difficulty" in f or "Miner" in f or "Address" in f or "Transaction" in f],
                    "Momentum": [f for f in features_list if "RSI" in f or "ROC" in f and "Hash" not in f], # Hash_Rate_ROC buraya girmesin
                    "Trend": [f for f in features_list if "SMA" in f or "MACD" in f],
                    "Volatilite": [f for f in features_list if "ATR" in f or "BB" in f or "Volatility" in f],
                    "Hafıza (Lag)": [f for f in features_list if "Lag" in f],
                    "Diğer": [f for f in features_list if "Return" in f and "Lag" not in f]
                }
                
                # Sadece dolu kategorileri göster
                for cat_name, feats in cats.items():
                    if feats:
                        st.markdown(f"**{cat_name}:** `{', '.join(feats)}`")
            else:
                st.warning("Model özellikleri (features) verisi bulunamadı.")

    except Exception as e:
        st.error(f"ML metrikleri gösterilirken hata: {e}")

st.divider()

# --- DETAYLI HTML RAPOR (Opsiyonel) ---
with st.expander("🔍 Detaylı Etkileşimli Raporu Görüntüle"):
    html_filename = run_data.get('files.html')
    # NaN kontrolü: float('nan') True döner, bu yüzden type check şart
    if html_filename and isinstance(html_filename, str):
        html_file = os.path.join(REPORT_DIR, html_filename)
        
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
                components.html(html_content, height=800, scrolling=True)
        else:
            st.error("HTML Rapor dosyası bulunamadı.")
    else:
        st.info("Bu rapor için HTML dosyası mevcut değil.")

# --- HAM VERİ ---
with st.expander("Ham Veri (JSON)"):
    st.json(run_data.to_dict())

# --------------------------------------------------------------------------
# YENİ ÖZELLİK: SENTIMENT vs FİYAT GRAFİĞİ
# --------------------------------------------------------------------------

st.markdown("---")
st.header("🧠 Piyasa Psikolojisi: Sentiment vs Fiyat")

sentiment_path = "data/dashboard_sentiment.csv"

if os.path.exists(sentiment_path):
    try:
        sent_df = pd.read_csv(sentiment_path)
        # Tarihi datetime yap
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
        
        # 3 Aylık veri varsayılan olsun
        lookback = st.slider("Geriye Dönük Gün Sayısı:", min_value=30, max_value=1800, value=365)
        
        chart_df = sent_df.tail(lookback)
        
        # Dual Axis Chart
        fig_sent = go.Figure()

        # 1. Eksen: Fiyat (Çizgi)
        fig_sent.add_trace(go.Scatter(
            x=chart_df['Date'],
            y=chart_df['Close'],
            name="Bitcoin Fiyatı ($)",
            line=dict(color='white', width=2)
        ))

        # 2. Eksen: Sentiment (Bar)
        # Renkleri belirle: <20 Kırmızı (Korku), >80 Yeşil (Açgözlülük), Arası Gri
        colors = ['#FF4136' if v <= 20 else '#2ECC40' if v >= 80 else '#808080' for v in chart_df['FNG_Value']]
        
        fig_sent.add_trace(go.Bar(
            x=chart_df['Date'],
            y=chart_df['FNG_Value'],
            name="Fear & Greed Index",
            yaxis="y2",
            marker_color=colors,
            opacity=0.3
        ))

        # Layout Ayarları
        fig_sent.update_layout(
            title="Fiyat Hareketleri ve Yatırımcı Duygusu",
            xaxis_title="Tarih",
            yaxis=dict(title="Fiyat ($)"),
            yaxis2=dict(
                title="Fear & Greed (0-100)",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(x=0, y=1.2, orientation="h"),
            height=500
        )
        
        st.plotly_chart(fig_sent, use_container_width=True)
        
        st.info("""
        **💡 Nasıl Okunmalı?**
        - **Kırmızı Barlar (<20):** Aşırı Korku. Genellikle piyasanın dip yaptığı ve **ALIM FIRSATI** verdiği yerlerdir.
        - **Yeşil Barlar (>80):** Aşırı Açgözlülük. Genellikle piyasanın tepe yaptığı ve **SATIŞ/DÜZELTME** gelebileceği yerlerdir.
        - **Gri:** Nötr bölge.
        """)
        
    except Exception as e:
        st.error(f"Sentiment grafiği oluşturulurken hata: {e}")

else:
    st.warning("Sentiment verisi bulunamadı. Lütfen `python data/prepare_dashboard_data.py` komutunu çalıştırın.")
