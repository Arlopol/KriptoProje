import streamlit as st
import pandas as pd
import json
import os
import glob
import streamlit.components.v1 as components

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
category = st.sidebar.radio("Kategori:", ["📈 Model Sonuçları", "🛡️ Sağlamlık Testleri", "🎲 Simülasyon Testleri"])

selected_filename = None

if category == "🎲 Simülasyon Testleri":
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
    df_display = df[df['is_robustness'] == True]
    if df_display.empty:
        st.sidebar.warning("Henüz sağlamlık testi raporu yok.")
    else:
        selected_filename = st.sidebar.radio(
            "Test Seçiniz:",
            df_display['json_filename'].tolist(),
            format_func=lambda x: f"{df_display[df_display['json_filename']==x]['date'].values[0]} | {df_display[df_display['json_filename']==x]['strategy'].values[0]}"
        )

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
