import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Dashboard F&B Holistik", 
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# FUNGSI-FUNGSI UTAMA
# ==============================================================================

@st.cache_data
def load_raw_data(file):
    """Memuat data mentah dari file yang diunggah tanpa modifikasi."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
        return df.copy()
    except Exception as e:
        raise ValueError(f"Gagal memuat file: {e}")

def reset_processing_state():
    """Callback untuk mereset state jika file diubah, memastikan UI tetap sinkron."""
    st.session_state.data_processed = False
    for key in list(st.session_state.keys()):
        if key.startswith("map_"):
            del st.session_state[key]

def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """Menganalisis tren dengan wawasan bisnis F&B: Tren Linear, YoY, dan Momentum."""
    if len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan)."}
    
    df = df_monthly.dropna(subset=[metric_col]).copy()
    if len(set(df[metric_col])) <= 1:
        return {'narrative': "Data konstan, tidak ada tren yang bisa dianalisis."}

    df['x_val'] = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05:
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"

    yoy_narrative = ""
    if len(df) >= 13:
        last_val = df.iloc[-1][metric_col]
        yoy_val = df.iloc[-13][metric_col]
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_performance = f"**tumbuh {yoy_change:.1%}**" if yoy_change > 0 else f"**menurun {abs(yoy_change):.1%}**"
            yoy_narrative = f" Dibandingkan bulan yang sama tahun lalu, performa bulan terakhir {yoy_performance}."

    ma_line, momentum_narrative = None, ""
    if len(df) >= 4:
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        momentum_narrative = " Momentum jangka pendek (3 bulan terakhir) terlihat **positif**." if ma_line.iloc[-1] > ma_line.iloc[-2] else " Momentum jangka pendek menunjukkan **perlambatan**."

    max_perf_month = df.loc[df[metric_col].idxmax()]
    min_perf_month = df.loc[df[metric_col].idxmin()]
    extrema_narrative = f" Performa tertinggi tercatat pada **{max_perf_month['Bulan'].strftime('%B %Y')}** dan terendah pada **{min_perf_month['Bulan'].strftime('%B %Y')}**."

    full_narrative = f"Secara keseluruhan, tren {metric_label} cenderung {trend_type}.{momentum_narrative}{yoy_narrative}{extrema_narrative}"
    
    return {'narrative': full_narrative, 'trendline': trendline, 'ma_line': ma_line, 'p_value': p_value}

def display_analysis_with_details_v3(title, analysis_result):
    """Menampilkan narasi analisis dan detail p-value dalam expander."""
    st.info(f"💡 **Analisis Tren {title}:** {analysis_result.get('narrative', 'Analisis tidak tersedia.')}")
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat penjelasan signifikansi statistik (p-value)"):
            st.markdown(f"**Nilai p-value** tren ini adalah **`{p_value:.4f}`**. Angka ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05:
                st.success("✔️ Karena kemungkinan kebetulan rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else:
                st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

def create_channel_analysis(df):
    """Membuat visualisasi untuk analisis saluran penjualan dengan layout dan insight baru."""
    st.subheader("📊 Analisis Saluran Penjualan")
    if 'Visit Purpose' not in df.columns:
        st.warning("Kolom 'Visit Purpose' tidak dipetakan. Analisis saluran penjualan tidak tersedia.")
        return

    col1, col2 = st.columns(2)
    with col1:
        channel_sales = df.groupby('Visit Purpose')['Nett Sales'].sum().sort_values(ascending=False)
        fig = px.pie(channel_sales, values='Nett Sales', names=channel_sales.index, title="Kontribusi Penjualan per Saluran", hole=0.4)
        fig.update_layout(legend_title_text='Saluran')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        agg_data = df.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
        agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
        aov_by_channel = agg_data['AOV'].sort_values(ascending=False)
        fig2 = px.bar(aov_by_channel, x=aov_by_channel.index, y=aov_by_channel.values,
                      labels={'y': 'Rata-rata Nilai Pesanan (AOV)', 'x': 'Saluran'}, title="AOV per Saluran Penjualan")
        st.plotly_chart(fig2, use_container_width=True)
        
    if not aov_by_channel.empty and not channel_sales.empty:
        highest_aov_channel = aov_by_channel.index[0]
        highest_contrib_channel = channel_sales.index[0]
        st.info(f"""
        **Insight Bisnis:**
        - **Kontributor Terbesar:** Saluran **{highest_contrib_channel}** adalah penyumbang pendapatan terbesar Anda. Prioritaskan operasional dan promosi untuk saluran ini.
        - **Nilai Pesanan Tertinggi:** Pelanggan dari saluran **{highest_aov_channel}** cenderung menghabiskan lebih banyak per transaksi. Pertimbangkan untuk memberikan penawaran eksklusif atau program loyalitas untuk segmen ini guna meningkatkan frekuensi kunjungan mereka.
        """)

def create_menu_engineering_chart(df):
    """Membuat visualisasi kuadran untuk menu engineering dengan expander untuk penjelasan."""
    st.subheader("🔬 Analisis Performa Menu")
    menu_perf = df.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
    if len(menu_perf) < 4:
        st.warning("Data menu tidak cukup untuk analisis kuadran yang berarti."); return

    avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
    show_text = len(menu_perf) < 75 
    text_arg = menu_perf['Menu'] if show_text else None

    fig = px.scatter(menu_perf, x='Qty', y='NettSales', text=text_arg, title="Kuadran Performa Menu",
                     labels={'Qty': 'Total Kuantitas Terjual', 'NettSales': 'Total Penjualan Bersih (Rp)'},
                     size='NettSales', color='NettSales', hover_name='Menu')
    
    fig.add_shape(type="rect", x0=avg_qty, y0=avg_sales, x1=menu_perf['Qty'].max(), y1=menu_perf['NettSales'].max(), fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=menu_perf['Qty'].min(), y0=avg_sales, x1=avg_qty, y1=menu_perf['NettSales'].max(), fillcolor="lightblue", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=avg_qty, y0=menu_perf['NettSales'].min(), x1=menu_perf['Qty'].max(), y1=avg_sales, fillcolor="lightyellow", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=menu_perf['Qty'].min(), y0=menu_perf['NettSales'].min(), x1=avg_qty, y1=avg_sales, fillcolor="lightcoral", opacity=0.2, layer="below", line_width=0)
    
    fig.add_vline(x=avg_qty, line_dash="dash", line_color="gray")
    fig.add_hline(y=avg_sales, line_dash="dash", line_color="gray")
    
    if show_text:
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_traces(textposition='top center')
    else:
        st.warning("⚠️ Label nama menu disembunyikan karena jumlahnya terlalu banyak. Arahkan mouse ke titik untuk melihat detail.")
        
    st.plotly_chart(fig, use_container_width=True)

    stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
    workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
    
    insight_text = "**Rekomendasi Aksi:**\n"
    if not stars.empty:
        top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
        insight_text += f"- **Fokuskan Promosi:** Menu **'{top_star}'** adalah Bintang (Star) utama Anda. Jadikan pusat promosi.\n"
    if not workhorses.empty:
        top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
        insight_text += f"- **Peluang Profit:** Menu **'{top_workhorse}'** sangat populer (Workhorse). Coba naikkan harga atau buat paket bundling."
    
    if len(insight_text) > 20: st.info(insight_text)
    
    # --- PERBAIKAN: Mengubah st.info menjadi st.expander ---
    with st.expander("💡 Cara Membaca Kuadran Performa Menu"):
        st.markdown("""
        Grafik ini membantu Anda mengkategorikan item menu berdasarkan popularitas (sumbu X) dan profitabilitas (sumbu Y) untuk membuat keputusan strategis.
        - **<span style='color:green; font-weight:bold;'>STARS 🌟 (Kanan Atas):</span>** Juara Anda! Populer dan menguntungkan. **Aksi: Pertahankan dan promosikan!**
        - **<span style='color:darkgoldenrod; font-weight:bold;'>WORKHORSES 🐴 (Kanan Bawah):</span>** Populer tapi kurang profit. **Aksi: Coba naikkan harga sedikit atau tawarkan paket bundling.**
        - **<span style='color:blue; font-weight:bold;'>PUZZLES 🤔 (Kiri Atas):</span>** Sangat profit tapi jarang dipesan. **Aksi: Cari tahu mengapa kurang laku, latih staf untuk merekomendasikan.**
        - **<span style='color:red; font-weight:bold;'>DOGS 🐶 (Kiri Bawah):</span>** Kurang populer & profit. **Aksi: Pertimbangkan untuk menghapusnya dari menu untuk efisiensi.**
        """, unsafe_allow_html=True)

def create_operational_efficiency_analysis(df):
    """
    Membuat visualisasi efisiensi, menggabungkan insight bisnis utama 
    dengan uji statistik opsional dalam sebuah expander.
    """
    st.subheader("⏱️ Analisis Efisiensi Operasional")
    required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
    if not all(col in df.columns for col in required_cols):
        st.warning("Kolom waktu (In, Out, Order Time) atau Bill Number tidak dipetakan."); return

    df_eff = df.copy()
    df_eff['Sales Date In'], df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce'), pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
    df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
    
    if df_eff.empty: return st.warning("Data waktu masuk/keluar tidak valid atau kosong.")
    
    df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
    df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]

    if df_eff.empty:
        st.warning("Tidak ada data waktu persiapan yang valid (0-60 menit) untuk dianalisis."); return

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Waktu Persiapan Rata-rata", f"{df_eff['Prep Time (Seconds)'].mean():.1f} dtk")
    kpi_col2.metric("Waktu Persiapan Tercepat", f"{df_eff['Prep Time (Seconds)'].min():.1f} dtk")
    kpi_col3.metric("Waktu Persiapan Terlama", f"{df_eff['Prep Time (Seconds)'].max():.1f} dtk")

    if 'Order Time' in df_eff.columns and not df_eff['Order Time'].isnull().all():
        df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
        
        agg_by_hour = df_eff.groupby('Hour').agg(
            AvgPrepTime=('Prep Time (Seconds)', 'mean'),
            TotalTransactions=('Bill Number', 'nunique')
        ).reset_index()

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=agg_by_hour['Hour'], y=agg_by_hour['TotalTransactions'], name="Jumlah Transaksi"), secondary_y=False)
        fig.add_trace(go.Scatter(x=agg_by_hour['Hour'], y=agg_by_hour['AvgPrepTime'], name="Rata-rata Waktu Persiapan", mode='lines+markers'), secondary_y=True)
        
        fig.update_layout(title_text="Korelasi Waktu Persiapan & Jumlah Pengunjung per Jam")
        fig.update_xaxes(title_text="Jam dalam Sehari")
        fig.update_yaxes(title_text="<b>Jumlah Transaksi</b> (Batang)", secondary_y=False)
        fig.update_yaxes(title_text="<b>Rata-rata Waktu Persiapan (Detik)</b> (Garis)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        
        # --- PERBAIKAN: Menggabungkan insight bisnis dengan uji statistik ---
        if not agg_by_hour.empty:
            peak_visitor_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]
            longest_prep_hour = agg_by_hour.loc[agg_by_hour['AvgPrepTime'].idxmax()]
            
            # 1. Tampilkan insight bisnis utama yang selalu terlihat
            st.info(f"""
            **Insight Bisnis:**
            - **Jam Puncak Pengunjung:** Kepadatan tertinggi terjadi pada jam **{int(peak_visitor_hour['Hour'])}:00**, dengan **{int(peak_visitor_hour['TotalTransactions'])}** transaksi.
            - **Layanan Melambat:** Waktu persiapan terlama terjadi pada jam **{int(longest_prep_hour['Hour'])}:00**.
            
            **Rekomendasi Aksi:** Jika jam layanan melambat **sama atau berdekatan** dengan jam puncak, ini adalah sinyal kuat dapur Anda kewalahan. Pertimbangkan untuk menambah staf atau menyederhanakan menu pada jam-jam krusial tersebut.
            """)
            
            # 2. Sediakan uji statistik di dalam expander
            with st.expander("🔬 Lihat Uji Statistik Korelasi"):
                if len(agg_by_hour) > 2:
                    correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
                    
                    stat_col1, stat_col2 = st.columns(2)
                    stat_col1.metric("Koefisien Korelasi Spearman (ρ)", f"{correlation:.3f}")
                    stat_col2.metric("P-value", f"{p_value:.3f}")

                    strength = ""
                    if abs(correlation) >= 0.7: strength = "sangat kuat"
                    elif abs(correlation) >= 0.5: strength = "kuat"
                    elif abs(correlation) >= 0.3: strength = "moderat"
                    else: strength = "lemah"

                    if p_value < 0.05:
                        st.success(f"""
                        **Kesimpulan Statistik:** Terdapat korelasi positif yang **{strength} dan signifikan secara statistik**. 
                        Ini **membuktikan secara angka** bahwa saat pengunjung lebih ramai, layanan dapur memang cenderung melambat.
                        """)
                    else:
                        st.warning(f"""
                        **Kesimpulan Statistik:** Meskipun terlihat ada korelasi, hasil ini **tidak signifikan secara statistik**. 
                        Artinya, kita **tidak bisa menyimpulkan dengan yakin** bahwa kepadatan pengunjung adalah penyebab layanan melambat berdasarkan data ini saja.
                        """)
                else:
                    st.warning("Tidak cukup data per jam untuk melakukan uji korelasi statistik.")

def old_generate_executive_summary(df_filtered, monthly_agg):
    """
    Menciptakan ringkasan eksekutif otomatis dari semua analisis,
    termasuk rekomendasi dari Saluran Penjualan dan Efisiensi Operasional (jika signifikan).
    """
    
    # --- 1. Analisis Kesehatan Makro ---
    health_status, health_color = "Perlu Perhatian", "orange"
    trend_analysis = analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan')
    trend_narrative = trend_analysis.get('narrative', 'Gagal menganalisis tren.')
    yoy_change_value = None

    if "meningkat** secara signifikan" in trend_narrative:
        health_status = "Baik" if "perlambatan" in trend_narrative else "Sangat Baik"
        health_color = "orange" if "perlambatan" in trend_narrative else "green"
    elif "menurun** secara signifikan" in trend_narrative:
        health_status, health_color = "Waspada", "red"

    if "Dibandingkan bulan yang sama tahun lalu" in trend_narrative:
        df = monthly_agg.dropna(subset=['TotalMonthlySales'])
        if len(df) >= 13:
            last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
            if yoy_val > 0: yoy_change_value = (last_val - yoy_val) / yoy_val

    # --- 2. Rekomendasi Aksi Otomatis ---
    recommendations = []
    
    # A. Rekomendasi dari Analisis Menu (Existing Logic)
    try:
        menu_perf = df_filtered.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
        if len(menu_perf) >= 4:
            avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
            stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
            workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
            if not stars.empty:
                top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
                recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{top_star}'**.")
            if not workhorses.empty:
                top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
                recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{top_workhorse}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    except Exception: pass

    # B. Rekomendasi dari Analisis Saluran Penjualan (BARU)
    try:
        if 'Visit Purpose' in df_filtered.columns:
            channel_sales = df_filtered.groupby('Visit Purpose')['Nett Sales'].sum()
            agg_data = df_filtered.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
            agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
            aov_by_channel = agg_data['AOV']

            if not channel_sales.empty and not aov_by_channel.empty:
                highest_contrib_channel = channel_sales.idxmax()
                highest_aov_channel = aov_by_channel.idxmax()
                
                # Hanya tambahkan rekomendasi jika salurannya berbeda dan signifikan
                if highest_contrib_channel == highest_aov_channel:
                     recommendations.append(f"🏆 **Maksimalkan Saluran Utama:** Saluran **'{highest_contrib_channel}'** adalah kontributor terbesar DAN memiliki AOV tertinggi. Prioritaskan segalanya di sini!")
                else:
                    recommendations.append(f"💰 **Jaga Kontributor Terbesar:** Pertahankan performa saluran **'{highest_contrib_channel}'** yang menjadi penyumbang pendapatan utama Anda.")
                    recommendations.append(f"📈 **Tingkatkan Frekuensi Saluran AOV Tinggi:** Pelanggan di **'{highest_aov_channel}'** belanja paling banyak per transaksi. Buat program loyalitas untuk mereka.")
    except Exception: pass

    # C. Rekomendasi dari Analisis Efisiensi (BARU & Berbasis Signifikansi)
    try:
        required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
        if all(col in df_filtered.columns for col in required_cols):
            df_eff = df_filtered.copy()
            df_eff['Sales Date In'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce')
            df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
            df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
            df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
            df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
            df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
            
            agg_by_hour = df_eff.groupby('Hour').agg(
                AvgPrepTime=('Prep Time (Seconds)', 'mean'),
                TotalTransactions=('Bill Number', 'nunique')
            ).reset_index()

            if len(agg_by_hour) > 2:
                correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
                # Rekomendasi hanya muncul jika korelasi POSITIF dan SIGNIFIKAN secara statistik
                if p_value < 0.05 and correlation > 0.3:
                    peak_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]['Hour']
                    recommendations.append(f"⏱️ **Atasi Kepadatan:** Layanan melambat saat ramai (terbukti statistik). Tambah sumber daya atau sederhanakan menu pada jam puncak sekitar pukul **{int(peak_hour)}:00**.")
    except Exception: pass


    # --- 3. Fokus Bulan Berikutnya ---
    next_focus = "Pantau dampak eksekusi rekomendasi pada **AOV** dan **kecepatan layanan**."

    return {
        "health_status": health_status, "health_color": health_color, "yoy_change": yoy_change_value,
        "trend_narrative": trend_narrative, "recommendations": recommendations, "next_focus": next_focus
    }

def sc_generate_executive_summary(df_filtered, monthly_agg):
    """
    Menciptakan ringkasan eksekutif dengan analisis multi-metrik (Penjualan, Transaksi, AOV)
    dan narasi kontekstual yang cerdas.
    """
    
    # --- 1. Analisis Kesehatan Makro dengan Sistem Skor ---
    
    # Langkah 1: Analisis tren untuk setiap metrik kunci
    sales_analysis = analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan')
    trx_analysis = analyze_trend_v3(monthly_agg, 'TotalTransactions', 'Transaksi')
    aov_analysis = analyze_trend_v3(monthly_agg, 'AOV', 'AOV')

    # Langkah 2: Hitung skor kesehatan berdasarkan tren setiap metrik
    health_score = 0
    for analysis in [sales_analysis, trx_analysis, aov_analysis]:
        narrative = analysis.get('narrative', '')
        if "meningkat** secara signifikan" in narrative:
            health_score += 2
        elif "menurun** secara signifikan" in narrative:
            health_score -= 2
        
        if "Momentum jangka pendek (3 bulan terakhir) terlihat **positif**" in narrative:
            health_score += 1
        elif "Momentum jangka pendek menunjukkan **perlambatan**" in narrative:
            health_score -= 1

    # Langkah 3: Tentukan status kesehatan akhir dari total skor
    health_status, health_color = "Perlu Perhatian", "orange"
    if health_score > 4:
        health_status, health_color = "Sangat Baik", "green"
    elif health_score >= 1:
        health_status, health_color = "Baik", "green"
    elif health_score <= -4:
        health_status, health_color = "Waspada", "red"
    # Skor antara -3 dan 0 tetap "Perlu Perhatian" karena sinyalnya campuran

    # Langkah 4: Buat narasi kontekstual berdasarkan pola antar metrik
    health_context_narrative = ""
    sales_up = "meningkat** secara signifikan" in sales_analysis.get('narrative', '')
    sales_down = "menurun** secara signifikan" in sales_analysis.get('narrative', '')
    trx_up = "meningkat** secara signifikan" in trx_analysis.get('narrative', '')
    aov_up = "meningkat** secara signifikan" in aov_analysis.get('narrative', '')
    aov_down = "menurun** secara signifikan" in aov_analysis.get('narrative', '')

    if sales_up and aov_up and not trx_up:
        health_context_narrative = "💡 **Insight Kunci:** Pertumbuhan didorong oleh **nilai belanja yang lebih tinggi (AOV naik)**, bukan dari penambahan jumlah transaksi."
    elif sales_up and trx_up and aov_down:
        health_context_narrative = "⚠️ **Perhatian:** Penjualan naik karena **volume transaksi yang tinggi**, namun AOV turun. Ini bisa jadi sinyal **terlalu banyak diskon** atau pergeseran ke produk yang lebih murah."
    elif sales_down and trx_up and aov_down:
        health_context_narrative = "🚨 **Waspada:** Jumlah transaksi mungkin naik, tapi **penurunan AOV yang tajam** menekan total penjualan secara signifikan. Analisis strategi harga dan promosi."
    
    # --- Sisanya sama dengan sebelumnya ---
    yoy_change_value = None
    if "Dibandingkan bulan yang sama tahun lalu" in sales_analysis.get('narrative', ''):
        df = monthly_agg.dropna(subset=['TotalMonthlySales'])
        if len(df) >= 13:
            last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
            if yoy_val > 0: yoy_change_value = (last_val - yoy_val) / yoy_val

    # --- 2. Rekomendasi Aksi Otomatis ---
    recommendations = []
    
    # A. Rekomendasi dari Analisis Menu (Existing Logic)
    try:
        menu_perf = df_filtered.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
        if len(menu_perf) >= 4:
            avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
            stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
            workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
            if not stars.empty:
                top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
                recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{top_star}'**.")
            if not workhorses.empty:
                top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
                recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{top_workhorse}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    except Exception: pass

    # B. Rekomendasi dari Analisis Saluran Penjualan (BARU)
    try:
        if 'Visit Purpose' in df_filtered.columns:
            channel_sales = df_filtered.groupby('Visit Purpose')['Nett Sales'].sum()
            agg_data = df_filtered.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
            agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
            aov_by_channel = agg_data['AOV']

            if not channel_sales.empty and not aov_by_channel.empty:
                highest_contrib_channel = channel_sales.idxmax()
                highest_aov_channel = aov_by_channel.idxmax()
                
                # Hanya tambahkan rekomendasi jika salurannya berbeda dan signifikan
                if highest_contrib_channel == highest_aov_channel:
                     recommendations.append(f"🏆 **Maksimalkan Saluran Utama:** Saluran **'{highest_contrib_channel}'** adalah kontributor terbesar DAN memiliki AOV tertinggi. Prioritaskan segalanya di sini!")
                else:
                    recommendations.append(f"💰 **Jaga Kontributor Terbesar:** Pertahankan performa saluran **'{highest_contrib_channel}'** yang menjadi penyumbang pendapatan utama Anda.")
                    recommendations.append(f"📈 **Tingkatkan Frekuensi Saluran AOV Tinggi:** Pelanggan di **'{highest_aov_channel}'** belanja paling banyak per transaksi. Buat program loyalitas untuk mereka.")
    except Exception: pass

    # C. Rekomendasi dari Analisis Efisiensi (BARU & Berbasis Signifikansi)
    try:
        required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
        if all(col in df_filtered.columns for col in required_cols):
            df_eff = df_filtered.copy()
            df_eff['Sales Date In'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce')
            df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
            df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
            df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
            df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
            df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
            
            agg_by_hour = df_eff.groupby('Hour').agg(
                AvgPrepTime=('Prep Time (Seconds)', 'mean'),
                TotalTransactions=('Bill Number', 'nunique')
            ).reset_index()

            if len(agg_by_hour) > 2:
                correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
                # Rekomendasi hanya muncul jika korelasi POSITIF dan SIGNIFIKAN secara statistik
                if p_value < 0.05 and correlation > 0.3:
                    peak_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]['Hour']
                    recommendations.append(f"⏱️ **Atasi Kepadatan:** Layanan melambat saat ramai (terbukti statistik). Tambah sumber daya atau sederhanakan menu pada jam puncak sekitar pukul **{int(peak_hour)}:00**.")
    except Exception: pass


    # --- 3. Fokus Bulan Berikutnya ---
    next_focus = "Pantau dampak eksekusi rekomendasi pada **AOV** dan **kecepatan layanan**."

    # --- Pengembalian nilai ---
    return {
        "health_status": health_status, "health_color": health_color, "yoy_change": yoy_change_value,
        "trend_narrative": sales_analysis.get('narrative', 'Gagal menganalisis tren penjualan.'),
        "health_context_narrative": health_context_narrative, # <-- NILAI BARU
        "recommendations": recommendations, "next_focus": "Pantau dampak eksekusi pada AOV, Transaksi, dan Kecepatan Layanan."
    }

def old2_generate_executive_summary(df_filtered, monthly_agg):
    """
    Menciptakan ringkasan eksekutif otomatis dari semua analisis,
    termasuk skor kesehatan multi-metrik, narasi kontekstual, dan rekomendasi cerdas.
    """
    
    # --- 1. Analisis Kesehatan Makro dengan Sistem Skor ---
    
    # Langkah 1: Analisis tren untuk setiap metrik kunci
    analyses = {
        'Penjualan': analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan'),
        'Transaksi': analyze_trend_v3(monthly_agg, 'TotalTransactions', 'Transaksi'),
        'AOV': analyze_trend_v3(monthly_agg, 'AOV', 'AOV')
    }

    # Langkah 2: Hitung skor kesehatan dan simpan rinciannya
    score_details = {}
    health_score = 0
    for metric, analysis in analyses.items():
        metric_score = 0
        narrative = analysis.get('narrative', '')
        if "meningkat** secara signifikan" in narrative:
            metric_score += 2
        elif "menurun** secara signifikan" in narrative:
            metric_score -= 2
        
        if "Momentum jangka pendek (3 bulan terakhir) terlihat **positif**" in narrative:
            metric_score += 1
        elif "Momentum jangka pendek menunjukkan **perlambatan**" in narrative:
            metric_score -= 1
        
        score_details[metric] = metric_score
        health_score += metric_score

    # Langkah 3: Tentukan status kesehatan akhir dari total skor
    health_status, health_color = "Perlu Perhatian", "orange"
    if health_score > 4:
        health_status, health_color = "Sangat Baik", "green"
    elif health_score >= 1:
        health_status, health_color = "Baik", "green"
    elif health_score <= -4:
        health_status, health_color = "Waspada", "red"

    # Langkah 4: Buat narasi kontekstual berdasarkan pola antar metrik
    health_context_narrative = ""
    sales_up = "meningkat** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    sales_down = "menurun** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    trx_up = "meningkat** secara signifikan" in analyses['Transaksi'].get('narrative', '')
    aov_up = "meningkat** secara signifikan" in analyses['AOV'].get('narrative', '')
    aov_down = "menurun** secara signifikan" in analyses['AOV'].get('narrative', '')

    if sales_up and aov_up and not trx_up:
        health_context_narrative = "💡 **Insight Kunci:** Pertumbuhan didorong oleh **nilai belanja yang lebih tinggi (AOV naik)**, bukan dari penambahan jumlah transaksi."
    elif sales_up and trx_up and aov_down:
        health_context_narrative = "⚠️ **Perhatian:** Penjualan naik karena **volume transaksi yang tinggi**, namun AOV turun. Ini bisa jadi sinyal **terlalu banyak diskon** atau pergeseran ke produk yang lebih murah."
    elif sales_down and trx_up and aov_down:
        health_context_narrative = "🚨 **Waspada:** Jumlah transaksi mungkin naik, tapi **penurunan AOV yang tajam** menekan total penjualan secara signifikan. Analisis strategi harga dan promosi."

    # --- 2. Ekstraksi Data Tambahan & Rekomendasi Aksi Otomatis ---
    yoy_change_value = None
    if "Dibandingkan bulan yang sama tahun lalu" in analyses['Penjualan'].get('narrative', ''):
        df = monthly_agg.dropna(subset=['TotalMonthlySales'])
        if len(df) >= 13:
            last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
            if yoy_val > 0: yoy_change_value = (last_val - yoy_val) / yoy_val

    recommendations = []
    
    # A. Rekomendasi dari Analisis Menu
    try:
        menu_perf = df_filtered.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
        if len(menu_perf) >= 4:
            avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
            stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
            workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
            if not stars.empty:
                top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
                recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{top_star}'**.")
            if not workhorses.empty:
                top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
                recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{top_workhorse}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    except Exception: pass

    # B. Rekomendasi dari Analisis Saluran Penjualan
    try:
        if 'Visit Purpose' in df_filtered.columns:
            channel_sales = df_filtered.groupby('Visit Purpose')['Nett Sales'].sum()
            agg_data = df_filtered.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
            agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
            aov_by_channel = agg_data['AOV']

            if not channel_sales.empty and not aov_by_channel.empty:
                highest_contrib_channel = channel_sales.idxmax()
                highest_aov_channel = aov_by_channel.idxmax()
                
                if highest_contrib_channel == highest_aov_channel:
                     recommendations.append(f"🏆 **Maksimalkan Saluran Utama:** Saluran **'{highest_contrib_channel}'** adalah kontributor terbesar DAN memiliki AOV tertinggi. Prioritaskan segalanya di sini!")
                else:
                    recommendations.append(f"💰 **Jaga Kontributor Terbesar:** Pertahankan performa saluran **'{highest_contrib_channel}'** yang menjadi penyumbang pendapatan utama Anda.")
                    recommendations.append(f"📈 **Tingkatkan Frekuensi Saluran AOV Tinggi:** Pelanggan di **'{highest_aov_channel}'** belanja paling banyak per transaksi. Buat program loyalitas untuk mereka.")
    except Exception: pass

    # C. Rekomendasi dari Analisis Efisiensi (Berbasis Signifikansi)
    try:
        required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
        if all(col in df_filtered.columns for col in required_cols):
            df_eff = df_filtered.copy()
            df_eff['Sales Date In'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce')
            df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
            df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
            df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
            df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
            df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
            
            agg_by_hour = df_eff.groupby('Hour').agg(
                AvgPrepTime=('Prep Time (Seconds)', 'mean'),
                TotalTransactions=('Bill Number', 'nunique')
            ).reset_index()

            if len(agg_by_hour) > 2:
                correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
                if p_value < 0.05 and correlation > 0.3:
                    peak_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]['Hour']
                    recommendations.append(f"⏱️ **Atasi Kepadatan:** Layanan melambat saat ramai (terbukti statistik). Tambah sumber daya atau sederhanakan menu pada jam puncak sekitar pukul **{int(peak_hour)}:00**.")
    except Exception: pass

    # --- 3. Fokus Bulan Berikutnya ---
    next_focus = "Pantau dampak eksekusi rekomendasi pada AOV, Transaksi, dan Kecepatan Layanan."

    # --- 4. Kumpulkan Semua Hasil ---
    return {
        "health_status": health_status, 
        "health_color": health_color, 
        "yoy_change": yoy_change_value,
        "trend_narrative": analyses['Penjualan'].get('narrative', 'Gagal menganalisis tren penjualan.'),
        "health_context_narrative": health_context_narrative,
        "score_details": score_details,
        "recommendations": recommendations, 
        "next_focus": next_focus
    }

def generate_executive_summary(df_filtered, monthly_agg):
    """
    Menciptakan ringkasan eksekutif otomatis dari semua analisis,
    termasuk skor kesehatan multi-metrik, narasi kontekstual, dan rekomendasi cerdas.
    """
    
    # --- 1. Analisis Kesehatan Makro dengan Sistem Skor ---
    
    # Langkah 1: Analisis tren untuk setiap metrik kunci
    analyses = {
        'Penjualan': analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan'),
        'Transaksi': analyze_trend_v3(monthly_agg, 'TotalTransactions', 'Transaksi'),
        'AOV': analyze_trend_v3(monthly_agg, 'AOV', 'AOV')
    }

    # Langkah 2: Hitung skor kesehatan dan simpan rinciannya
    score_details = {}
    health_score = 0
    for metric, analysis in analyses.items():
        trend_score = 0
        momentum_score = 0
        narrative = analysis.get('narrative', '')
        
        # Skor untuk tren jangka panjang
        if "meningkat** secara signifikan" in narrative:
            trend_score = 2
        elif "menurun** secara signifikan" in narrative:
            trend_score = -2
        
        # Skor untuk momentum jangka pendek
        if "Momentum jangka pendek (3 bulan terakhir) terlihat **positif**" in narrative:
            momentum_score = 1
        elif "Momentum jangka pendek menunjukkan **perlambatan**" in narrative:
            momentum_score = -1
        
        total_metric_score = trend_score + momentum_score
        
        # Simpan struktur data yang lebih kaya
        score_details[metric] = {
            'total': total_metric_score,
            'tren_jangka_panjang': trend_score,
            'momentum_jangka_pendek': momentum_score
        }
        
        health_score += total_metric_score

    # Langkah 3: Tentukan status kesehatan akhir dari total skor
    health_status, health_color = "Perlu Perhatian", "orange"
    if health_score > 4:
        health_status, health_color = "Sangat Baik", "green"
    elif health_score >= 1:
        health_status, health_color = "Baik", "green"
    elif health_score <= -4:
        health_status, health_color = "Waspada", "red"

    # Langkah 4: Buat narasi kontekstual berdasarkan pola antar metrik
    health_context_narrative = ""
    sales_up = "meningkat** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    sales_down = "menurun** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    trx_up = "meningkat** secara signifikan" in analyses['Transaksi'].get('narrative', '')
    aov_up = "meningkat** secara signifikan" in analyses['AOV'].get('narrative', '')
    aov_down = "menurun** secara signifikan" in analyses['AOV'].get('narrative', '')

    if sales_up and aov_up and not trx_up:
        health_context_narrative = "💡 **Insight Kunci:** Pertumbuhan didorong oleh **nilai belanja yang lebih tinggi (AOV naik)**, bukan dari penambahan jumlah transaksi."
    elif sales_up and trx_up and aov_down:
        health_context_narrative = "⚠️ **Perhatian:** Penjualan naik karena **volume transaksi yang tinggi**, namun AOV turun. Ini bisa jadi sinyal **terlalu banyak diskon** atau pergeseran ke produk yang lebih murah."
    elif sales_down and trx_up and aov_down:
        health_context_narrative = "🚨 **Waspada:** Jumlah transaksi mungkin naik, tapi **penurunan AOV yang tajam** menekan total penjualan secara signifikan. Analisis strategi harga dan promosi."

    # --- 2. Ekstraksi Data Tambahan & Rekomendasi Aksi Otomatis ---
    yoy_change_value = None
    if "Dibandingkan bulan yang sama tahun lalu" in analyses['Penjualan'].get('narrative', ''):
        df = monthly_agg.dropna(subset=['TotalMonthlySales'])
        if len(df) >= 13:
            last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
            if yoy_val > 0: yoy_change_value = (last_val - yoy_val) / yoy_val

    recommendations = []
    
    # A. Rekomendasi dari Analisis Menu
    try:
        menu_perf = df_filtered.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
        if len(menu_perf) >= 4:
            avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
            stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
            workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
            if not stars.empty:
                top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
                recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{top_star}'**.")
            if not workhorses.empty:
                top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
                recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{top_workhorse}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    except Exception: pass

    # B. Rekomendasi dari Analisis Saluran Penjualan
    try:
        if 'Visit Purpose' in df_filtered.columns:
            channel_sales = df_filtered.groupby('Visit Purpose')['Nett Sales'].sum()
            agg_data = df_filtered.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
            agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
            aov_by_channel = agg_data['AOV']

            if not channel_sales.empty and not aov_by_channel.empty:
                highest_contrib_channel = channel_sales.idxmax()
                highest_aov_channel = aov_by_channel.idxmax()
                
                if highest_contrib_channel == highest_aov_channel:
                     recommendations.append(f"🏆 **Maksimalkan Saluran Utama:** Saluran **'{highest_contrib_channel}'** adalah kontributor terbesar DAN memiliki AOV tertinggi. Prioritaskan segalanya di sini!")
                else:
                    recommendations.append(f"💰 **Jaga Kontributor Terbesar:** Pertahankan performa saluran **'{highest_contrib_channel}'** yang menjadi penyumbang pendapatan utama Anda.")
                    recommendations.append(f"📈 **Tingkatkan Frekuensi Saluran AOV Tinggi:** Pelanggan di **'{highest_aov_channel}'** belanja paling banyak per transaksi. Buat program loyalitas untuk mereka.")
    except Exception: pass

    # C. Rekomendasi dari Analisis Efisiensi (Berbasis Signifikansi)
    try:
        required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
        if all(col in df_filtered.columns for col in required_cols):
            df_eff = df_filtered.copy()
            df_eff['Sales Date In'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce')
            df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
            df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
            df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
            df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
            df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
            
            agg_by_hour = df_eff.groupby('Hour').agg(
                AvgPrepTime=('Prep Time (Seconds)', 'mean'),
                TotalTransactions=('Bill Number', 'nunique')
            ).reset_index()

            if len(agg_by_hour) > 2:
                correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
                if p_value < 0.05 and correlation > 0.3:
                    peak_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]['Hour']
                    recommendations.append(f"⏱️ **Atasi Kepadatan:** Layanan melambat saat ramai (terbukti statistik). Tambah sumber daya atau sederhanakan menu pada jam puncak sekitar pukul **{int(peak_hour)}:00**.")
    except Exception: pass

    # --- 3. Fokus Bulan Berikutnya ---
    next_focus = "Pantau dampak eksekusi rekomendasi pada AOV, Transaksi, dan Kecepatan Layanan."

    # --- 4. Kumpulkan Semua Hasil ---
    return {
        "health_status": health_status, 
        "health_color": health_color, 
        "yoy_change": yoy_change_value,
        "trend_narrative": analyses['Penjualan'].get('narrative', 'Gagal menganalisis tren penjualan.'),
        "health_context_narrative": health_context_narrative,
        "score_details": score_details,
        "recommendations": recommendations, 
        "next_focus": next_focus
    }

def old_display_executive_summary(summary):
    """Menampilkan ringkasan eksekutif dengan layout UI/UX yang compact."""
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metrik utama yang paling penting
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else None
            st.metric(
                label="Status Kesehatan Bisnis", 
                value=summary['health_status'],
                delta=f"{delta_text} vs Tahun Lalu" if delta_text else None,
                delta_color="normal" # Warna delta mengikuti positif/negatif
            )
        
        with col2:
            # Kesimpulan akhir yang langsung bisa ditindaklanjuti
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

        # Gunakan expander untuk menyembunyikan detail
        with st.expander("🔍 Lihat Detail Analisis & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama:**")
            st.write(summary['trend_narrative'])
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

def sc_display_executive_summary(summary):
    """Menampilkan ringkasan eksekutif dengan layout UI/UX yang compact."""
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else None
            st.metric(
                label="Status Kesehatan Bisnis", 
                value=summary['health_status'],
                delta=f"{delta_text} vs Tahun Lalu" if delta_text else None,
                delta_color="normal"
            )
        
        with col2:
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

        # Gunakan expander untuk menyembunyikan detail
        with st.expander("🔍 Lihat Detail Analisis & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])

            # --- TAMBAHAN BARU DI SINI ---
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])
            # --- AKHIR TAMBAHAN ---
            
            st.markdown("---") # Tambahkan pemisah
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

def old2_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout UI/UX yang compact,
    termasuk narasi kontekstual dan rincian skor yang bisa diperluas.
    """
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metrik utama yang paling penting
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else None
            st.metric(
                label="Status Kesehatan Bisnis", 
                value=summary['health_status'],
                delta=f"{delta_text} vs Tahun Lalu" if delta_text else None,
                delta_color="normal" # Warna delta mengikuti positif/negatif
            )
        
        with col2:
            # Kesimpulan akhir yang langsung bisa ditindaklanjuti
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

        # Gunakan expander untuk menyembunyikan detail
        with st.expander("🔍 Lihat Detail Analisis & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            # Menampilkan narasi kontekstual jika ada
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            # Expander baru untuk rincian skor
            with st.expander("🔬 Lihat Rincian Perhitungan Skor Kesehatan"):
                total_score = sum(summary['score_details'].values())
                st.markdown(f"**Total Skor Kesehatan: `{total_score}`**")
                
                for metric, score in summary['score_details'].items():
                    emoji = "➡️"
                    if score > 0: emoji = "📈"
                    elif score < 0: emoji = "📉"
                    st.markdown(f"- **{metric}:** `{score}` poin {emoji}")
                st.caption("Skor dihitung dari tren jangka panjang (+2/-2) dan momentum jangka pendek (+1/-1).")
            
            # st.markdown("---")
            
            # Menampilkan rekomendasi aksi
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

def sc2_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout UI/UX yang compact,
    termasuk narasi kontekstual dan rincian skor analitis yang bisa diperluas.
    """
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Metrik utama yang paling penting
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else None
            st.metric(
                label="Status Kesehatan Bisnis", 
                value=summary['health_status'],
                delta=f"{delta_text} vs Tahun Lalu" if delta_text else None,
                delta_color="normal" # Warna delta mengikuti positif/negatif
            )
        
        with col2:
            # Kesimpulan akhir yang langsung bisa ditindaklanjuti
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

        # Gunakan expander untuk menyembunyikan detail
        with st.expander("🔍 Lihat Detail Analisis & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            # Menampilkan narasi kontekstual jika ada
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            # Expander baru untuk rincian skor analitis
            with st.expander("🔬 Lihat Rincian Analisis Skor Kesehatan"):
                total_score = sum(item['total'] for item in summary['score_details'].values())
                st.markdown(f"#### Total Skor Kesehatan: `{total_score}` Poin")
                
                for metric, details in summary['score_details'].items():
                    score = details['total']
                    emoji = "➡️"
                    if score > 0: emoji = "📈"
                    elif score < 0: emoji = "📉"
                    
                    st.markdown(f"##### **{metric}: `{score}` Poin** {emoji}")
                    
                    # Tampilkan komponen skor
                    tren_text = f"*- Tren Jangka Panjang: `{details['tren_jangka_panjang']}`*"
                    momentum_text = f"*- Momentum Jangka Pendek: `{details['momentum_jangka_pendek']}`*"

                    # Beri penjelasan agar lebih jelas
                    if details['tren_jangka_panjang'] == 2: tren_text += " *(Naik Signifikan)*"
                    if details['tren_jangka_panjang'] == -2: tren_text += " *(Turun Signifikan)*"
                    if details['momentum_jangka_pendek'] == 1: momentum_text += " *(Positif)*"
                    if details['momentum_jangka_pendek'] == -1: momentum_text += " *(Melambat)*"

                    st.markdown(tren_text)
                    st.markdown(momentum_text)
            
            # Menampilkan rekomendasi aksi
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

def sc3_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout baru yang lebih intuitif,
    menempatkan rincian skor di sebelah status dan menambahkan legenda.
    """
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        # --- BLOK UTAMA YANG DIUBAH ---
        # Kita gunakan 2 kolom untuk menampilkan status, legenda, dan rincian skor secara berdampingan.
        col1, col2 = st.columns(2, gap="large")

        with col1:
            # 1. Menampilkan Status Kesehatan Utama
            st.markdown("##### Status Kesehatan Bisnis")
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else None
            st.metric(
                label=summary['health_status'], 
                value=f"Total Skor: {sum(item['total'] for item in summary['score_details'].values())}",
                delta=f"{delta_text} vs Tahun Lalu" if delta_text else None,
                delta_color="normal"
            )
            st.markdown("---")
            
            # 2. Menampilkan Legenda Skor
            st.markdown("##### Legenda Skor")
            st.markdown(
                """
                - <span style='color:green; font-weight:bold;'>Sangat Baik</span> : Skor > 4
                - <span style='color:green; font-weight:bold;'>Baik</span> : Skor 1 s.d. 4
                - <span style='color:orange; font-weight:bold;'>Perlu Perhatian</span> : Skor -3 s.d. 0
                - <span style='color:red; font-weight:bold;'>Waspada</span> : Skor < -4
                """,
                unsafe_allow_html=True
            )

        with col2:
            # 3. Menampilkan Rincian Analisis Skor secara langsung
            st.markdown("##### Rincian Analisis Skor")
            for metric, details in summary['score_details'].items():
                score = details['total']
                emoji = "➡️"
                if score > 0: emoji = "📈"
                elif score < 0: emoji = "📉"
                
                st.markdown(f"**{metric}: `{score}` Poin** {emoji}")
                
                tren_text = f"*- Tren Jk. Panjang: `{details['tren_jangka_panjang']}`*"
                momentum_text = f"*- Momentum Jk. Pendek: `{details['momentum_jangka_pendek']}`*"

                if details['tren_jangka_panjang'] == 2: tren_text += " *(Naik Signifikan)*"
                if details['tren_jangka_panjang'] == -2: tren_text += " *(Turun Signifikan)*"
                if details['momentum_jangka_pendek'] == 1: momentum_text += " *(Positif)*"
                if details['momentum_jangka_pendek'] == -1: momentum_text += " *(Melambat)*"

                st.markdown(f"<small>{tren_text}<br>{momentum_text}</small>", unsafe_allow_html=True)
        
        # --- AKHIR BLOK UTAMA YANG DIUBAH ---

        # Detail lainnya tetap berada di dalam expander
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

def sc4_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout horizontal yang sangat compact,
    memprioritaskan kepadatan informasi dalam ruang minimal.
    """
    
    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        # Gunakan 3 kolom untuk layout horizontal yang padat
        col1, col2, col3 = st.columns([1.5, 1, 2.5])

        # KOLOM 1: Status Kesehatan sebagai "Badge" Visual
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            status = summary['health_status']
            color = summary['health_color']
            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">
                    {status.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

        # KOLOM 2: KPI Kinerja Utama (YoY)
        with col2:
            st.markdown("Performa YoY")
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else "N/A"
            st.metric(label="vs. Tahun Lalu", value=delta_text)

        # KOLOM 3: Rincian Skor yang Diringkas
        with col3:
            st.markdown("Analisis Skor")
            score_texts = []
            for metric, details in summary['score_details'].items():
                # Format ringkas: "Metric: Total (T:X, M:Y)"
                score_texts.append(
                    f"**{metric}: {details['total']}**&nbsp;(T: {details['tren_jangka_panjang']}, M: {details['momentum_jangka_pendek']})"
                )
            # Gabungkan dengan pemisah "•" untuk kepadatan
            st.markdown(" &bull; ".join(score_texts))
            st.caption("T: Tren Jk. Panjang, M: Momentum Jk. Pendek")

        # --- Detail lainnya tetap di dalam expander ---
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            # (Tidak ada perubahan pada bagian ini)
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

def sc5_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout horizontal yang sangat compact,
    menggunakan ikon panah berwarna untuk visualisasi skor yang intuitif.
    """
    
    # --- Fungsi Bantuan (Helper Function) untuk membuat panah ---
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.2em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.2em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.2em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.2em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.2em;'>↓↓</span>"
        return ""

    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.5, 1, 2.5])

        # KOLOM 1: Status Kesehatan sebagai "Badge" Visual
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            status = summary['health_status']
            color = summary['health_color']
            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">
                    {status.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

        # KOLOM 2: KPI Kinerja Utama (YoY)
        with col2:
            st.markdown("Performa YoY")
            delta_text = f"{summary['yoy_change']:.1%}" if summary['yoy_change'] is not None else "N/A"
            st.metric(label="vs. Tahun Lalu", value=delta_text)

        # KOLOM 3: Rincian Skor dengan Visualisasi Panah
        with col3:
            st.markdown("Analisis Skor")
            score_texts = []
            for metric, details in summary['score_details'].items():
                # --- PERUBAHAN UTAMA DI SINI ---
                # Menggunakan helper function untuk mengubah angka menjadi panah
                total_arrow = format_score_with_arrows(details['total'])
                trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                
                score_texts.append(
                    f"**{metric}:** {total_arrow} (T: {trend_arrow}, M: {momentum_arrow})"
                )
            
            st.markdown(" &bull; ".join(score_texts), unsafe_allow_html=True)
            st.caption("T: Tren Jk. Panjang, M: Momentum Jk. Pendek")

        # --- Detail lainnya tetap di dalam expander ---
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            # (Tidak ada perubahan pada bagian ini)
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

def sc6_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout horizontal yang compact,
    visualisasi YoY yang disempurnakan, dan analisis skor multi-baris.
    """
    
    # --- Fungsi Bantuan (Helper Function) untuk membuat panah ---
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.2em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.2em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.2em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.2em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.2em;'>↓↓</span>"
        return ""

    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.5, 1.2, 2.3])

        # KOLOM 1: Status Kesehatan sebagai "Badge" Visual
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            status = summary['health_status']
            color = summary['health_color']
            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">
                    {status.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

        # KOLOM 2: KPI Kinerja Utama (YoY) dengan Panah & Warna
        with col2:
            st.markdown("Performa YoY")
            yoy_change = summary.get('yoy_change')
            
            # --- PERUBAHAN 1: Logika untuk menampilkan YoY dengan panah & warna ---
            if yoy_change is not None:
                if yoy_change > 0:
                    display_val = f"<h3 style='color:green;'>↑ {yoy_change:.1%}</h3>"
                else:
                    display_val = f"<h3 style='color:red;'>↓ {abs(yoy_change):.1%}</h3>"
            else:
                display_val = "<h3>N/A</h3>"
            
            st.markdown(display_val, unsafe_allow_html=True)
            st.caption("vs. Tahun Lalu")


        # KOLOM 3: Rincian Skor dalam 3 Baris untuk Keterbacaan
        with col3:
            st.markdown("Analisis Skor")
            
            # --- PERUBAHAN 2: Menampilkan setiap metrik di baris baru ---
            for metric, details in summary['score_details'].items():
                total_arrow = format_score_with_arrows(details['total'])
                trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                
                # Format teks dan tampilkan langsung di dalam loop
                score_text = f"**{metric}:** {total_arrow} (T: {trend_arrow}, M: {momentum_arrow})"
                st.markdown(score_text, unsafe_allow_html=True)

            st.caption("T: Tren Jangka Panjang, M: Momentum Jangka Pendek")

        # --- Detail lainnya tetap di dalam expander ---
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            # (Tidak ada perubahan pada bagian ini)
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

def old_display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout 2 kolom:
    Kolom 1 berisi status kesehatan dan performa YoY.
    Kolom 2 berisi rincian analisis skor.
    """
    
    # --- Fungsi Bantuan (Helper Function) untuk membuat panah ---
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.2em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.2em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.2em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.2em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.2em;'>↓↓</span>"
        return ""

    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        # --- PERUBAHAN UTAMA: Menggunakan 2 kolom dengan rasio tertentu ---
        col1, col2 = st.columns([1, 1.8])

        # KOLOM 1: Berisi Status Kesehatan & Performa YoY
        with col1:
            # 1a. Status Kesehatan Bisnis
            st.markdown("Status Kesehatan Bisnis")
            status = summary['health_status']
            color = summary['health_color']
            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">
                    {status.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("---") # Pemisah visual

            # 1b. Performa YoY di bawahnya
            st.markdown("Performa YoY")
            yoy_change = summary.get('yoy_change')
            
            if yoy_change is not None:
                if yoy_change > 0:
                    display_val = f"<h3 style='color:green;'>↑ {yoy_change:.1%}</h3>"
                else:
                    display_val = f"<h3 style='color:red;'>↓ {abs(yoy_change):.1%}</h3>"
            else:
                display_val = "<h3>N/A</h3>"
            
            st.markdown(display_val, unsafe_allow_html=True)
            st.caption("vs. Tahun Lalu")


        # KOLOM 2: Berisi Rincian Analisis Skor
        with col2:
            st.markdown("Analisis Skor Kesehatan")
            
            for metric, details in summary['score_details'].items():
                total_arrow = format_score_with_arrows(details['total'])
                trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                
                score_text = f"**{metric}:** {total_arrow} (T: {trend_arrow}, M: {momentum_arrow})"
                st.markdown(score_text, unsafe_allow_html=True)

            st.caption("T: Tren Jk. Panjang, M: Momentum Jk. Pendek")

        # --- Detail lainnya tetap di dalam expander ---
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            # (Tidak ada perubahan pada bagian ini)
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

def display_executive_summary(summary):
    """
    Menampilkan ringkasan eksekutif dengan layout baris tunggal yang jernih,
    menggunakan kolom-mini untuk memisahkan metrik skor secara visual.
    """
    
    # --- Fungsi Bantuan (Helper Function) untuk membuat panah ---
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.2em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.2em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.2em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.2em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.2em;'>↓↓</span>"
        return ""

    st.subheader("Ringkasan Eksekutif")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.5, 1.2, 2.3])

        # KOLOM 1: Status Kesehatan
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            status = summary['health_status']
            color = summary['health_color']
            st.markdown(
                f"""
                <div style="background-color:{color}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">
                    {status.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

        # KOLOM 2: Performa YoY
        with col2:
            st.markdown("Performa YoY")
            yoy_change = summary.get('yoy_change')
            
            if yoy_change is not None:
                if yoy_change > 0:
                    display_val = f"<h3 style='color:green;'>↑ {yoy_change:.1%}</h3>"
                else:
                    display_val = f"<h3 style='color:red;'>↓ {abs(yoy_change):.1%}</h3>"
            else:
                display_val = "<h3>N/A</h3>"
            
            st.markdown(display_val, unsafe_allow_html=True)
            st.caption("vs. Tahun Lalu")

        # KOLOM 3: Analisis Skor dalam Baris Tunggal yang Terstruktur
        with col3:
            st.markdown("Analisis Skor Kesehatan")
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            
            metrics_to_display = ['Penjualan', 'Transaksi', 'AOV']
            columns_to_use = [sub_col1, sub_col2, sub_col3]

            for metric_name, sub_col in zip(metrics_to_display, columns_to_use):
                with sub_col:
                    details = summary['score_details'].get(metric_name)
                    if details:
                        total_arrow = format_score_with_arrows(details['total'])
                        trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                        momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                        
                        st.markdown(f"**{metric_name}**")
                        # st.markdown(f"<h3>{total_arrow}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='margin:0;'>{total_arrow}</h3>", unsafe_allow_html=True)

                        
                        # --- INI BARIS YANG DIPERBAIKI ---
                        # Mengganti st.caption dengan st.markdown dan tag <small>
                        caption_text = f"<small>Tren: {trend_arrow}, Momentum: {momentum_arrow}</small>"
                        st.markdown(caption_text, unsafe_allow_html=True)

        # --- Detail lainnya tetap di dalam expander ---
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            # (Tidak ada perubahan pada bagian ini)
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            
            if summary.get('health_context_narrative'):
                st.markdown(summary['health_context_narrative'])

            st.markdown("---")
            
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")

            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

# ==============================================================================
# LOGIKA AUTENTIKASI DAN APLIKASI UTAMA
# ==============================================================================
config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False: st.error("Username atau password salah.")
elif auth_status is None: st.warning("Silakan masukkan username dan password.")
elif auth_status:
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False

    if os.path.exists("logo.png"): st.sidebar.image("logo.png", width=150)
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{name}**")
    st.sidebar.title("📤 Unggah & Mapping Kolom")

    uploaded_sales_file = st.sidebar.file_uploader(
        "1. Unggah Sales Recapitulation Detail Report", type=["xlsx", "xls", "csv"],
        key="sales_uploader", on_change=reset_processing_state
    )
    
    if uploaded_sales_file is None:
        st.info("👋 Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis."); st.stop()

    df_raw = load_raw_data(uploaded_sales_file)
    
    st.sidebar.subheader("🔗 Mapping Kolom")
    REQUIRED_COLS_MAP = {'Sales Date': 'Tgl. Transaksi', 'Branch': 'Nama Cabang', 'Bill Number': 'No. Struk/Bill', 'Nett Sales': 'Penjualan Bersih', 'Menu': 'Nama Item/Menu', 'Qty': 'Kuantitas'}
    OPTIONAL_COLS_MAP = {'Visit Purpose': 'Saluran Penjualan', 'Payment Method': 'Metode Pembayaran', 'Sales Date In': 'Waktu Pesanan Masuk', 'Sales Date Out': 'Waktu Pesanan Selesai', 'Order Time': 'Jam Pesanan'}
    user_mapping = {}
    
    def find_best_match(col_list, keywords):
        for col in col_list:
            if col and isinstance(col, str):
                for keyword in keywords:
                    if keyword in col.lower().replace("_", "").replace(" ", ""): return col
        return None

    with st.sidebar.expander("Atur Kolom Wajib", expanded=not st.session_state.data_processed):
        all_cols = [""] + df_raw.columns.tolist()
        for internal_name, desc in REQUIRED_COLS_MAP.items():
            best_guess = find_best_match(all_cols, [internal_name.lower().replace("_","").replace(" ",""), desc.lower()])
            user_selection = st.selectbox(f"**{desc}**:", options=all_cols, index=(all_cols.index(best_guess) if best_guess else 0), key=f"map_req_{internal_name}")
            if user_selection: user_mapping[internal_name] = user_selection
    
    with st.sidebar.expander("Atur Kolom Opsional"):
        for internal_name, desc in OPTIONAL_COLS_MAP.items():
            best_guess = find_best_match(all_cols, [internal_name.lower().replace("_","").replace(" ",""), desc.lower()])
            user_selection = st.selectbox(f"**{desc}**:", options=all_cols, index=(all_cols.index(best_guess) if best_guess else 0), key=f"map_opt_{internal_name}")
            if user_selection: user_mapping[internal_name] = user_selection

    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        mapped_req_cols = [user_mapping.get(k) for k in REQUIRED_COLS_MAP.keys()]
        if not all(mapped_req_cols):
            st.error("❌ Harap petakan semua kolom WAJIB diisi."); st.stop()
        
        chosen_cols = [c for c in user_mapping.values() if c]
        if len(chosen_cols) != len(set(chosen_cols)):
            st.error("❌ Terdeteksi satu kolom dipilih untuk beberapa peran berbeda."); st.stop()

        try:
            df = pd.DataFrame()
            for internal_name, source_col in user_mapping.items():
                if source_col: df[internal_name] = df_raw[source_col]

            df['Menu'] = df['Menu'].fillna('N/A').astype(str)
            df['Branch'] = df['Branch'].fillna('Tidak Diketahui').astype(str)
            if 'Visit Purpose' in df.columns: df['Visit Purpose'] = df['Visit Purpose'].fillna('N/A').astype(str)
            if 'Payment Method' in df.columns: df['Payment Method'] = df['Payment Method'].fillna('N/A').astype(str)
            df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce')
            if 'Sales Date In' in df.columns: df['Sales Date In'] = pd.to_datetime(df['Sales Date In'], errors='coerce')
            if 'Sales Date Out' in df.columns: df['Sales Date Out'] = pd.to_datetime(df['Sales Date Out'], errors='coerce')
            if 'Order Time' in df.columns: df['Order Time'] = pd.to_datetime(df['Order Time'], errors='coerce', format='mixed').dt.time
            numeric_cols = ['Qty', 'Nett Sales'] 
            for col in numeric_cols:
                 if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            st.session_state.df_processed = df 
            st.session_state.data_processed = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}"); st.stop()

    # GANTI seluruh blok if st.session_state.data_processed dengan ini
    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        st.sidebar.title("⚙️ Filter Global")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date, max_date = df_processed['Sales Date'].dt.date.min(), df_processed['Sales Date'].dt.date.max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered = df_processed[
            (df_processed['Branch'] == selected_branch) & 
            (df_processed['Sales Date'].dt.date >= start_date) & 
            (df_processed['Sales Date'].dt.date <= end_date)
        ]
        
        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih."); st.stop()
            
        st.title(f"Dashboard Holistik: {selected_branch}")
        st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

        # --- PERBAIKAN: Menghitung agregasi bulanan di awal untuk summary ---
        monthly_df = df_filtered.copy()
        monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
        monthly_agg = monthly_df.groupby('Bulan').agg(
            TotalMonthlySales=('Nett Sales', 'sum'),
            TotalTransactions=('Bill Number', 'nunique')
        ).reset_index()
        if not monthly_agg.empty:
            monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

        # --- PEMANGGILAN FUNGSI SUMMARY & TAMPILANNYA ---
        if not monthly_agg.empty and len(monthly_agg) >=3 :
            summary = generate_executive_summary(df_filtered, monthly_agg)  
            # Panggil fungsi display yang baru
            display_executive_summary(summary)

        # --- Tampilan Tab Dasbor ---
        trend_tab, ops_tab = st.tabs(["📈 **Dashboard Tren Performa**", "🚀 **Dashboard Analisis Operasional**"])

        with trend_tab:
            st.header("Analisis Tren Performa Jangka Panjang")
            
            monthly_df = df_filtered.copy()
            monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
            monthly_agg = monthly_df.groupby('Bulan').agg(
                TotalMonthlySales=('Nett Sales', 'sum'),
                TotalTransactions=('Bill Number', 'nunique')
            ).reset_index()
            monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

            if not monthly_agg.empty:
                monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
                monthly_agg.fillna(0, inplace=True)

                kpi_cols = st.columns(3)
                last_month = monthly_agg.iloc[-1]
                prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None

                def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
                    if pd.isna(current_val): col.metric(title, "N/A"); return
                    delta = 0
                    if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
                        delta = (current_val - prev_val) / prev_val
                    val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.2f}".rstrip('0').rstrip('.')
                    delta_display = f"{delta:.1%}" if prev_val is not None and pd.notna(prev_val) else None
                    col.metric(title, val_format, delta_display, help=help_text if delta_display else None)

                help_str = f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
                display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales'), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
                display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions'), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
                display_kpi(kpi_cols[2], "💳 AOV Bulanan", last_month.get('AOV'), prev_month.get('AOV') if prev_month is not None else None, help_str, True)
                
                st.markdown("---")

                def create_trend_chart_v3(df_data, y_col, y_label, color):
                    analysis_result = analyze_trend_v3(df_data, y_col, y_label)
                    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
                    fig.update_traces(line_color=color, name=y_label)
                    if analysis_result.get('trendline') is not None: fig.add_scatter(x=df_data['Bulan'], y=analysis_result['trendline'], mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
                    if analysis_result.get('ma_line') is not None: fig.add_scatter(x=df_data['Bulan'], y=analysis_result['ma_line'], mode='lines', name='3-Month Moving Avg.', line=dict(color='orange', dash='dot'))
                    st.plotly_chart(fig, use_container_width=True)
                    display_analysis_with_details_v3(y_label, analysis_result)

                create_trend_chart_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
                create_trend_chart_v3(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
                create_trend_chart_v3(monthly_agg, 'AOV', 'AOV', 'green')
            else:
                st.warning("Tidak ada data bulanan yang cukup untuk analisis tren pada periode ini.")

        with ops_tab:
            st.header("Wawasan Operasional dan Taktis")
            create_channel_analysis(df_filtered.copy())
            st.markdown("---")
            create_menu_engineering_chart(df_filtered.copy())
            st.markdown("---")
            create_operational_efficiency_analysis(df_filtered.copy())