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
# FUNGSI-FUNGSI ANALISIS & TAMPILAN
# ==============================================================================
@st.cache_data
def load_raw_data(file):
    """Memuat data mentah dari file yang diunggah."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
        return df.copy()
    except Exception as e:
        raise ValueError(f"Gagal memuat file: {e}")

def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """Menganalisis tren dengan wawasan bisnis F&B."""
    if len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': "Data tidak cukup untuk analisis tren (minimal 3 bulan)."}
    df = df_monthly.dropna(subset=[metric_col]).copy()
    if len(set(df[metric_col])) <= 1:
        return {'narrative': "Data konstan, tidak ada tren yang bisa dianalisis."}
    df['x_val'] = np.arange(len(df))
    slope, intercept, _, p_value, _ = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05:
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    yoy_narrative = ""
    if len(df) >= 13:
        last_val, yoy_val = df.iloc[-1][metric_col], df.iloc[-13][metric_col]
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

# --- FUNGSI YANG DIKEMBALIKAN 1 ---
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

def generate_executive_summary(df_filtered, monthly_agg):
    """Menciptakan ringkasan eksekutif otomatis yang holistik."""
    analyses = {'Penjualan': analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan'),'Transaksi': analyze_trend_v3(monthly_agg, 'TotalTransactions', 'Transaksi'),'AOV': analyze_trend_v3(monthly_agg, 'AOV', 'AOV')}
    score_details = {}
    trend_health_score = 0
    for metric, analysis in analyses.items():
        trend_score, momentum_score = 0, 0
        narrative = analysis.get('narrative', '')
        if "meningkat** secara signifikan" in narrative: trend_score = 2
        elif "menurun** secara signifikan" in narrative: trend_score = -2
        if "Momentum jangka pendek (3 bulan terakhir) terlihat **positif**" in narrative: momentum_score = 1
        elif "Momentum jangka pendek menunjukkan **perlambatan**" in narrative: momentum_score = -1
        total_metric_score = trend_score + momentum_score
        score_details[metric] = {'total': total_metric_score,'tren_jangka_panjang': trend_score,'momentum_jangka_pendek': momentum_score}
        trend_health_score += total_metric_score
    yoy_change_value, yoy_score = None, 0
    if "Dibandingkan bulan yang sama tahun lalu" in analyses['Penjualan'].get('narrative', ''):
        df = monthly_agg.dropna(subset=['TotalMonthlySales'])
        if len(df) >= 13:
            last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
            if yoy_val > 0:
                yoy_change_value = (last_val - yoy_val) / yoy_val
                if yoy_change_value > 0.15: yoy_score = 2
                elif yoy_change_value > 0.05: yoy_score = 1
                elif yoy_change_value < -0.15: yoy_score = -2
                elif yoy_change_value < -0.05: yoy_score = -1
    score_details['YoY'] = {'total': yoy_score, 'tren_jangka_panjang': yoy_score, 'momentum_jangka_pendek': 0}
    health_score = trend_health_score + yoy_score
    health_status, health_color = "Perlu Perhatian", "orange"
    if health_score > 5: health_status, health_color = "Sangat Baik", "green"
    elif health_score >= 2: health_status, health_color = "Baik", "green"
    elif health_score <= -5: health_status, health_color = "Waspada", "red"
    health_context_narrative = ""
    sales_up = "meningkat** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    sales_down = "menurun** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    trx_up = "meningkat** secara signifikan" in analyses['Transaksi'].get('narrative', '')
    aov_up = "meningkat** secara signifikan" in analyses['AOV'].get('narrative', '')
    aov_down = "menurun** secara signifikan" in analyses['AOV'].get('narrative', '')
    if sales_up and aov_up and not trx_up: health_context_narrative = "💡 **Insight Kunci:** Pertumbuhan didorong oleh **nilai belanja yang lebih tinggi (AOV naik)**, bukan dari penambahan jumlah transaksi."
    elif sales_up and trx_up and aov_down: health_context_narrative = "⚠️ **Perhatian:** Penjualan naik karena **volume transaksi yang tinggi**, namun AOV turun. Ini bisa jadi sinyal **terlalu banyak diskon**."
    elif sales_down and trx_up and aov_down: health_context_narrative = "🚨 **Waspada:** Jumlah transaksi mungkin naik, tapi **penurunan AOV yang tajam** menekan total penjualan."
    recommendations = []
    try:
        menu_perf = df_filtered.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
        if len(menu_perf) >= 4:
            avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
            stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
            workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
            if not stars.empty: recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{stars.nlargest(1, 'NettSales')['Menu'].iloc[0]}'**.")
            if not workhorses.empty: recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    except Exception: pass
    next_focus = "Pantau dampak eksekusi rekomendasi pada AOV, Transaksi, dan Kecepatan Layanan."
    return {"health_status": health_status, "health_color": health_color, "yoy_change": yoy_change_value,"trend_narrative": analyses['Penjualan'].get('narrative', 'Gagal menganalisis tren.'),"health_context_narrative": health_context_narrative,"score_details": score_details,"recommendations": recommendations, "next_focus": next_focus}

def display_executive_summary(summary):
    """Menampilkan ringkasan eksekutif dengan layout baris tunggal yang jernih."""
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.1em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.1em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.1em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.1em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.1em;'>↓↓</span>"
        return ""
    st.subheader("Ringkasan Eksekutif")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.5, 1.2, 2.3])
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            st.markdown(f'<div style="background-color:{summary["health_color"]}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">{summary["health_status"].upper()}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("Performa YoY")
            yoy_change = summary.get('yoy_change')
            display_val = f"<h3 style='color:green; margin:0;'>↑ {yoy_change:.1%}</h3>" if yoy_change is not None and yoy_change > 0 else f"<h3 style='color:red; margin:0;'>↓ {abs(yoy_change):.1%}</h3>" if yoy_change is not None else "<h3 style='margin:0;'>N/A</h3>"
            st.markdown(display_val, unsafe_allow_html=True)
            st.caption("vs. Tahun Lalu")
        with col3:
            st.markdown("Analisis Skor Kesehatan")
            sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
            metrics_to_display = ['Penjualan', 'Transaksi', 'AOV', 'YoY']
            columns_to_use = [sub_col1, sub_col2, sub_col3, sub_col4]
            for metric_name, sub_col in zip(metrics_to_display, columns_to_use):
                with sub_col:
                    details = summary['score_details'].get(metric_name)
                    if details:
                        total_arrow = format_score_with_arrows(details['total'])
                        trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                        momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                        combined_html = f'<div style="line-height: 1.2;"><strong>{metric_name}</strong><br><span style="font-size: 2em; font-weight: bold;">{total_arrow}</span><br><small>T: {trend_arrow}, M: {momentum_arrow}</small></div>'
                        st.markdown(combined_html, unsafe_allow_html=True)
        with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
            st.markdown("**Narasi Tren Utama (Penjualan):**")
            st.write(summary['trend_narrative'])
            if summary.get('health_context_narrative'): st.markdown(summary['health_context_narrative'])
            st.markdown("---")
            if summary['recommendations']:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                for rec in summary['recommendations']: st.markdown(f"- {rec}")
            else:
                st.markdown("**Rekomendasi Aksi Teratas:**")
                st.write("Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")
            st.markdown("---")
            st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")

# --- FUNGSI YANG DIKEMBALIKAN 2 ---
def create_channel_analysis(df):
    """Membuat visualisasi untuk analisis saluran penjualan."""
    st.subheader("📊 Analisis Saluran Penjualan")
    if 'Visit Purpose' not in df.columns:
        st.warning("Kolom 'Visit Purpose' tidak dipetakan.")
        return
    col1, col2 = st.columns(2)
    with col1:
        channel_sales = df.groupby('Visit Purpose')['Nett Sales'].sum().sort_values(ascending=False)
        fig = px.pie(channel_sales, values='Nett Sales', names=channel_sales.index, title="Kontribusi Penjualan per Saluran", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        agg_data = df.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
        agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
        aov_by_channel = agg_data['AOV'].sort_values(ascending=False)
        fig2 = px.bar(aov_by_channel, x=aov_by_channel.index, y='AOV', title="AOV per Saluran Penjualan")
        st.plotly_chart(fig2, use_container_width=True)

# --- FUNGSI YANG DIKEMBALIKAN 3 ---
def create_menu_engineering_chart(df):
    """Membuat visualisasi kuadran untuk menu engineering."""
    st.subheader("🔬 Analisis Performa Menu")
    menu_perf = df.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
    if len(menu_perf) < 4:
        st.warning("Data menu tidak cukup untuk analisis kuadran."); return
    avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
    fig = px.scatter(menu_perf, x='Qty', y='NettSales', title="Kuadran Performa Menu",
                     labels={'Qty': 'Total Kuantitas Terjual', 'NettSales': 'Total Penjualan Bersih'},
                     hover_name='Menu')
    fig.add_vline(x=avg_qty, line_dash="dash")
    fig.add_hline(y=avg_sales, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("💡 Cara Membaca Kuadran"):
        st.markdown("""
        - **STARS (Kanan Atas):** Populer & menguntungkan. Pertahankan!
        - **WORKHORSES (Kanan Bawah):** Populer, kurang profit. Coba naikkan harga.
        - **PUZZLES (Kiri Atas):** Profit, kurang populer. Promosikan.
        - **DOGS (Kiri Bawah):** Kurang populer & profit. Pertimbangkan untuk hapus.
        """)

# --- FUNGSI YANG DIKEMBALIKAN 4 ---
def create_operational_efficiency_analysis(df):
    """Membuat visualisasi efisiensi operasional."""
    st.subheader("⏱️ Analisis Efisiensi Operasional")
    if not all(col in df.columns for col in ['Sales Date In', 'Sales Date Out', 'Order Time']):
        st.warning("Kolom waktu tidak lengkap untuk analisis efisiensi."); return
    df_eff = df.copy()
    df_eff['Prep Time (Seconds)'] = (pd.to_datetime(df_eff['Sales Date Out']) - pd.to_datetime(df_eff['Sales Date In'])).dt.total_seconds()
    df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
    if df_eff.empty:
        st.warning("Tidak ada data waktu persiapan yang valid."); return
    df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'].astype(str), errors='coerce').dt.hour
    agg_by_hour = df_eff.groupby('Hour').agg(AvgPrepTime=('Prep Time (Seconds)', 'mean'), TotalTransactions=('Bill Number', 'nunique')).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg_by_hour['Hour'], y=agg_by_hour['TotalTransactions'], name="Jumlah Transaksi"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg_by_hour['Hour'], y=agg_by_hour['AvgPrepTime'], name="Rata-rata Waktu Persiapan"), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --- FUNGSI ANALISIS STRATEGIS (LEVEL SENIOR) ---
def create_waiter_performance_analysis(df):
    """Menganalisis performa pramusaji melampaui total penjualan."""
    st.subheader("👨‍🍳 Analisis Kinerja Pramusaji (Waiter)")
    if 'Waiter' not in df.columns or 'Discount' not in df.columns:
        st.warning("Kolom 'Waiter' atau 'Discount' tidak dipetakan.")
        return
    waiter_perf = df.groupby('Waiter').agg(TotalSales=('Nett Sales', 'sum'),TotalTransactions=('Bill Number', 'nunique'),TotalDiscount=('Discount', 'sum')).reset_index()
    waiter_perf = waiter_perf[waiter_perf['TotalTransactions'] > 5] # Filter pramusaji dengan minimal 5 transaksi
    if waiter_perf.empty: st.info("Tidak ada data pramusaji yang cukup untuk analisis."); return
    waiter_perf['AOV'] = waiter_perf['TotalSales'] / waiter_perf['TotalTransactions']
    waiter_perf['DiscountRate'] = (waiter_perf['TotalDiscount'] / waiter_perf['TotalSales']).fillna(0)
    col1, col2, col3 = st.columns(3)
    if not waiter_perf.empty:
        with col1:
            top_sales = waiter_perf.nlargest(1, 'TotalSales').iloc[0]
            st.metric("Penjualan Tertinggi", top_sales['Waiter'], f"Rp {top_sales['TotalSales']:,.0f}")
        with col2:
            top_aov = waiter_perf.nlargest(1, 'AOV').iloc[0]
            st.metric("AOV Tertinggi (Upseller)", top_aov['Waiter'], f"Rp {top_aov['AOV']:,.0f}")
        with col3:
            top_discounter = waiter_perf.nlargest(1, 'DiscountRate').iloc[0]
            st.metric("Rasio Diskon Tertinggi", top_discounter['Waiter'], f"{top_discounter['DiscountRate']:.1%}")
    st.info("Perhatikan pramusaji dengan AOV tertinggi, mereka adalah *upseller* terbaik Anda. Selidiki pramusaji dengan rasio diskon tertinggi.")

def create_discount_effectiveness_analysis(df):
    """Menganalisis apakah diskon efektif meningkatkan belanja pelanggan."""
    st.subheader("📉 Analisis Efektivitas Diskon")
    if 'Discount' not in df.columns and 'Bill Discount' not in df.columns:
        st.warning("Kolom diskon tidak dipetakan."); return
    df_analysis = df.copy()
    df_analysis['TotalDiscount'] = df_analysis.get('Discount', 0) + df_analysis.get('Bill Discount', 0)
    df_analysis['HasDiscount'] = df_analysis['TotalDiscount'] > 0
    aov_comparison = df_analysis.groupby('HasDiscount').agg(AOV=('Nett Sales', lambda x: x.sum() / df_analysis.loc[x.index, 'Bill Number'].nunique()))
    if True in aov_comparison.index and False in aov_comparison.index:
        aov_with_discount, aov_without_discount = aov_comparison.loc[True, 'AOV'], aov_comparison.loc[False, 'AOV']
        col1, col2 = st.columns(2)
        col1.metric("AOV dengan Diskon", f"Rp {aov_with_discount:,.0f}")
        col2.metric("AOV tanpa Diskon", f"Rp {aov_without_discount:,.0f}")
        diff = (aov_with_discount - aov_without_discount) / aov_without_discount
        if diff > 0.05: st.success(f"✅ **Efektif**: Diskon berhasil meningkatkan AOV sebesar **{diff:.1%}**.")
        else: st.warning(f"⚠️ **Potensi Kanibalisasi**: Diskon tidak meningkatkan AOV secara signifikan.")

def create_regional_analysis(df):
    """Menganalisis perbedaan performa dan preferensi antar kota."""
    st.subheader("🏙️ Analisis Kinerja Regional")
    if 'City' not in df.columns: st.warning("Kolom 'City' tidak dipetakan."); return
    city_perf = df.groupby('City').agg(TotalSales=('Nett Sales', 'sum'),TotalTransactions=('Bill Number', 'nunique')).reset_index()
    city_perf = city_perf[city_perf['TotalTransactions'] > 5] # Filter kota dengan minimal 5 transaksi
    if city_perf.empty: st.info("Tidak ada data kota yang cukup untuk analisis."); return
    city_perf['AOV'] = city_perf['TotalSales'] / city_perf['TotalTransactions']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=city_perf['City'], y=city_perf['TotalSales'], name='Total Penjualan'), secondary_y=False)
    fig.add_trace(go.Scatter(x=city_perf['City'], y=city_perf['AOV'], name='AOV', mode='lines+markers'), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    top_cities = city_perf.nlargest(min(3, len(city_perf)), 'TotalSales')['City'].tolist()
    if top_cities:
        st.markdown("**Preferensi Menu Teratas per Kota**")
        cols = st.columns(len(top_cities))
        for i, city in enumerate(top_cities):
            with cols[i]:
                st.markdown(f"**📍 {city}**")
                top_menus = df[df['City'] == city].groupby('Menu')['Qty'].sum().nlargest(5).index.tolist()
                for menu in top_menus: st.caption(menu)

# ==============================================================================
# APLIKASI UTAMA
# ==============================================================================
# Bypass autentikasi untuk kemudahan pengembangan
auth_status = True 
if auth_status is False: st.error("Username atau password salah.")
elif auth_status is None: st.warning("Silakan masukkan username dan password.")
elif auth_status:
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False

    st.sidebar.title("📤 Unggah & Mapping Kolom")
    uploaded_sales_file = st.sidebar.file_uploader("1. Unggah Laporan Penjualan Detail", type=["xlsx", "xls", "csv"], key="sales_uploader")
    if uploaded_sales_file is None: st.info("👋 Selamat datang! Unggah file data penjualan untuk memulai."); st.stop()
    df_raw = load_raw_data(uploaded_sales_file)
    st.sidebar.subheader("🔗 Mapping Kolom")
    REQUIRED_COLS_MAP = {'Sales Date': 'Tgl. Transaksi', 'Branch': 'Nama Cabang', 'Bill Number': 'No. Struk/Bill', 'Nett Sales': 'Penjualan Bersih', 'Menu': 'Nama Item/Menu', 'Qty': 'Kuantitas'}
    OPTIONAL_COLS_MAP = {'Visit Purpose': 'Saluran Penjualan', 'Payment Method': 'Metode Pembayaran', 'Sales Date In': 'Waktu Pesanan Masuk', 'Sales Date Out': 'Waktu Pesanan Selesai', 'Order Time': 'Jam Pesanan', 'City': 'Kota', 'Discount': 'Diskon Item', 'Bill Discount': 'Diskon Struk', 'Waiter': 'Pramusaji'}
    user_mapping = {}
    all_cols = [""] + df_raw.columns.tolist()
    with st.sidebar.expander("Atur Kolom Wajib", expanded=True):
        for internal_name, desc in REQUIRED_COLS_MAP.items(): user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, key=f"map_req_{internal_name}")
    with st.sidebar.expander("Atur Kolom Opsional"):
        for internal_name, desc in OPTIONAL_COLS_MAP.items(): user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, key=f"map_opt_{internal_name}")

    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        if not all(user_mapping.get(k) for k in REQUIRED_COLS_MAP.keys()): st.error("❌ Harap petakan semua kolom WAJIB."); st.stop()
        try:
            df = pd.DataFrame()
            for internal_name, source_col in user_mapping.items():
                if source_col: df[internal_name] = df_raw[source_col]
            for col in ['Menu', 'Branch', 'Visit Purpose', 'Payment Method', 'City', 'Waiter']:
                if col in df.columns: df[col] = df[col].fillna('N/A').astype(str)
            for col in ['Sales Date', 'Sales Date In', 'Sales Date Out']:
                if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            if 'Order Time' in df.columns: df['Order Time'] = pd.to_datetime(df['Order Time'], errors='coerce', format='mixed').dt.time
            for col in ['Qty', 'Nett Sales', 'Discount', 'Bill Discount']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            st.session_state.df_processed = df 
            st.session_state.data_processed = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}"); st.stop()

    if st.session_state.get('data_processed', False):
        df_processed = st.session_state.df_processed
        st.sidebar.title("⚙️ Filter Global")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date, max_date = df_processed['Sales Date'].dt.date.min(), df_processed['Sales Date'].dt.date.max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) != 2: st.stop()
        start_date, end_date = date_range
        df_filtered = df_processed[(df_processed['Branch'] == selected_branch) & (df_processed['Sales Date'].dt.date >= start_date) & (df_processed['Sales Date'].dt.date <= end_date)]
        if df_filtered.empty: st.warning("Tidak ada data untuk filter yang dipilih."); st.stop()
        st.title(f"Dashboard Holistik F&B: {selected_branch}")
        st.markdown(f"Periode: **{start_date.strftime('%d %b %Y')}** - **{end_date.strftime('%d %b %Y')}**")
        monthly_df = df_filtered.copy()
        monthly_df['Bulan'] = monthly_df['Sales Date'].dt.to_period('M')
        monthly_agg = monthly_df.groupby('Bulan').agg(TotalMonthlySales=('Nett Sales', 'sum'),TotalTransactions=('Bill Number', 'nunique')).reset_index()
        if not monthly_agg.empty: monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)
        if not monthly_agg.empty and len(monthly_agg) >=3 :
            summary = generate_executive_summary(df_filtered, monthly_agg)
            display_executive_summary(summary)

        # --- Tampilan Tab Dasbor dengan Konten yang Dikembalikan ---
        trend_tab, ops_tab, strategic_tab = st.tabs(["📈 Tren Performa", "🚀 Analisis Operasional", "🧠 Wawasan Strategis"])
        with trend_tab:
            st.header("Analisis Tren Performa Jangka Panjang")
            if not monthly_agg.empty:
                monthly_agg_display = monthly_agg.copy()
                monthly_agg_display['Bulan'] = monthly_agg_display['Bulan'].dt.to_timestamp()
                # --- FUNGSI TREN CHART YANG SEBELUMNYA HILANG ---
                def create_trend_chart_v3(df_data, y_col, y_label, color):
                    analysis_result = analyze_trend_v3(df_data, y_col, y_label)
                    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
                    fig.update_traces(line_color=color)
                    if analysis_result.get('trendline') is not None: fig.add_scatter(x=df_data['Bulan'], y=analysis_result['trendline'], mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
                    if analysis_result.get('ma_line') is not None: fig.add_scatter(x=df_data['Bulan'], y=analysis_result['ma_line'], mode='lines', name='3-Month Moving Avg.', line=dict(color='orange', dash='dot'))
                    st.plotly_chart(fig, use_container_width=True)
                    display_analysis_with_details_v3(y_label, analysis_result)
                create_trend_chart_v3(monthly_agg_display, 'TotalMonthlySales', 'Penjualan', 'royalblue')
                create_trend_chart_v3(monthly_agg_display, 'TotalTransactions', 'Transaksi', 'orange')
                create_trend_chart_v3(monthly_agg_display, 'AOV', 'AOV', 'green')
        with ops_tab:
            st.header("Wawasan Operasional dan Taktis")
            create_channel_analysis(df_filtered.copy())
            st.markdown("---")
            create_menu_engineering_chart(df_filtered.copy())
            st.markdown("---")
            create_operational_efficiency_analysis(df_filtered.copy())
        with strategic_tab:
            st.header("Wawasan Strategis Lanjutan")
            st.markdown("Bagian ini berisi analisis silang untuk menemukan insight level senior yang dapat ditindaklanjuti.")
            st.markdown("---")
            create_waiter_performance_analysis(df_filtered.copy())
            st.markdown("---")
            create_discount_effectiveness_analysis(df_filtered.copy())
            st.markdown("---")
            create_regional_analysis(df_filtered.copy())