import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dashboard F&B Holistik",
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# FUNGSI PEMUATAN DATA
# ==============================================================================
@st.cache_data
def load_feather_file(uploaded_file):
    """Memuat satu file Feather yang sudah digabungkan."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_feather(uploaded_file)
        # Pastikan kolom tanggal memiliki tipe data datetime
        if 'Sales Date' in df.columns:
            df['Sales Date'] = pd.to_datetime(df['Sales Date'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}")
        return None

# ==============================================================================
# FUNGSI-FUNGSI ANALISIS & VISUALISASI (Tidak ada perubahan di bagian ini)
# ==============================================================================

# --- FUNGSI UNTUK ANALISIS PENJUALAN ---

def analyze_monthly_trends(df_filtered):
    """Menghitung agregasi data bulanan untuk metrik kunci (Penjualan, Transaksi, AOV)."""
    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = monthly_df['Sales Date'].dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(
        TotalMonthlySales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()

    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(
            lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0,
            axis=1
        )
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
    return monthly_agg

def display_monthly_kpis(monthly_agg):
    """Menampilkan KPI bulanan teratas dengan perbandingan bulan sebelumnya."""
    if len(monthly_agg) < 1:
        return
    kpi_cols = st.columns(3)
    last_month = monthly_agg.iloc[-1]
    prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None

    def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
        delta = None
        if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
            delta = (current_val - prev_val) / prev_val
        val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.0f}"
        col.metric(
            title,
            val_format,
            f"{delta:.1%}" if delta is not None else None,
            help=help_text if delta is not None else None
        )

    help_str = f"Dibandingkan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
    display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales', 0), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
    display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions', 0), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
    display_kpi(kpi_cols[2], "💳 AOV Bulanan", last_month.get('AOV', 0), prev_month.get('AOV') if prev_month is not None else None, help_str, True)

def display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    """Membuat grafik tren bulanan."""
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)
    st.plotly_chart(fig, use_container_width=True)

def calculate_price_group_analysis(df):
    """Menganalisis performa berdasarkan kelompok harga."""
    if 'Menu' not in df.columns or 'Nett Sales' not in df.columns or 'Qty' not in df.columns:
        return None

    menu_prices = df.groupby('Menu').agg(TotalSales=('Nett Sales', 'sum'), TotalQty=('Qty', 'sum')).reset_index()
    menu_prices = menu_prices[menu_prices['TotalQty'] > 0]
    menu_prices['AvgPrice'] = menu_prices['TotalSales'] / menu_prices['TotalQty']

    if len(menu_prices) < 4: return None

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    menu_prices['PriceGroupLabel'] = kmeans.fit_predict(menu_prices[['AvgPrice']])

    cluster_centers = menu_prices.groupby('PriceGroupLabel')['AvgPrice'].mean().sort_values().index
    label_mapping = {center: f"Kelompok {i+1}" for i, center in enumerate(cluster_centers)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroupLabel'].map(label_mapping)

    price_order = [" (Termurah)", " (Menengah)", " (Mahal)", " (Termahal)"]
    sorted_groups = menu_prices.groupby('PriceGroup')['AvgPrice'].mean().sort_values().index
    final_label_map = {group: group + price_order[i] for i, group in enumerate(sorted_groups)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroup'].map(final_label_map)

    df_with_groups = pd.merge(df, menu_prices[['Menu', 'PriceGroup']], on='Menu', how='left')
    group_performance = df_with_groups.groupby('PriceGroup').agg(TotalSales=('Nett Sales', 'sum'), TotalQty=('Qty', 'sum')).reset_index()

    group_performance['sort_order'] = group_performance['PriceGroup'].str.extract('(\\d+)').astype(int)
    group_performance = group_performance.sort_values('sort_order').drop(columns='sort_order')

    return group_performance

def display_price_group_analysis(analysis_results):
    """Menampilkan visualisasi analisis kelompok harga."""
    st.subheader("📊 Analisis Kelompok Harga")
    if analysis_results is None or analysis_results.empty:
        st.warning("Data tidak cukup untuk analisis kelompok harga.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=analysis_results['PriceGroup'], y=analysis_results['TotalSales'], name='Total Penjualan', marker_color='royalblue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=analysis_results['PriceGroup'], y=analysis_results['TotalQty'], name='Total Kuantitas', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)

    fig.update_layout(title_text="Kontribusi Penjualan vs. Kuantitas per Kelompok Harga", xaxis_title="Kelompok Harga", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="<b>Total Penjualan (Rp)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Kuantitas Terjual</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Grafik ini membandingkan **pendapatan (biru)** dengan **popularitas (oranye)** untuk setiap kelompok harga.")

# --- FUNGSI UNTUK ANALISIS KUALITAS, KOMPLAIN, & ANALISIS QA/QC ---

def calculate_branch_health(df_sales, df_complaints):
    """Menghitung metrik kesehatan cabang."""
    sales_agg = df_sales.groupby('Branch').agg(TotalSales=('Nett Sales', 'sum'), TotalTransactions=('Bill Number', 'nunique')).reset_index()
    
    if df_complaints.empty:
        complaints_agg = pd.DataFrame(columns=['Branch', 'TotalComplaints', 'AvgResolutionTime'])
    else:
        complaints_agg = df_complaints.groupby('Branch').agg(TotalComplaints=('Branch', 'count'), AvgResolutionTime=('Waktu Penyelesaian (Jam)', 'mean')).reset_index()

    df_health = pd.merge(sales_agg, complaints_agg, on='Branch', how='left').fillna(0)
    df_health['ComplaintRatio'] = df_health.apply(lambda row: (row['TotalComplaints'] / row['TotalTransactions']) * 1000 if row['TotalTransactions'] > 0 else 0, axis=1)
    return df_health

def display_branch_health(df_health):
    """Menampilkan dashboard kesehatan cabang."""
    st.subheader("Dashboard Kesehatan Cabang")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_health, x='Branch', y='ComplaintRatio', title="Rasio Komplain per 1000 Transaksi", color='ComplaintRatio', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_health, x='Branch', y='AvgResolutionTime', title="Rata-rata Waktu Penyelesaian Komplain (Jam)", color='AvgResolutionTime', color_continuous_scale='Oranges')
        st.plotly_chart(fig2, use_container_width=True)

    st.info("Cabang yang sehat memiliki **Rasio Komplain** yang rendah dan **Waktu Penyelesaian** yang cepat.")

def display_complaint_analysis(df_complaints):
    """Menampilkan analisis detail dari data komplain."""
    st.subheader("Analisis Detail Komplain")

    if df_complaints.empty:
        st.warning("Tidak ada data komplain untuk periode dan cabang yang dipilih.")
        return

    col1, col2 = st.columns(2)
    with col1:
        kesalahan_agg = df_complaints['kesalahan'].value_counts().reset_index()
        fig1 = px.pie(kesalahan_agg, names='kesalahan', values='count', title="Proporsi Kategori Kesalahan", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        golongan_agg = df_complaints['golongan'].value_counts().reset_index()
        fig2 = px.bar(golongan_agg, x='golongan', y='count', title="Jumlah Komplain per Golongan Prioritas", color='golongan', color_discrete_map={'merah':'#EF553B', 'orange':'#FFA15A', 'hijau':'#00CC96'})
        st.plotly_chart(fig2, use_container_width=True)

def display_qa_qc_analysis(df_qa_qc):
    """Menampilkan analisis data audit QA/QC."""
    st.subheader("⭐ Dashboard Kepatuhan Standar (QA/QC)")

    if df_qa_qc.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih.")
        return

    avg_score = df_qa_qc['Skor Kepatuhan'].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")

    st.markdown("##### Tren Skor Kepatuhan per Audit")
    fig1 = px.line(
        df_qa_qc.sort_values('Sales Date'),
        x='Sales Date',
        y='Skor Kepatuhan',
        color='Branch',
        markers=True,
        title="Tren Skor Kepatuhan per Audit"
    )
    fig1.update_yaxes(range=[0, 105])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("##### Perbandingan Rata-rata Skor per Cabang")
    branch_avg_score = df_qa_qc.groupby('Branch')['Skor Kepatuhan'].mean().reset_index().sort_values('Skor Kepatuhan', ascending=False)
    fig2 = px.bar(
        branch_avg_score,
        x='Branch',
        y='Skor Kepatuhan',
        title="Perbandingan Rata-rata Skor Kepatuhan antar Cabang",
        color='Skor Kepatuhan',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# [BARU] FUNGSI-FUNGSI UNTUK AGENTIC AI ROOT CAUSE ANALYSIS
# ==============================================================================

def analyze_long_term_metric_status(df, date_col, metric_col, agg_method='sum'):
    """
    Menganalisis tren bulanan (min 4 bulan) menggunakan regresi linear
    untuk evaluasi strategis dan deteksi momentum.
    """
    if df is None or df.empty or metric_col not in df.columns:
        return "TIDAK CUKUP DATA"

    # Tentukan metode agregasi data bulanan
    df_resampled = df.set_index(date_col).resample('M')
    if agg_method == 'sum':
        monthly_df = df_resampled[metric_col].sum().reset_index()
    elif agg_method == 'mean':
        monthly_df = df_resampled[metric_col].mean().reset_index()
    elif agg_method == 'nunique':
         monthly_df = df_resampled[metric_col].nunique().reset_index()
    else: # default to count
        monthly_df = df_resampled.size().reset_index(name=metric_col)

    if len(monthly_df) < 4:
        return "DATA < 4 BULAN"

    monthly_df['x'] = np.arange(len(monthly_df))
    # Menggunakan fillna(0) untuk metrik yang mungkin kosong di bulan tertentu
    slope, _, _, p_value, _ = stats.linregress(monthly_df['x'], monthly_df[metric_col].fillna(0))

    trend_status = "TREN STABIL"
    # Menggunakan p-value < 0.1 untuk menangkap tren bisnis yang relevan
    if p_value < 0.1:
        if slope > 0.05: trend_status = "TREN MENINGKAT"
        elif slope < -0.05: trend_status = "TREN MENURUN"

    # Analisis momentum 3 bulan terakhir vs. 3 bulan sebelumnya
    momentum_status = ""
    if len(monthly_df) >= 6: # Butuh setidaknya 6 bulan untuk perbandingan momentum
        last_3_months_avg = monthly_df[metric_col].tail(3).mean()
        prev_3_months_avg = monthly_df[metric_col].iloc[-6:-3].mean()
        if prev_3_months_avg > 0:
            momentum_change = (last_3_months_avg - prev_3_months_avg) / prev_3_months_avg
            if momentum_change > 0.1: momentum_status = " | MOMENTUM POSITIF"
            elif momentum_change < -0.1: momentum_status = " | MOMENTUM NEGATIF"

    return f"{trend_status}{momentum_status}"

def get_data_driven_knowledge_base():
    """
    Basis pengetahuan yang diekstrak dari analisis tabel yang disediakan.
    Aturan ini menghubungkan tren metrik jangka panjang dengan kemungkinan akar masalah
    yang realistis dan berbasis data.
    """
    return [
        # Kasus 1: Masalah terkait layanan dan operasional buruk
        {
            "condition": lambda s: "TREN MENURUN" in s["sales"] and "TREN MENINGKAT" in s["complaints"],
            "root_cause": "Layanan buruk secara konsisten menggerus loyalitas pelanggan (Bad service causing churn)."
        },
        {
            "condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"],
            "root_cause": "Kepatuhan SOP rendah & operasional buruk berdampak negatif pada penjualan (Poor operations hurting loyalty)."
        },
         {
            "condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"] and "MOMENTUM NEGATIF" in s["sales"],
            "root_cause": "Masalah fundamental pada operasional, kemungkinan terkait rekrutmen atau pengadaan (Poor purchasing/hiring loyalty)."
        },
        # Kasus 2: Masalah terkait traffic atau daya tarik
        {
            "condition": lambda s: "TREN MENURUN" in s["transactions"] and "TREN STABIL" in s["sales"] and "TREN STABIL" in s["complaints"],
            "root_cause": "Penurunan jumlah pengunjung (Reduced visitors/traffic), namun nilai belanja per transaksi tetap stabil."
        },
        {
            "condition": lambda s: "TREN MENURUN" in s["transactions"] and "TREN MENURUN" in s["sales"],
             "root_cause": "Penurunan jumlah pengunjung yang signifikan (Reduced visitors/traffic) menjadi pendorong utama penurunan penjualan."
        },

        # Kasus 3: Kinerja AOV dan promosi
        {
            "condition": lambda s: "TREN MENINGKAT" in s["aov"] and "TREN MENURUN" not in s["sales"],
            "root_cause": "Strategi harga/promo berhasil meningkatkan nilai belanja (Good promo / Customers buy more)."
        },
        {
            "condition": lambda s: "TREN MENURUN" in s["aov"],
            "root_cause": "Efektivitas promo/upselling menurun atau terjadi pergeseran belanja ke produk lebih murah."
        },

        # Kasus 4: Performa positif
        {
            "condition": lambda s: "TREN MENINGKAT" in s["sales"] and ("TINGGI" in s["qa_qc"] or "TREN MENURUN" in s["complaints"]),
            "root_cause": "Standar operasional yang baik dan konsisten mendorong pertumbuhan (Good operations standard / Good service)."
        },

        # Kasus 5: Sinyal peringatan dini
        {
            "condition": lambda s: "MOMENTUM NEGATIF" in s["sales"] or "MOMENTUM NEGATIF" in s["transactions"],
            "root_cause": "Waspada! Performa melambat dalam 3 bulan terakhir, sinyal awal potensi penurunan tren."
        }
    ]

def run_strategic_agent(df_sales, df_complaints, df_qa_qc):
    """Menjalankan agent untuk analisis strategis (jangka panjang) berbasis data."""
    all_branches = sorted([str(b) for b in df_sales['Branch'].unique() if pd.notna(b)])
    knowledge_base = get_data_driven_knowledge_base()
    results = []

    for branch in all_branches:
        sales_br = df_sales[df_sales['Branch'] == branch]
        complaints_br = df_complaints[df_complaints['Branch'] == branch]
        qa_qc_br = df_qa_qc[df_qa_qc['Branch'] == branch]

        # Menghitung AOV bulanan sebagai rata-rata nilai transaksi
        bill_agg = sales_br.groupby(['Bill Number', pd.Grouper(key='Sales Date', freq='M')])['Nett Sales'].sum().reset_index()
        monthly_aov_df = bill_agg.groupby('Sales Date')['Nett Sales'].mean().reset_index().rename(columns={'Nett Sales': 'AOV'})

        # Menghitung skor QA/QC rata-rata
        qa_qc_score = "TIDAK ADA DATA"
        if not qa_qc_br.empty:
            avg_score = qa_qc_br['Skor Kepatuhan'].mean()
            if avg_score < 75: qa_qc_score = "RENDAH"
            elif avg_score < 85: qa_qc_score = "SEDANG"
            else: qa_qc_score = "TINGGI"

        # Mengumpulkan status dari semua metrik
        status = {
            "sales": analyze_long_term_metric_status(sales_br, 'Sales Date', 'Nett Sales', 'sum'),
            "transactions": analyze_long_term_metric_status(sales_br, 'Sales Date', 'Bill Number', 'nunique'),
            "aov": analyze_long_term_metric_status(monthly_aov_df, 'Sales Date', 'AOV', 'mean'),
            "complaints": analyze_long_term_metric_status(complaints_br, 'Sales Date', 'Branch', 'count'),
            "qa_qc": qa_qc_score
        }

        # Mencocokkan status dengan knowledge base
        causes = {rule["root_cause"] for rule in knowledge_base if rule["condition"](status)}
        results.append({
            "Toko": branch,
            "Tren Penjualan": status["sales"],
            "Tren Transaksi": status["transactions"],
            "Tren AOV": status["aov"],
            "Skor QA/QC": status["qa_qc"],
            "Tren Komplain": status["complaints"],
            "Analisis & Kemungkinan Akar Masalah": ", ".join(causes) if causes else "Tidak ada pola signifikan yang terdeteksi."
        })
    return pd.DataFrame(results)

def display_agent_analysis(df_analysis, title, info_text):
    """Menampilkan hasil analisis dari AI Agent dalam bentuk tabel yang di-styling."""
    st.header(title)
    st.info(info_text)

    def style_status(val):
        color = "grey"
        if isinstance(val, str):
            if any(keyword in val for keyword in ["MENINGKAT", "TINGGI", "POSITIF"]): color = "#2ca02c" # Hijau
            if any(keyword in val for keyword in ["MENURUN", "RENDAH", "NEGATIF"]): color = "#d62728" # Merah
        return f'color: {color}'

    # Mengubah nama kolom agar lebih mudah dibaca di tabel
    df_display = df_analysis.set_index('Toko')
    styled_df = df_display.style.apply(lambda col: col.map(style_status), subset=pd.IndexSlice[:, df_display.columns != 'Analisis & Kemungkinan Akar Masalah'])
    
    st.dataframe(styled_df, use_container_width=True)

# ==============================================================================
# APLIKASI UTAMA STREAMLIT (Tidak ada perubahan signifikan di sini)
# ==============================================================================
def main_app(user_name):
    """Fungsi utama yang menjalankan seluruh aplikasi dashboard."""

    # --- Bagian 1: Sidebar - Autentikasi dan Unggah File ---
    if 'authenticator' in globals() and 'name' in st.session_state and st.session_state.name:
      authenticator.logout("Logout", "sidebar")
      st.sidebar.success(f"Login sebagai: **{user_name}**")

    st.sidebar.title("📤 Unggah Data Master")

    sales_file = st.sidebar.file_uploader("1. Unggah Penjualan Master (.feather)", type=["feather"])
    complaint_file = st.sidebar.file_uploader("2. Unggah Komplain Master (.feather)", type=["feather"])
    qa_qc_file = st.sidebar.file_uploader("3. Unggah QA/QC Master (.feather)", type=["feather"])

    # --- Bagian 2: Pemuatan Data ---
    if sales_file is None or complaint_file is None or qa_qc_file is None:
        st.info("👋 Selamat datang! Silakan unggah ketiga file master: penjualan, komplain, dan QA/QC.")
        st.stop()
        
    df_sales = load_feather_file(sales_file)
    df_complaints = load_feather_file(complaint_file)
    df_qa_qc = load_feather_file(qa_qc_file)
    
    if df_sales is None or df_complaints is None or df_qa_qc is None:
        st.error("Gagal memuat salah satu dari tiga file data. Pastikan semua file terunggah.")
        st.stop()
        
    st.session_state.df_sales = df_sales
    st.session_state.df_complaints = df_complaints
    st.session_state.df_qa_qc = df_qa_qc

    # --- Bagian 3: Sidebar - Filter Global ---
    st.sidebar.title("⚙️ Filter Global")
    
    ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
    unique_branches = sorted([str(branch) for branch in df_sales['Branch'].unique() if pd.notna(branch)])
    branch_options = [ALL_BRANCHES_OPTION] + unique_branches
    selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
    
    min_date = df_sales['Sales Date'].min().date()
    max_date = df_sales['Sales Date'].max().date()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2: st.stop()
    start_date, end_date = date_range

    # --- Bagian 4: Pemfilteran Data ---
    date_mask_sales = (df_sales['Sales Date'].dt.date >= start_date) & (df_sales['Sales Date'].dt.date <= end_date)
    df_sales_filtered = df_sales[date_mask_sales]
    
    date_mask_complaints = (df_complaints['Sales Date'].dt.date >= start_date) & (df_complaints['Sales Date'].dt.date <= end_date)
    df_complaints_filtered = df_complaints[date_mask_complaints]
    
    date_mask_qa_qc = (df_qa_qc['Sales Date'].dt.date >= start_date) & (df_qa_qc['Sales Date'].dt.date <= end_date)
    df_qa_qc_filtered = df_qa_qc[date_mask_qa_qc]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered['Branch'] == selected_branch]
        df_complaints_filtered = df_complaints_filtered[df_complaints_filtered['Branch'] == selected_branch]
        df_qa_qc_filtered = df_qa_qc_filtered[df_qa_qc_filtered['Branch'] == selected_branch]

    if df_sales_filtered.empty:
        st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
        st.stop()

    st.title(f"Dashboard Analisis Holistik: {selected_branch}")
    st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    # --- Bagian 5: Kalkulasi Semua Analisis ---
    with st.spinner("Menganalisis data... 🤖"):
        monthly_agg = analyze_monthly_trends(df_sales_filtered)
        price_group_results = calculate_price_group_analysis(df_sales_filtered)
        df_branch_health = calculate_branch_health(df_sales_filtered, df_complaints_filtered)
        
        # Agent hanya dijalankan satu kali dengan data lengkap untuk konteks historis
        if 'strategic_agent_results' not in st.session_state:
             st.session_state.strategic_agent_results = run_strategic_agent(df_sales, df_complaints, df_qa_qc)
    
    # --- Bagian 6: Tampilan Dashboard dengan Tab ---
    penjualan_tab, kualitas_tab, qa_qc_tab, agent_tab = st.tabs([
        "📈 **Performa Penjualan**",
        "✅ **Kualitas & Komplain**",
        "⭐ **Kepatuhan QA/QC**",
        "🤖 **AI Root Cause Agent**"
    ])
    
    with penjualan_tab:
        st.header("Analisis Tren Performa Penjualan")
        if monthly_agg is not None and not monthly_agg.empty:
            display_monthly_kpis(monthly_agg)
            st.markdown("---")
            display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
            display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
            display_trend_chart_and_analysis(monthly_agg, 'AOV', 'AOV', 'green')
        else:
            st.warning("Data bulanan tidak cukup untuk analisis tren.")
        st.markdown("---")
        display_price_group_analysis(price_group_results)

    with kualitas_tab:
        st.header("Analisis Kualitas Layanan dan Penanganan Komplain")
        display_branch_health(df_branch_health)
        st.markdown("---")
        display_complaint_analysis(df_complaints_filtered)
        
    with qa_qc_tab:
        st.header("Analisis Kepatuhan Standar Operasional")
        display_qa_qc_analysis(df_qa_qc_filtered)
        
    with agent_tab:
        display_agent_analysis(
            st.session_state.strategic_agent_results,
            title="Diagnosis Strategis Jangka Panjang",
            info_text="Agent ini menganalisis tren bulanan (minimal 4 bulan) secara statistik untuk mengidentifikasi pola fundamental dan kemungkinan akar masalah berdasarkan basis pengetahuan yang diekstrak dari data historis."
        )

# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================
# Untuk development, Anda bisa menonaktifkan autentikasi dan langsung memanggil main_app
# Cukup hapus tanda # dari baris di bawah dan berikan tanda # pada blok try-except
# main_app("Developer")

try:
    # Menggunakan st.secrets untuk konfigurasi
    if 'credentials' not in st.secrets or 'cookie' not in st.secrets:
        st.error("Konfigurasi autentikasi tidak ditemukan di secrets.toml.")
        st.info("Menjalankan aplikasi tanpa autentikasi untuk mode development.")
        main_app("Developer")
    else:
        config = {
            'credentials': dict(st.secrets['credentials']),
            'cookie': dict(st.secrets['cookie'])
        }
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
        name, auth_status, username = authenticator.login("Login", "main")
        
        if auth_status is False:
            st.error("Username atau password salah.")
        elif auth_status is None:
            st.warning("Silakan masukkan username dan password.")
        elif auth_status:
            main_app(name)
            
except Exception as e:
    st.error(f"Terjadi kesalahan saat inisialisasi autentikasi: {e}")
    st.info("Menjalankan aplikasi tanpa autentikasi.")
    main_app("Developer")