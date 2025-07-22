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
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}")
        return None

# ==============================================================================
# FUNGSI-FUNGSI ANALISIS & VISUALISASI
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

# FUNGSI UNTUK 
def display_qa_qc_analysis(df_qa_qc):
    """Menampilkan analisis data audit QA/QC."""
    st.subheader("⭐ Dashboard Kepatuhan Standar (QA/QC)")
    
    if df_qa_qc.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih.")
        return
    
    # Tampilkan skor rata-rata sebagai KPI utama
    avg_score = df_qa_qc['Skor Kepatuhan'].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")

    # Tampilkan tren skor kepatuhan dari waktu ke waktu
    st.markdown("##### Tren Skor Kepatuhan per Audit")
    fig1 = px.line(
        df_qa_qc.sort_values('Sales Date'), 
        x='Sales Date', 
        y='Skor Kepatuhan',
        color='Branch',
        markers=True,
        title="Tren Skor Kepatuhan per Audit"
    )
    fig1.update_yaxes(range=[0, 105]) # Set sumbu Y dari 0-105%
    st.plotly_chart(fig1, use_container_width=True)

    # Tampilkan perbandingan skor rata-rata antar cabang
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
# APLIKASI UTAMA STREAMLIT
# ==============================================================================
def old_main_app(user_name):
    """Fungsi utama yang menjalankan seluruh aplikasi dashboard."""

    # Sidebar: Autentikasi dan Unggah
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{user_name}**")
    st.sidebar.title("📤 Unggah Data Master")

    # UPLOAD DUA FILE MASTER
    sales_file = st.sidebar.file_uploader("1. Unggah Penjualan Master (.feather)", type=["feather"])
    complaint_file = st.sidebar.file_uploader("2. Unggah Komplain Master (.feather)", type=["feather"])

    if sales_file is None or complaint_file is None:
        st.info("👋 Selamat datang! Silakan unggah file `penjualan_master.feather` dan `komplain_master.feather`.")
        st.stop()
        
    # Memuat data
    df_sales = load_feather_file(sales_file)
    df_complaints = load_feather_file(complaint_file)
    
    if df_sales is None or df_complaints is None:
        st.error("Gagal memuat salah satu atau kedua file data.")
        st.stop()
        
    # Simpan ke session_state agar tidak perlu di-reload
    st.session_state.df_sales = df_sales
    st.session_state.df_complaints = df_complaints

    # Sidebar: Filter Global
    st.sidebar.title("⚙️ Filter Global")
    
    ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
    # PERBAIKAN: Gunakan df_sales dan tangani nilai kosong sebelum diurutkan
    unique_branches = sorted([str(branch) for branch in df_sales['Branch'].unique() if pd.notna(branch)])
    branch_options = [ALL_BRANCHES_OPTION] + unique_branches
    selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
    
    min_date = df_sales['Sales Date'].min().date()
    max_date = df_sales['Sales Date'].max().date()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2: st.stop()
    start_date, end_date = date_range

    # Menerapkan filter ke kedua DataFrame
    date_mask_sales = (df_sales['Sales Date'].dt.date >= start_date) & (df_sales['Sales Date'].dt.date <= end_date)
    df_sales_filtered = df_sales[date_mask_sales]
    
    date_mask_complaints = (df_complaints['Sales Date'].dt.date >= start_date) & (df_complaints['Sales Date'].dt.date <= end_date)
    df_complaints_filtered = df_complaints[date_mask_complaints]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered['Branch'] == selected_branch]
        df_complaints_filtered = df_complaints_filtered[df_complaints_filtered['Branch'] == selected_branch]

    if df_sales_filtered.empty:
        st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
        st.stop()

    st.title(f"Dashboard Analisis Holistik: {selected_branch}")
    st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    # --- Kalkulasi Analisis ---
    monthly_agg = analyze_monthly_trends(df_sales_filtered)
    price_group_results = calculate_price_group_analysis(df_sales_filtered)
    df_branch_health = calculate_branch_health(df_sales_filtered, df_complaints_filtered)
    
    # --- Tata Letak Tab ---
    penjualan_tab, kualitas_tab = st.tabs([
        "📈 **Dashboard Performa Penjualan**", 
        "✅ **Dashboard Kualitas & Komplain**"
    ])
    
    with penjualan_tab:
        st.header("Analisis Tren Performa Jangka Panjang")
        if monthly_agg is not None and not monthly_agg.empty:
            display_monthly_kpis(monthly_agg)
            display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
            display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
            display_trend_chart_and_analysis(monthly_agg, 'AOV', 'AOV', 'green')
        else:
            st.warning("Tidak ada data bulanan untuk analisis tren.")
        
        st.markdown("---")
        display_price_group_analysis(price_group_results)

    with kualitas_tab:
        st.header("Analisis Kualitas Layanan dan Penanganan Komplain")
        display_branch_health(df_branch_health)
        st.markdown("---")
        display_complaint_analysis(df_complaints_filtered)

# GANTI FUNGSI main_app ANDA DENGAN VERSI LENGKAP INI

def main_app(user_name):
    """Fungsi utama yang menjalankan seluruh aplikasi dashboard."""

    # Sidebar: Autentikasi dan Unggah
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{user_name}**")
    st.sidebar.title("📤 Unggah Data Master")

    # UPLOAD TIGA FILE MASTER
    sales_file = st.sidebar.file_uploader("1. Unggah Penjualan Master (.feather)", type=["feather"])
    complaint_file = st.sidebar.file_uploader("2. Unggah Komplain Master (.feather)", type=["feather"])
    qa_qc_file = st.sidebar.file_uploader("3. Unggah QA/QC Master (.feather)", type=["feather"])

    if sales_file is None or complaint_file is None or qa_qc_file is None:
        st.info("👋 Selamat datang! Silakan unggah ketiga file master: penjualan, komplain, dan QA/QC.")
        st.stop()
        
    # Memuat data
    df_sales = load_feather_file(sales_file)
    df_complaints = load_feather_file(complaint_file)
    df_qa_qc = load_feather_file(qa_qc_file)
    
    if df_sales is None or df_complaints is None or df_qa_qc is None:
        st.error("Gagal memuat salah satu dari tiga file data.")
        st.stop()
        
    st.session_state.df_sales = df_sales
    st.session_state.df_complaints = df_complaints
    st.session_state.df_qa_qc = df_qa_qc

    # Sidebar: Filter Global
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

    # Menerapkan filter ke semua DataFrame
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

    # --- Kalkulasi Analisis ---
    monthly_agg = analyze_monthly_trends(df_sales_filtered)
    price_group_results = calculate_price_group_analysis(df_sales_filtered)
    df_branch_health = calculate_branch_health(df_sales_filtered, df_complaints_filtered)
    
    # --- Tata Letak Tab ---
    penjualan_tab, kualitas_tab, qa_qc_tab = st.tabs([
        "📈 **Dashboard Performa Penjualan**", 
        "✅ **Dashboard Kualitas & Komplain**",
        "⭐ **Dashboard Kepatuhan QA/QC**"
    ])
    
    with penjualan_tab:
        st.header("Analisis Tren Performa Penjualan")
        if monthly_agg is not None and not monthly_agg.empty:
            display_monthly_kpis(monthly_agg)
            display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
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
# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================
try:
    # Ganti dengan path ke file config.yaml Anda jika menggunakan authenticator
    # Untuk development, Anda bisa menonaktifkan ini dan langsung memanggil main_app
    # main_app("Developer")
    
    config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
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
        
except KeyError as e:
    st.error(f"❌ Kesalahan Konfigurasi 'secrets.toml': Key {e} tidak ditemukan.")
    st.info("Pastikan file secrets Anda memiliki struktur yang benar.")
except Exception as e:
    st.error(f"Terjadi kesalahan tak terduga saat inisialisasi: {e}")