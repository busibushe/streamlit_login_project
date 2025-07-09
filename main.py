import os
os.environ["WATCHFILES_DISABLE_INOTIFY"] = "1"

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats

# ==============================================================================
# KONFIGURASI HALAMAN & FUNGSI-FUNGSI
# ==============================================================================

st.set_page_config(page_title="Dashboard Penjualan & Operasional", layout="wide")

# --- Fungsi-fungsi Bantuan ---

@st.cache_data
def load_data(file):
    """Memuat dan membersihkan data penjualan dari file yang diunggah."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Validasi dan rename kolom tanggal
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Sales Date'}, inplace=True)
        if 'Sales Date' not in df.columns:
            raise ValueError("Kolom 'Sales Date' atau 'Date' tidak ditemukan di file penjualan.")
        
        # Validasi kolom wajib lainnya
        required_sales_cols = ['Branch', 'Nett Sales', 'Bill Number', 'Menu', 'Qty']
        for col in required_sales_cols:
            if col not in df.columns:
                raise ValueError(f"Kolom wajib '{col}' tidak ditemukan di file penjualan.")

        # Proses data
        df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce').dt.date
        df['Branch'] = df['Branch'].fillna('Tidak Diketahui')
        st.write("Contoh data Sales Date:", df['Sales Date'].head())
        st.write("Tipe data Sales Date:", df['Sales Date'].dtype)
        st.write("Nilai min:", df['Sales Date'].min(), "| max:", df['Sales Date'].max())
        
        numeric_cols = ['Qty', 'Nett Sales', 'Bill Number']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Sales Date'] + numeric_cols, inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memproses file penjualan: {e}")

@st.cache_data
def load_qc_data(file):
    """Memuat dan membersihkan data QC dari file yang diunggah."""
    try:
        if file.name.endswith('.csv'):
            df_qc = pd.read_csv(file)
        else:
            df_qc = pd.read_excel(file)
        
        required_qc_cols = ['Branch', 'Month', 'QC Score']
        for col in required_qc_cols:
            if col not in df_qc.columns:
                raise ValueError(f"Kolom wajib '{col}' tidak ditemukan di file QC.")
        
        # Ubah 'Month' menjadi format periode bulanan agar bisa digabung
        df_qc['Bulan'] = pd.to_datetime(df_qc['Month'], errors='coerce').dt.to_period('M')
        df_qc.dropna(subset=['Bulan', 'QC Score'], inplace=True)
        return df_qc
    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memproses file QC: {e}")

def analyze_trend_v2(data_series, time_series):
    """Menganalisis tren (tidak berubah)."""
    data_series = data_series.dropna() # Pastikan tidak ada NaN
    if len(data_series) < 3: return "Data tidak cukup untuk analisis tren.", None, None
    if len(set(data_series)) == 1: return "Data konstan, tidak ada tren.", None, None
    x, y = np.arange(len(data_series)), data_series.values
    slope, _, _, p_value, _ = stats.linregress(x, y)
    trendline = slope * x + np.mean(y) - slope * np.mean(x)
    if p_value < 0.05:
        overall_trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    else:
        overall_trend = "cenderung **stabil/fluktuatif**"
    return overall_trend, trendline, p_value

def display_analysis_with_details(title, analysis_text, p_value):
    """Menampilkan analisis (tidak berubah)."""
    st.info(f"💡 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Lihat detail signifikansi (p-value)"):
            st.markdown(f"- **Nilai p-value:** `{p_value:.4f}`")
    st.markdown("---")

# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================

config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
name, auth_status, username = authenticator.login("Login", "main")

if not auth_status:
    if auth_status is False: st.error("Username atau password salah.")
    else: st.warning("Silakan masukkan username dan password.")
    st.stop()

# ==============================================================================
# APLIKASI UTAMA (SETELAH LOGIN BERHASIL)
# ==============================================================================
authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Login sebagai: {name}")
st.sidebar.title("Filter & Data")

# --- Unggah File ---
sales_file = st.sidebar.file_uploader("1. Unggah File Penjualan", type=["xlsx", "xls", "csv"])
qc_file = st.sidebar.file_uploader("2. (Opsional) Unggah File QC Score", type=["xlsx", "xls", "csv"])

if sales_file is None:
    st.info("Selamat datang! Silakan unggah file data penjualan untuk memulai analisis.")
    st.stop()

# --- Pemuatan & Pemrosesan Data ---
try:
    df = load_data(sales_file)
    df_qc = None
    if qc_file:
        df_qc = load_qc_data(qc_file)
except ValueError as e:
    st.error(e)
    st.stop()

# --- UI Filter ---
unique_branches = sorted(df['Branch'].unique())
selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
min_date = df['Sales Date'].min()
max_date = df['Sales Date'].max()
date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2: st.warning("Mohon pilih rentang tanggal yang valid."); st.stop()
start_date, end_date = date_range

# Filter data penjualan
df_filtered = df[(df['Branch'] == selected_branch) & (df['Sales Date'] >= start_date) & (df['Sales Date'] <= end_date)]
if df_filtered.empty: st.warning("Tidak ada data penjualan untuk filter yang Anda pilih."); st.stop()

# --- Agregasi & Penggabungan Data ---
monthly_df = df_filtered.copy()
monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
monthly_agg = monthly_df.groupby('Bulan').agg(
    TotalMonthlySales=('Nett Sales', 'sum'),
    TotalTransactions=('Bill Number', 'nunique'),
    DaysInMonth=('Sales Date', 'nunique') # Hitung jumlah hari penjualan
).reset_index()

if not monthly_agg.empty:
    monthly_agg['AvgDailySales'] = monthly_agg['TotalMonthlySales'] / monthly_agg['DaysInMonth']
    monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

    # --- PENGGABUNGAN DATA DENGAN QC SCORE (BARU) ---
    if df_qc is not None:
        qc_branch_data = df_qc[df_qc['Branch'] == selected_branch][['Bulan', 'QC Score']]
        monthly_agg = pd.merge(monthly_agg, qc_branch_data, on='Bulan', how='left')

    monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

# --- Tampilan Dashboard ---
st.title(f"📊 Dashboard Performa: {selected_branch}")
st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
st.markdown("---")

# Tampilan KPI
col1, col2, col3, col4 = st.columns(4) # Tambah satu kolom untuk KPI baru
if len(monthly_agg) >= 2:
    last_month = monthly_agg.iloc[-1]; prev_month = monthly_agg.iloc[-2]
    # Hitung delta KPI
    sales_delta = (last_month['AvgDailySales'] - prev_month['AvgDailySales']) / prev_month['AvgDailySales'] if prev_month['AvgDailySales'] > 0 else 0
    trx_delta = (last_month['TotalTransactions'] - prev_month['TotalTransactions']) / prev_month['TotalTransactions'] if prev_month['TotalTransactions'] > 0 else 0
    aov_delta = (last_month['AOV'] - prev_month['AOV']) / prev_month['AOV'] if prev_month['AOV'] > 0 else 0
    # Tampilkan KPI
    col1.metric("Avg Daily Sales", f"Rp {last_month['AvgDailySales']:,.0f}", f"{sales_delta:.1%}")
    col2.metric("Total Transactions", f"{last_month['TotalTransactions']:,}", f"{trx_delta:.1%}")
    col3.metric("AOV", f"Rp {last_month['AOV']:,.0f}", f"{aov_delta:.1%}")
    if 'QC Score' in last_month and pd.notna(last_month['QC Score']):
        col4.metric("QC Score", f"{last_month['QC Score']:.1f}")
elif not monthly_agg.empty:
    last_month = monthly_agg.iloc[-1]
    col1.metric("Avg Daily Sales", f"Rp {last_month['AvgDailySales']:,.0f}")
    col2.metric("Total Transactions", f"{last_month['TotalTransactions']:,}")
    col3.metric("AOV", f"Rp {last_month['AOV']:,.0f}")
    if 'QC Score' in last_month and pd.notna(last_month['QC Score']):
        col4.metric("QC Score", f"{last_month['QC Score']:.1f}")
st.markdown("---")

# Tampilan Visualisasi
col_chart, col_table = st.columns([2, 1])
with col_chart:
    if not monthly_agg.empty:
        # Grafik Rata-Rata Penjualan Harian
        st.subheader("Tren Rata-Rata Penjualan Harian")
        fig_sales = px.line(monthly_agg, x='Bulan', y='AvgDailySales', markers=True, labels={'Bulan': 'Bulan', 'AvgDailySales': 'Rata-Rata Penjualan Harian (Rp)'})
        st.plotly_chart(fig_sales, use_container_width=True)

        # Grafik Transaksi
        st.subheader("Tren Transaksi Bulanan")
        fig_trx = px.line(monthly_agg, x='Bulan', y='TotalTransactions', markers=True, labels={'Bulan': 'Bulan', 'TotalTransactions': 'Jumlah Transaksi'}).update_traces(line_color='orange')
        st.plotly_chart(fig_trx, use_container_width=True)

        # Grafik AOV
        st.subheader("Tren AOV Bulanan")
        fig_aov = px.line(monthly_agg, x='Bulan', y='AOV', markers=True, labels={'Bulan': 'Bulan', 'AOV': 'Average Order Value (Rp)'}).update_traces(line_color='green')
        st.plotly_chart(fig_aov, use_container_width=True)

        # --- GRAFIK BARU UNTUK QC SCORE ---
        if 'QC Score' in monthly_agg.columns:
            st.subheader("Tren QC Score Bulanan")
            fig_qc = px.line(monthly_agg.dropna(subset=['QC Score']), x='Bulan', y='QC Score', markers=True, labels={'Bulan': 'Bulan', 'QC Score': 'Skor QC'})
            fig_qc.update_traces(line_color='purple')
            st.plotly_chart(fig_qc, use_container_width=True)
            
with col_table:
    st.subheader("Menu Terlaris (berdasarkan Qty)")
    if 'Menu' in df_filtered.columns and 'Qty' in df_filtered.columns:
        top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
        st.dataframe(top_menus, use_container_width=True)