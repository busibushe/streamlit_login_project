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

st.set_page_config(page_title="Dashboard Penjualan Dinamis", layout="wide")

# --- Fungsi-fungsi Bantuan ---

@st.cache_data
def load_raw_data(file):
    """Hanya memuat data mentah dari file tanpa proses apa pun."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        raise ValueError(f"Tidak dapat membaca file. Pastikan formatnya benar. Error: {e}")

def process_data(df, column_mapping):
    """
    Memproses DataFrame mentah setelah pemetaan kolom: mengganti nama,
    mengonversi tipe data, dan membersihkan.
    """
    rename_dict = {v: k for k, v in column_mapping.items()}
    df_processed = df.rename(columns=rename_dict)
    
    df_processed['Sales Date'] = pd.to_datetime(df_processed['Sales Date'], errors='coerce').dt.date
    df_processed['Branch'] = df_processed['Branch'].fillna('Tidak Diketahui').astype(str)
    
    numeric_cols = ['Nett Sales', 'Qty', 'Bill Number', 'Menu']
    for col in numeric_cols:
        if col in df_processed.columns:
            if col in ['Nett Sales', 'Qty', 'Bill Number']:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.dropna(subset=['Sales Date', 'Nett Sales', 'Qty', 'Bill Number', 'Menu'], inplace=True)
    return df_processed

def analyze_trend_v2(data_series, time_series):
    """Menganalisis tren (tidak berubah)."""
    if len(data_series) < 3: return "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan).", None, None
    if len(set(data_series.dropna())) == 1: return "Data konstan, tidak ada tren yang bisa dianalisis.", None, None
    x, y = np.arange(len(data_series)), data_series.values
    slope, _, _, p_value, _ = stats.linregress(x, y)
    trendline = slope * x + np.mean(y) - slope * np.mean(x)
    if p_value < 0.05:
        overall_trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan (statistik)"
    else:
        overall_trend = "cenderung **stabil/fluktuatif** tanpa tren yang jelas secara statistik"
    monthly_changes = data_series.pct_change().dropna()
    event_info = ""
    if not monthly_changes.empty and monthly_changes.std() > 0:
        std_dev_changes = monthly_changes.std()
        significant_change_threshold = 1.5 * std_dev_changes
        if monthly_changes.max() > significant_change_threshold: event_info += f" Terjadi **lonjakan** tertinggi pada **{time_series[monthly_changes.idxmax()]}**."
        if abs(monthly_changes.min()) > significant_change_threshold: event_info += f" Terjadi **penurunan tertajam** pada **{time_series[monthly_changes.idxmin()]}**."
    return f"Secara keseluruhan, data {overall_trend}.{event_info}", trendline, p_value

def display_analysis_with_details(title, analysis_text, p_value):
    """Menampilkan analisis (tidak berubah)."""
    st.info(f"💡 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Lihat detail signifikansi (p-value)"):
            st.markdown(f"- **Nilai p-value:** `{p_value:.4f}`. Ini berarti ada **`{p_value:.2%}` kemungkinan** melihat tren sekuat ini hanya karena kebetulan.")
            if p_value < 0.05: st.success("✔️ **Kesimpulan:** Karena kemungkinan ini sangat rendah, kita yakin tren ini **nyata secara statistik**.")
            else: st.warning("⚠️ **Kesimpulan:** Karena kemungkinan ini cukup tinggi, kita **tidak bisa yakin** tren ini nyata.")
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
st.sidebar.title("Pengaturan Data")

# --- Tahap 1: Unggah File ---
uploaded_file = st.sidebar.file_uploader("1. Unggah File Excel/CSV Anda", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name
        try:
            with st.spinner("Membaca file..."):
                st.session_state.df_raw = load_raw_data(uploaded_file)
        except ValueError as e:
            st.error(e)
            st.stop()
        st.rerun()

# --- Alur Aplikasi Berbasis Session State ---
if 'df_raw' not in st.session_state:
    st.info("Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
    st.stop()

if 'df_processed' not in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Pemetaan Kolom")
    st.sidebar.info("Cocokkan kolom dari file Anda dengan kolom yang dibutuhkan aplikasi.")
    
    df_columns = st.session_state.df_raw.columns.tolist()
    required_columns = {
        "Sales Date": "Kolom berisi tanggal transaksi", "Branch": "Kolom berisi nama cabang/toko",
        "Nett Sales": "Kolom berisi nilai penjualan bersih", "Bill Number": "Kolom berisi nomor unik transaksi/struk",
        "Menu": "Kolom berisi nama item/menu", "Qty": "Kolom berisi kuantitas item terjual"
    }
    
    with st.sidebar.form(key='mapping_form'):
        column_mapping = {std_col: st.selectbox(f"Pilih kolom untuk **{std_col}**", df_columns, help=help_text) for std_col, help_text in required_columns.items()}
        submit_button = st.form_submit_button(label='Terapkan Pemetaan & Proses Data')

    if submit_button:
        if len(set(column_mapping.values())) != len(required_columns):
            st.error("Setiap kolom dari file Anda hanya boleh dipetakan satu kali.")
        else:
            with st.spinner("Memproses dan membersihkan data..."):
                try:
                    st.session_state.df_processed = process_data(st.session_state.df_raw, column_mapping)
                    st.success("Data berhasil diproses!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal memproses data: {e}")
    else:
        st.info("Silakan lakukan pemetaan kolom di sidebar kiri dan klik tombol 'Terapkan' untuk melanjutkan.")
        st.stop()

# --- Tahap 3: Tampilkan Dashboard ---
if 'df_processed' in st.session_state:
    df = st.session_state.df_processed
    
    # --- PERBAIKAN KRUSIAL DI SINI ---
    # Tambahkan pengecekan apakah DataFrame kosong SETELAH diproses
    # Ini adalah pintu pengaman untuk mencegah error pada st.date_input
    if df.empty:
        st.error(
            "Tidak ada data yang valid ditemukan setelah pemrosesan. "
            "Ini bisa terjadi jika kolom yang Anda petakan untuk 'Sales Date' tidak berisi format tanggal yang bisa dikenali atau kolom wajib lainnya kosong. "
            "Mohon periksa kembali file atau pemetaan kolom Anda."
        )
        # Hapus state agar pengguna bisa mencoba lagi dengan file/mapping baru
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.button("Coba Lagi")
        st.stop()
    # ------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.header("3. Filter Dashboard")
    
    unique_branches = sorted(df['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
    min_date = df['Sales Date'].min()
    max_date = df['Sales Date'].max()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2: st.warning("Mohon pilih rentang tanggal yang valid."); st.stop()
    start_date, end_date = date_range
    df_filtered = df[(df['Branch'] == selected_branch) & (df['Sales Date'] >= start_date) & (df['Sales Date'] <= end_date)]
    if df_filtered.empty: st.warning("Tidak ada data yang ditemukan untuk filter yang Anda pilih."); st.stop()

    st.title(f"📊 Dashboard Performa: {selected_branch}")
    st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
    st.markdown("---")

    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(
        TotalMonthlySales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

    col1, col2, col3 = st.columns(3)
    if len(monthly_agg) >= 2:
        last_month = monthly_agg.iloc[-1]
        prev_month = monthly_agg.iloc[-2]
        sales_delta = (last_month['TotalMonthlySales'] - prev_month['TotalMonthlySales']) / prev_month['TotalMonthlySales']
        trx_delta = (last_month['TotalTransactions'] - prev_month['TotalTransactions']) / prev_month['TotalTransactions']
        aov_delta = (last_month['AOV'] - prev_month['AOV']) / prev_month['AOV']
        col1.metric("💰 Penjualan Bulan Terakhir", f"Rp {last_month['TotalMonthlySales']:,.0f}", f"{sales_delta:.1%}", help=f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}")
        col2.metric("🛒 Transaksi Bulan Terakhir", f"{last_month['TotalTransactions']:,}", f"{trx_delta:.1%}", help=f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}")
        col3.metric("💳 AOV Bulan Terakhir", f"Rp {last_month['AOV']:,.0f}", f"{aov_delta:.1%}", help=f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}")
    elif not monthly_agg.empty:
        last_month = monthly_agg.iloc[-1]
        col1.metric("💰 Penjualan Bulan Terakhir", f"Rp {last_month['TotalMonthlySales']:,.0f}")
        col2.metric("🛒 Transaksi Bulan Terakhir", f"{last_month['TotalTransactions']:,}")
        col3.metric("💳 AOV Bulan Terakhir", f"Rp {last_month['AOV']:,.0f}")
    st.markdown("---")

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        if not monthly_agg.empty:
            st.subheader("Tren Penjualan Bulanan")
            fig_sales = px.line(monthly_agg, x='Bulan', y='TotalMonthlySales', markers=True, labels={'Bulan': 'Bulan', 'TotalMonthlySales': 'Total Penjualan (Rp)'})
            sales_analysis, sales_trendline, sales_p_value = analyze_trend_v2(monthly_agg['TotalMonthlySales'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if sales_trendline is not None: fig_sales.add_scatter(x=monthly_agg['Bulan'], y=sales_trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_sales, use_container_width=True)
            display_analysis_with_details("Analisis Tren Penjualan", sales_analysis, sales_p_value)
            
            st.subheader("Tren Transaksi Bulanan")
            fig_trx = px.line(monthly_agg, x='Bulan', y='TotalTransactions', markers=True, labels={'Bulan': 'Bulan', 'TotalTransactions': 'Jumlah Transaksi'})
            fig_trx.update_traces(line_color='orange')
            trx_analysis, trx_trendline, trx_p_value = analyze_trend_v2(monthly_agg['TotalTransactions'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if trx_trendline is not None: fig_trx.add_scatter(x=monthly_agg['Bulan'], y=trx_trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_trx, use_container_width=True)
            display_analysis_with_details("Analisis Tren Transaksi", trx_analysis, trx_p_value)

            st.subheader("Tren AOV Bulanan")
            fig_aov = px.line(monthly_agg, x='Bulan', y='AOV', markers=True, labels={'Bulan': 'Bulan', 'AOV': 'Average Order Value (Rp)'})
            fig_aov.update_traces(line_color='green')
            aov_analysis, aov_trendline, aov_p_value = analyze_trend_v2(monthly_agg['AOV'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if aov_trendline is not None: fig_aov.add_scatter(x=monthly_agg['Bulan'], y=aov_trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_aov, use_container_width=True)
            display_analysis_with_details("Analisis Tren AOV", aov_analysis, aov_p_value)
        else:
            st.warning("Tidak ada data bulanan untuk divisualisasikan pada rentang waktu ini.")
            
    with col_table:
        st.subheader("Menu Terlaris (berdasarkan Qty)")
        if 'Menu' in df_filtered.columns and 'Qty' in df_filtered.columns:
            top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
            top_menus.index = top_menus.index + 1
            st.dataframe(top_menus, use_container_width=True)
        else:
            st.warning("Kolom 'Menu' atau 'Qty' tidak tersedia untuk ditampilkan.")