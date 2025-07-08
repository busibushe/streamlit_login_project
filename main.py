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

# 1. Pindahkan st.set_page_config() ke bagian paling atas
st.set_page_config(page_title="Dashboard Penjualan", layout="wide")

# 2. Definisikan semua fungsi di sini, di luar blok autentikasi

@st.cache_data
def load_data(file):
    """
    Memuat dan membersihkan data dari file yang diunggah.
    Fungsi ini sekarang "murni" dan tidak mengandung perintah UI (st.error).
    Jika terjadi kesalahan, fungsi akan melempar ValueError.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Validasi dan rename kolom tanggal
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Sales Date'}, inplace=True)
        if 'Sales Date' not in df.columns:
            raise ValueError("Kolom 'Sales Date' atau 'Date' tidak ditemukan di dalam file.")

        # Validasi kolom Branch
        if 'Branch' not in df.columns:
            raise ValueError("Kolom 'Branch' tidak ditemukan di dalam file.")

        # Proses data
        df['Sales Date'] = pd.to_datetime(df['Sales Date']).dt.date
        df['Branch'] = df['Branch'].fillna('Tidak Diketahui')

        numeric_cols = ['Qty', 'Price', 'Subtotal', 'Discount', 'Service Charge', 'Tax', 'VAT', 'Total', 'Nett Sales', 'Bill Discount', 'Total After Bill Discount']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

        return df

    except Exception as e:
        # Melempar kembali error yang lebih spesifik untuk ditangkap nanti
        raise ValueError(f"Terjadi kesalahan saat memproses file: {e}")


def analyze_trend_v2(data_series, time_series):
    """
    Menganalisis tren dan mengembalikan narasi, garis tren, dan p-value untuk penjelasan.
    """
    if len(data_series) < 3:
        return "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan).", None, None
    
    # Mencegah error jika semua nilai y sama
    if len(set(data_series.dropna())) == 1:
        return "Data konstan, tidak ada tren yang bisa dianalisis.", None, None

    x = np.arange(len(data_series))
    y = data_series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trendline = slope * x + intercept

    if p_value < 0.05:
        overall_trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan (statistik)"
    else:
        overall_trend = "cenderung **stabil/fluktuatif** tanpa tren yang jelas secara statistik"

    monthly_changes = data_series.pct_change().dropna()
    event_info = ""
    if not monthly_changes.empty:
        std_dev_changes = monthly_changes.std()
        if std_dev_changes > 0: # Pastikan ada variasi
            significant_change_threshold = 1.5 * std_dev_changes
            max_increase_month = monthly_changes.idxmax()
            max_increase_value = monthly_changes.max()
            max_decrease_month = monthly_changes.idxmin()
            max_decrease_value = monthly_changes.min()

            if max_increase_value > significant_change_threshold:
                event_info += f" Terjadi **lonjakan** tertinggi pada **{time_series[max_increase_month]}**."
            if abs(max_decrease_value) > significant_change_threshold:
                event_info += f" Terjadi **penurunan tertajam** pada **{time_series[max_decrease_month]}**."
    
    narrative = f"Secara keseluruhan, data {overall_trend}.{event_info}"
    return narrative, trendline, p_value


def display_analysis_with_details(title, analysis_text, p_value):
    """Menampilkan analisis utama dan penjelasan p-value dalam expander."""
    st.info(f"💡 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Mengapa tren ini dianggap signifikan/tidak signifikan? Klik untuk lihat penjelasan p-value."):
            st.markdown(f"""
            - **Nilai p-value tren ini adalah `{p_value:.4f}`**.
            - **Analogi:** Angka ini berarti ada **`{p_value:.2%}` kemungkinan** untuk melihat pola data sekuat ini murni hanya karena **kebetulan**.
            """)
            if p_value < 0.05:
                st.success("✔️ **Kesimpulan:** Karena kemungkinan kebetulan ini **sangat rendah** (di bawah 5%), kita bisa yakin bahwa tren yang terlihat adalah **nyata secara statistik**.")
            else:
                st.warning("⚠️ **Kesimpulan:** Karena kemungkinan ini **cukup tinggi**, kita **tidak bisa yakin** bahwa tren ini nyata. Bisa jadi ini hanya fluktuasi acak.")
    st.markdown("---")

# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================

# Baca konfigurasi dari secrets.toml
config = {
    'credentials': st.secrets['credentials'].to_dict(),
    'cookie': st.secrets['cookie'].to_dict()
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
    st.stop()
elif auth_status:
    # ==============================================================================
    # APLIKASI UTAMA (SETELAH LOGIN BERHASIL)
    # ==============================================================================
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: {name}")

    st.sidebar.title("Filter Data")
    uploaded_file = st.sidebar.file_uploader("📂 Unggah file Excel atau CSV Anda", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is None:
        st.info("Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
        st.stop()
    
    # 3. Tangkap error dari load_data di sini, di alur utama aplikasi
    try:
        df = load_data(uploaded_file)
    except ValueError as e:
        st.error(e)
        st.stop()

    # --- UI Filter ---
    unique_branches = sorted(df['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
    min_date = df['Sales Date'].min()
    max_date = df['Sales Date'].max()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2:
        st.warning("Mohon pilih rentang tanggal yang valid.")
        st.stop()
    
    start_date, end_date = date_range
    df_filtered = df[(df['Branch'] == selected_branch) & (df['Sales Date'] >= start_date) & (df['Sales Date'] <= end_date)]
    
    if df_filtered.empty:
        st.warning("Tidak ada data yang ditemukan untuk filter yang Anda pilih.")
        st.stop()

    # --- Tampilan Dashboard ---
    st.title(f"📊 Dashboard Performa: {selected_branch}")
    st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
    st.markdown("---")

    # Agregasi Bulanan
    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(
        TotalMonthlySales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

    # Tampilan KPI
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

    # Tampilan Visualisasi
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        if not monthly_agg.empty:
            st.subheader("Tren Penjualan Bulanan")
            fig_sales = px.line(monthly_agg, x='Bulan', y='TotalMonthlySales', markers=True, labels={'Bulan': 'Bulan', 'TotalMonthlySales': 'Total Penjualan (Rp)'})
            sales_analysis, sales_trendline, sales_p_value = analyze_trend_v2(monthly_agg['TotalMonthlySales'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if sales_trendline is not None:
                fig_sales.add_scatter(x=monthly_agg['Bulan'], y=sales_trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_sales, use_container_width=True)
            display_analysis_with_details("Analisis Tren Penjualan", sales_analysis, sales_p_value)

            st.subheader("Tren Transaksi Bulanan")
            fig_trx = px.line(monthly_agg, x='Bulan', y='TotalTransactions', markers=True, labels={'Bulan': 'Bulan', 'TotalTransactions': 'Jumlah Transaksi'})
            fig_trx.update_traces(line_color='orange')
            trx_analysis, trx_trendline, trx_p_value = analyze_trend_v2(monthly_agg['TotalTransactions'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if trx_trendline is not None:
                fig_trx.add_scatter(x=monthly_agg['Bulan'], y=trx_trendline, mode='lines', name='Gris Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_trx, use_container_width=True)
            display_analysis_with_details("Analisis Tren Transaksi", trx_analysis, trx_p_value)

            st.subheader("Tren AOV Bulanan")
            fig_aov = px.line(monthly_agg, x='Bulan', y='AOV', markers=True, labels={'Bulan': 'Bulan', 'AOV': 'Average Order Value (Rp)'})
            fig_aov.update_traces(line_color='green')
            aov_analysis, aov_trendline, aov_p_value = analyze_trend_v2(monthly_agg['AOV'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
            if aov_trendline is not None:
                fig_aov.add_scatter(x=monthly_agg['Bulan'], y=aov_trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_aov, use_container_width=True)
            display_analysis_with_details("Analisis Tren AOV", aov_analysis, aov_p_value)
        else:
            st.warning("Tidak ada data bulanan untuk divisualisasikan pada rentang waktu ini.")

    with col_table:
        st.subheader("Menu Terlaris (berdasarkan Qty)")
        top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
        top_menus.index = top_menus.index + 1
        st.dataframe(top_menus, use_container_width=True)