import os
os.environ["WATCHFILES_DISABLE_INOTIFY"] = "1"

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Penjualan Dinamis", layout="wide")

# ======================================================================
# Fungsi Bantuan
# ======================================================================

@st.cache_data
def load_raw_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, low_memory=False)
        else:
            return pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Tidak dapat membaca file. Pastikan formatnya benar. Error: {e}")

def process_data(df, column_mapping):
    rename_dict = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=rename_dict)
    df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce').dt.date
    df['Branch'] = df['Branch'].fillna('Tidak Diketahui').astype(str)
    numeric_cols = ['Nett Sales', 'Qty', 'Bill Number', 'Menu']
    for col in numeric_cols:
        if col in df.columns:
            if col in ['Nett Sales', 'Qty', 'Bill Number']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Sales Date', 'Nett Sales', 'Qty', 'Bill Number', 'Menu'], inplace=True)
    return df

def analyze_trend_v2(data_series, time_series):
    if len(data_series) < 3:
        return "Data tidak cukup untuk analisis tren.", None, None
    if len(set(data_series.dropna())) == 1:
        return "Data konstan, tidak ada tren.", None, None
    x, y = np.arange(len(data_series)), data_series.values
    slope, _, _, p_value, _ = stats.linregress(x, y)
    trendline = slope * x + np.mean(y) - slope * np.mean(x)
    if p_value < 0.05:
        trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    else:
        trend = "cenderung **stabil/fluktuatif**"
    return f"Data {trend}.", trendline, p_value

def display_analysis_with_details(title, analysis_text, p_value):
    st.info(f"\U0001F4A1 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Detail Statistik"):
            st.markdown(f"- **p-value:** `{p_value:.4f}`")
            if p_value < 0.05:
                st.success("✔️ Tren signifikan secara statistik.")
            else:
                st.warning("⚠️ Tren tidak signifikan secara statistik.")

# ======================================================================
# Autentikasi
# ======================================================================

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

if not auth_status:
    if auth_status is False:
        st.error("Username atau password salah.")
    else:
        st.warning("Silakan login.")
    st.stop()

authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Login sebagai: {name}")

# ======================================================================
# Upload & Validasi File
# ======================================================================

uploaded_file = st.sidebar.file_uploader("1. Unggah File Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.df_raw = None

        try:
            with st.spinner("Membaca file..."):
                st.session_state.df_raw = load_raw_data(uploaded_file)
        except ValueError as e:
            st.error(e)
            st.stop()

        st.success("File berhasil dimuat.")
        st.rerun()

if 'df_raw' not in st.session_state or st.session_state.df_raw is None:
    st.info("Unggah file penjualan untuk memulai.")
    st.stop()

# ======================================================================
# Pemetaan Kolom
# ======================================================================

if 'df_processed' not in st.session_state:
    st.sidebar.header("2. Pemetaan Kolom")
    df_columns = st.session_state.df_raw.columns.tolist()
    required_columns = {
        "Sales Date": "Tanggal transaksi",
        "Branch": "Nama cabang/toko",
        "Nett Sales": "Nilai penjualan",
        "Bill Number": "Nomor struk",
        "Menu": "Nama menu",
        "Qty": "Jumlah terjual"
    }

    with st.sidebar.form("mapping_form"):
        mapping = {key: st.selectbox(f"{key}", df_columns, help=val) for key, val in required_columns.items()}
        submit = st.form_submit_button("Terapkan")

    if submit:
        if len(set(mapping.values())) != len(mapping):
            st.error("Setiap kolom hanya boleh dipilih sekali.")
            st.stop()
        try:
            with st.spinner("Memproses data..."):
                st.session_state.df_processed = process_data(st.session_state.df_raw, mapping)
                st.success("Data berhasil diproses!")
                st.rerun()
        except Exception as e:
            st.error(f"Gagal memproses: {e}")
            st.stop()
    else:
        st.stop()

# ======================================================================
# Dashboard
# ======================================================================

df = st.session_state.df_processed
if df.empty:
    st.error("Tidak ada data valid setelah pemrosesan. Periksa pemetaan kolom.")
    st.session_state.clear()
    st.button("Coba Lagi")
    st.stop()

st.sidebar.header("3. Filter")
branches = sorted(df['Branch'].unique())
selected_branch = st.sidebar.selectbox("Cabang", branches)
min_date, max_date = df['Sales Date'].min(), df['Sales Date'].max()
date_range = st.sidebar.date_input("Rentang Tanggal", (min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) != 2:
    st.warning("Pilih rentang tanggal valid.")
    st.stop()

start_date, end_date = date_range
df_filtered = df[(df['Branch'] == selected_branch) & (df['Sales Date'] >= start_date) & (df['Sales Date'] <= end_date)]
if df_filtered.empty:
    st.warning("Tidak ada data sesuai filter.")
    st.stop()

st.title(f"📊 Dashboard: {selected_branch}")
st.markdown(f"Periode: **{start_date.strftime('%d %B %Y')}** - **{end_date.strftime('%d %B %Y')}**")

monthly_df = df_filtered.copy()
monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
monthly_agg = monthly_df.groupby('Bulan').agg(
    TotalMonthlySales=('Nett Sales', 'sum'),
    TotalTransactions=('Bill Number', 'nunique')
).reset_index()

if not monthly_agg.empty:
    monthly_agg['AOV'] = monthly_agg['TotalMonthlySales'] / monthly_agg['TotalTransactions']
    monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

    col1, col2, col3 = st.columns(3)
    last, prev = monthly_agg.iloc[-1], monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else (None, None)
    if prev is not None:
        col1.metric("💰 Penjualan Terakhir", f"Rp {last['TotalMonthlySales']:,.0f}", f"{(last['TotalMonthlySales'] - prev['TotalMonthlySales']) / prev['TotalMonthlySales']:.1%}")
        col2.metric("🛒 Transaksi Terakhir", f"{last['TotalTransactions']:,.0f}", f"{(last['TotalTransactions'] - prev['TotalTransactions']) / prev['TotalTransactions']:.1%}")
        col3.metric("💳 AOV Terakhir", f"Rp {last['AOV']:,.0f}", f"{(last['AOV'] - prev['AOV']) / prev['AOV']:.1%}")
    else:
        col1.metric("💰 Penjualan Terakhir", f"Rp {last['TotalMonthlySales']:,.0f}")
        col2.metric("🛒 Transaksi Terakhir", f"{last['TotalTransactions']:,.0f}")
        col3.metric("💳 AOV Terakhir", f"Rp {last['AOV']:,.0f}")

    st.markdown("---")

    with st.container():
        fig_sales = px.line(monthly_agg, x='Bulan', y='TotalMonthlySales', title="Tren Penjualan", markers=True)
        desc, trendline, pval = analyze_trend_v2(monthly_agg['TotalMonthlySales'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
        if trendline is not None:
            fig_sales.add_scatter(x=monthly_agg['Bulan'], y=trendline, mode='lines', name='Garis Tren')
        st.plotly_chart(fig_sales, use_container_width=True)
        display_analysis_with_details("Tren Penjualan", desc, pval)

        fig_trx = px.line(monthly_agg, x='Bulan', y='TotalTransactions', title="Tren Transaksi", markers=True)
        desc, trendline, pval = analyze_trend_v2(monthly_agg['TotalTransactions'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
        if trendline is not None:
            fig_trx.add_scatter(x=monthly_agg['Bulan'], y=trendline, mode='lines', name='Garis Tren')
        st.plotly_chart(fig_trx, use_container_width=True)
        display_analysis_with_details("Tren Transaksi", desc, pval)

        fig_aov = px.line(monthly_agg, x='Bulan', y='AOV', title="Tren AOV", markers=True)
        desc, trendline, pval = analyze_trend_v2(monthly_agg['AOV'], monthly_agg['Bulan'].dt.strftime('%b %Y'))
        if trendline is not None:
            fig_aov.add_scatter(x=monthly_agg['Bulan'], y=trendline, mode='lines', name='Garis Tren')
        st.plotly_chart(fig_aov, use_container_width=True)
        display_analysis_with_details("Tren AOV", desc, pval)

    st.markdown("---")

    st.subheader("Menu Terlaris")
    if 'Menu' in df_filtered.columns and 'Qty' in df_filtered.columns:
        top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
        top_menus.index = top_menus.index + 1
        st.dataframe(top_menus, use_container_width=True)
else:
    st.warning("Tidak ada data bulanan untuk ditampilkan.")
