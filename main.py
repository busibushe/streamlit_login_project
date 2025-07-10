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

st.set_page_config(
    page_title="Dashboard F&B Holistik", 
    page_icon="🚀",
    layout="wide"
)

# --- FUNGSI PEMUATAN & PEMROSESAN DATA (Tidak diubah) ---
@st.cache_data
def load_raw_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
        return df.copy()
    except Exception as e:
        raise ValueError(f"Gagal memuat file: {e}")

@st.cache_data
def process_additional_data(df_add_raw, user_mapping):
    # ... (Fungsi ini tidak diubah)
    pass

# --- FUNGSI ANALISIS (Tidak diubah) ---
def analyze_trend_v3(df_monthly, metric_col, metric_label):
    # ... (Fungsi ini tidak diubah)
    pass
    
def display_analysis_with_details_v3(title, analysis_result):
    # ... (Fungsi ini tidak diubah)
    pass

# ==============================================================================
# FUNGSI-FUNGSI BARU UNTUK DASBOR OPERASIONAL
# ==============================================================================

def create_channel_analysis(df):
    st.subheader("📊 Analisis Saluran Penjualan (Channel)")
    if 'Visit Purpose' not in df.columns:
        st.warning("Kolom 'Visit Purpose' tidak dipetakan. Analisis saluran penjualan tidak tersedia.")
        return

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Saluran Penjualan", df['Visit Purpose'].nunique())
        channel_sales = df.groupby('Visit Purpose')['Nett Sales'].sum().sort_values(ascending=False)
        fig = px.pie(channel_sales, values='Nett Sales', names=channel_sales.index, title="Kontribusi Penjualan per Saluran", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        aov_by_channel = df.groupby('Visit Purpose').apply(lambda x: x['Nett Sales'].sum() / x['Bill Number'].nunique()).sort_values(ascending=False)
        fig2 = px.bar(aov_by_channel, y=aov_by_channel.index, x=aov_by_channel.values, orientation='h', 
                      labels={'x': 'Rata-rata Nilai Pesanan (AOV)', 'y': 'Saluran'}, title="AOV per Saluran Penjualan")
        st.plotly_chart(fig2, use_container_width=True)

def create_menu_engineering_chart(df):
    st.subheader("🔬 Analisis Performa Menu (Menu Engineering)")
    menu_perf = df.groupby('Menu').agg(
        Qty=('Qty', 'sum'),
        NettSales=('Nett Sales', 'sum')
    ).reset_index()

    if len(menu_perf) < 2:
        st.warning("Data menu tidak cukup untuk analisis engineering.")
        return

    avg_qty = menu_perf['Qty'].mean()
    avg_sales = menu_perf['NettSales'].mean()

    fig = px.scatter(menu_perf, x='Qty', y='NettSales', text='Menu', title="Kuadran Performa Menu",
                     labels={'Qty': 'Total Kuantitas Terjual', 'NettSales': 'Total Penjualan Bersih (Rp)'},
                     size='NettSales', color='NettSales', hover_name='Menu')
    
    fig.add_vline(x=avg_qty, line_dash="dash", line_color="gray", annotation_text="Rata-rata Qty")
    fig.add_hline(y=avg_sales, line_dash="dash", line_color="gray", annotation_text="Rata-rata Sales")
    
    fig.update_traces(textposition='top_center')
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
    **Cara Membaca Kuadran:**
    - **Kanan Atas (STARS 🌟):** Juara Anda! Populer dan menguntungkan. Promosikan!
    - **Kanan Bawah (WORKHORSES 🐴):** Populer tapi kurang profit. Naikkan harga atau tawarkan paket bundling.
    - **Kiri Atas (PUZZLES 🤔):** Sangat profit tapi jarang dipesan. Latih staf untuk merekomendasikan.
    - **Kiri Bawah (DOGS 🐶):** Kurang populer & profit. Pertimbangkan untuk menghapus dari menu.
    """)

def create_operational_efficiency_analysis(df):
    st.subheader("⏱️ Analisis Efisiensi Operasional")
    if 'Sales Date In' not in df.columns or 'Sales Date Out' not in df.columns or 'Order Time' not in df.columns:
        st.warning("Kolom 'Sales Date In', 'Sales Date Out', atau 'Order Time' tidak dipetakan. Analisis efisiensi tidak tersedia.")
        return
    
    # Konversi ke datetime dengan penanganan error
    df['Sales Date In'] = pd.to_datetime(df['Sales Date In'], errors='coerce')
    df['Sales Date Out'] = pd.to_datetime(df['Sales Date Out'], errors='coerce')
    df.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
    
    df['Prep Time (Seconds)'] = (df['Sales Date Out'] - df['Sales Date In']).dt.total_seconds()
    # Filter waktu persiapan yang tidak wajar (misal, > 1 jam atau < 0)
    df = df[df['Prep Time (Seconds)'].between(0, 3600)]

    if df.empty:
        st.warning("Tidak ada data waktu persiapan yang valid untuk dianalisis.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Waktu Persiapan Rata-rata", f"{df['Prep Time (Seconds)'].mean():.1f} detik")
    col2.metric("Waktu Persiapan Tercepat", f"{df['Prep Time (Seconds)'].min():.1f} detik")
    col3.metric("Waktu Persiapan Terlama", f"{df['Prep Time (Seconds)'].max():.1f} detik")

    df['Hour'] = pd.to_datetime(df['Order Time'], format='%H:%M:%S', errors='coerce').dt.hour
    avg_prep_by_hour = df.groupby('Hour')['Prep Time (Seconds)'].mean().reset_index()

    fig = px.bar(avg_prep_by_hour, x='Hour', y='Prep Time (Seconds)', title="Rata-rata Waktu Persiapan per Jam",
                 labels={'Hour': 'Jam dalam Sehari', 'Prep Time (Seconds)': 'Rata-rata Waktu Persiapan (Detik)'})
    st.plotly_chart(fig, use_container_width=True)
    st.info("Perhatikan jam-jam di mana waktu persiapan melonjak. Ini mungkin menandakan dapur sedang di bawah tekanan dan butuh perhatian lebih.")

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

    logo_path = "logo.png"
    if os.path.exists(logo_path): st.sidebar.image(logo_path, width=150)
    else: st.sidebar.warning("Logo 'logo.png' tidak ditemukan.")
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{name}**")
    st.sidebar.title("📤 Unggah Data")
    
    uploaded_sales_file = st.sidebar.file_uploader("1. Unggah File Penjualan Detail", type=["xlsx", "xls", "csv"], key="sales_uploader")
    
    if uploaded_sales_file is None:
        st.info("👋 Selamat datang! Silakan unggah file data penjualan detail Anda untuk memulai analisis.")
        st.session_state.data_processed = False 
        st.stop()

    df_raw = load_raw_data(uploaded_sales_file)
    
    st.sidebar.title("🔗 Pemetaan Kolom")
    
    # --- MODIFIKASI: Pemetaan kolom wajib dan opsional ---
    REQUIRED_COLS_MAP = {
        'Sales Date': 'Kolom Tanggal Transaksi', 'Branch': 'Kolom Nama Cabang', 'Bill Number': 'Kolom Nomor Struk/Bill',
        'Nett Sales': 'Kolom Penjualan Bersih (Nett)', 'Menu': 'Kolom Nama Item/Menu', 'Qty': 'Kolom Kuantitas'
    }
    OPTIONAL_COLS_MAP = {
        'Visit Purpose': 'Saluran Penjualan (Online/Offline)', 'Payment Method': 'Metode Pembayaran',
        'Sales Date In': 'Waktu Pesanan Masuk', 'Sales Date Out': 'Waktu Pesanan Selesai', 'Order Time': 'Jam Pesanan'
    }
    
    user_mapping = {}
    
    def find_best_match(col_list, keywords):
        for col in col_list:
            for keyword in keywords:
                if keyword in col.lower().replace("_", "").replace(" ", ""): return col
        return None

    # Pemetaan kolom Wajib
    with st.sidebar.expander("Atur Pemetaan Kolom Wajib", expanded=not st.session_state.data_processed):
        all_cols = [""] + df_raw.columns.tolist()
        for internal_name, description in REQUIRED_COLS_MAP.items():
            # ... (logika tebak otomatis sama seperti sebelumnya) ...
            best_guess = find_best_match(all_cols, [internal_name.lower().replace("_", "").replace(" ","")])
            user_selection = st.selectbox(f"**{description}**:", options=all_cols, index=(all_cols.index(best_guess) if best_guess else 0), key=f"map_req_{internal_name}")
            if user_selection: user_mapping[internal_name] = user_selection
    
    # Pemetaan kolom Opsional
    with st.sidebar.expander("Atur Pemetaan Kolom Opsional (untuk Analisis Lanjutan)", expanded=False):
        for internal_name, description in OPTIONAL_COLS_MAP.items():
            best_guess = find_best_match(all_cols, [internal_name.lower().replace("_", "").replace(" ","")])
            user_selection = st.selectbox(f"**{description}**:", options=all_cols, index=(all_cols.index(best_guess) if best_guess else 0), key=f"map_opt_{internal_name}")
            if user_selection: user_mapping[internal_name] = user_selection

    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        # ... (Logika pemrosesan sama seperti sebelumnya) ...
        st.session_state.df_processed = df # Simpan dataframe yang sudah diproses
        st.session_state.data_processed = True
        st.rerun()

    # ==============================================================================
    # BAGIAN UTAMA DASBOR (Tampil setelah data diproses)
    # ==============================================================================
    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        st.sidebar.title("⚙️ Filter Global")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date = df_processed['Sales Date'].min()
        max_date = df_processed['Sales Date'].max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered = df_processed[(df_processed['Branch'] == selected_branch) & (df_processed['Sales Date'] >= start_date) & (df_processed['Sales Date'] <= end_date)]
        
        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
            st.stop()
            
        st.title(f"Dashboard Holistik: {selected_branch}")
        st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

        # --- PEMBUATAN TAB UTAMA ---
        trend_tab, ops_tab = st.tabs(["📈 **Dashboard Tren Performa**", "🚀 **Dashboard Analisis Operasional**"])

        with trend_tab:
            st.header("Analisis Tren Performa Jangka Panjang")
            # --- Di sini kita letakkan semua kode untuk dasbor tren yang sudah ada ---
            # (Agregasi bulanan, KPI, dan visualisasi tren)
            monthly_df = df_filtered.copy()
            # ... (tempel kode agregasi bulanan, KPI, dan `create_trend_chart` di sini) ...
            st.info("Dasbor tren performa akan ditampilkan di sini.")


        with ops_tab:
            st.header("Wawasan Operasional dan Taktis")
            
            # Panggil fungsi-fungsi analisis operasional baru
            create_channel_analysis(df_filtered.copy())
            st.markdown("---")
            create_menu_engineering_chart(df_filtered.copy())
            st.markdown("---")
            create_operational_efficiency_analysis(df_filtered.copy())