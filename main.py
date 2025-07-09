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
    page_title="Dashboard Performa Dinamis", 
    page_icon="📊",
    layout="wide"
)

# MODIFIKASI: Fungsi ini sekarang hanya memuat data, tidak lagi me-rename atau memvalidasi nama kolom.
@st.cache_data
def load_raw_data(file):
    """
    Memuat data mentah dari file yang diunggah tanpa modifikasi.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
        return df.copy() # Return a copy to prevent mutation issues with cache
    except Exception as e:
        raise ValueError(f"Gagal memuat file: {e}")

# Fungsi ini tetap sama untuk file metrik tambahan
@st.cache_data
def load_additional_data(file):
    # (Fungsi ini tidak diubah, namun prinsip pemetaan bisa diterapkan juga di sini jika diperlukan)
    try:
        df_add = pd.read_excel(file)
        # Validasi dasar
        if 'Date' not in df_add.columns or 'Branch' not in df_add.columns:
            st.warning("Untuk pemetaan otomatis, file metrik tambahan disarankan memiliki kolom 'Date' dan 'Branch'.")
            # Jika tidak ada, pengguna harus memetakannya secara manual (fitur lanjutan)
        
        # Proses seperti biasa untuk saat ini
        df_add.rename(columns={'Date': 'Metrics Date', 'Branch': 'Metrics Branch'}, inplace=True)
        df_add['Metrics Date'] = pd.to_datetime(df_add['Metrics Date']).dt.date
        df_add['Metrics Branch'] = df_add['Metrics Branch'].fillna('Tidak Diketahui')
        metric_cols = [col for col in df_add.columns if col not in ['Metrics Date', 'Metrics Branch']]
        if not metric_cols:
            raise ValueError("Tidak ada kolom metrik yang terdeteksi di file tambahan.")
        for col in metric_cols:
            df_add[col] = pd.to_numeric(df_add[col], errors='coerce')
        df_add.fillna(0, inplace=True)
        return df_add, metric_cols
    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memproses file metrik tambahan: {e}")

def analyze_trend_v2(data_series, time_series):
    # Fungsi ini tidak berubah
    if len(data_series.dropna()) < 3: return "Data tidak cukup untuk analisis tren.", None, None
    if len(set(data_series.dropna())) <= 1: return "Data konstan, tidak ada tren.", None, None
    data_series_interpolated = data_series.interpolate(method='linear', limit_direction='both')
    x = np.arange(len(data_series_interpolated))
    y = data_series_interpolated.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trendline = slope * x + intercept
    overall_trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan" if p_value < 0.05 else "cenderung **stabil/fluktuatif**"
    return f"Secara keseluruhan, data {overall_trend}.", trendline, p_value

def display_analysis_with_details(title, analysis_text, p_value):
    # Fungsi ini tidak berubah
    st.info(f"💡 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Lihat penjelasan p-value"):
            st.markdown(f"**Nilai p-value** tren ini adalah **`{p_value:.4f}`**. Angka ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05: st.success("✔️ Karena kemungkinan kebetulan rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else: st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================
# (Tidak ada perubahan di bagian ini)
config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False: st.error("Username atau password salah.")
elif auth_status is None: st.warning("Silakan masukkan username dan password.")
elif auth_status:
    # ==============================================================================
    # APLIKASI UTAMA (SETELAH LOGIN BERHASIL)
    # ==============================================================================
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False

    logo_path = "logo.png"
    if os.path.exists(logo_path): st.sidebar.image(logo_path, width=150)
    else: st.sidebar.warning("Logo 'logo.png' tidak ditemukan.")
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{name}**")
    st.sidebar.title("📤 Unggah Data")
    
    uploaded_sales_file = st.sidebar.file_uploader("1. Unggah File Penjualan", type=["xlsx", "xls", "csv"])
    
    if uploaded_sales_file is None:
        st.info("👋 Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
        st.stop()

    df_raw = load_raw_data(uploaded_sales_file)
    
    # --- BARU: Bagian Pemetaan Kolom Interaktif ---
    st.sidebar.title("🔗 Pemetaan Kolom")
    
    # Definisikan kolom yang dibutuhkan dan deskripsinya
    REQUIRED_COLS_MAP = {
        'Sales Date': 'Kolom Tanggal Transaksi',
        'Branch': 'Kolom Nama Cabang/Outlet',
        'Bill Number': 'Kolom Nomor Struk/Bill (Unik)',
        'Nett Sales': 'Kolom Penjualan Bersih (Nett)',
        'Menu': 'Kolom Nama Item/Menu',
        'Qty': 'Kolom Kuantitas/Jumlah Item'
    }

    # Buat dictionary untuk menampung pilihan pengguna
    user_mapping = {}
    
    # Fungsi untuk menebak kolom terbaik
    def find_best_match(col_list, keywords):
        for col in col_list:
            for keyword in keywords:
                if keyword in col.lower().replace("_", "").replace(" ", ""):
                    return col
        return None

    # Tampilkan selectbox untuk setiap kolom wajib
    with st.sidebar.expander("Atur Pemetaan Kolom Penjualan", expanded= not st.session_state.data_processed):
        all_cols = [""] + df_raw.columns.tolist()
        for internal_name, description in REQUIRED_COLS_MAP.items():
            # Coba tebak kolom yang paling mungkin
            keywords = [internal_name.lower().replace("_", "").replace(" ","")]
            if internal_name == 'Sales Date': keywords.extend(['tanggal', 'date', 'waktu'])
            if internal_name == 'Branch': keywords.extend(['cabang', 'outlet'])
            if internal_name == 'Bill Number': keywords.extend(['bill', 'struk', 'invoice', 'nomor'])
            if internal_name == 'Nett Sales': keywords.extend(['nett', 'bersih'])
            
            best_guess = find_best_match(all_cols, keywords)
            default_index = all_cols.index(best_guess) if best_guess else 0
            
            user_selection = st.selectbox(f"**{description}**:", options=all_cols, index=default_index)
            if user_selection:
                user_mapping[internal_name] = user_selection
    
    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        # Validasi: Pastikan semua kolom wajib telah dipetakan
        if not all(user_mapping.values()):
            st.error("❌ Harap petakan semua kolom yang wajib diisi.")
            st.stop()
        
        # Buat DataFrame baru dengan nama kolom yang sudah standar
        try:
            # Buat mapping dari nama pilihan -> nama internal
            rename_dict = {v: k for k, v in user_mapping.items()}
            df = df_raw.rename(columns=rename_dict)
            
            # Pastikan semua kolom ada setelah rename
            for col in REQUIRED_COLS_MAP.keys():
                if col not in df.columns:
                    raise ValueError(f"Kolom internal '{col}' tidak terbentuk. Periksa kembali pemetaan Anda.")

            # Proses data setelah di-rename
            df['Sales Date'] = pd.to_datetime(df['Sales Date']).dt.date
            df['Branch'] = df['Branch'].fillna('Tidak Diketahui')
            numeric_cols = ['Qty', 'Nett Sales'] # Hanya proses yang wajib
            for col in numeric_cols:
                 if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            df.fillna(0, inplace=True)

            st.session_state.df_processed = df # Simpan dataframe yang sudah diproses
            st.session_state.data_processed = True
            st.rerun() # Jalankan ulang skrip untuk menampilkan dashboard

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
            st.stop()

    # --- Tampilan Dasbor Utama (hanya tampil setelah data diproses) ---
    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        # --- UI Filter & Opsi Lanjutan ---
        st.sidebar.title("⚙️ Filter & Opsi")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date = df_processed['Sales Date'].min()
        max_date = df_processed['Sales Date'].max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        # (Logika file metrik tambahan bisa ditaruh di sini jika diperlukan)
        # ...
        
        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered = df_processed[(df_processed['Branch'] == selected_branch) & (df_processed['Sales Date'] >= start_date) & (df_processed['Sales Date'] <= end_date)]
        
        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
            st.stop()
            
        # --- Mulai Tampilkan Konten Dashboard ---
        st.title(f"📊 Dashboard Performa: {selected_branch}")
        st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
        st.markdown("---")

        # Proses Agregasi dan Visualisasi seperti kode sebelumnya
        monthly_df = df_filtered.copy()
        monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
        monthly_agg = monthly_df.groupby('Bulan').agg(
            TotalMonthlySales=('Nett Sales', 'sum'),
            TotalTransactions=('Bill Number', 'nunique')
        ).reset_index()
        monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

        if not monthly_agg.empty:
             monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

        # --- Tampilan KPI Dinamis ---
        if not monthly_agg.empty:
            # Sesuaikan jumlah KPI dengan metrik tambahan yang dipilih
            uploaded_metrics_file = st.session_state.get('uploaded_metrics_file', None)
            selected_metrics = st.session_state.get('selected_metrics', [])

            num_kpi_cols = 3 + len(selected_metrics)
            kpi_cols = st.columns(num_kpi_cols)

            last_month = monthly_agg.iloc[-1]
            prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None

            # Fungsi helper untuk menampilkan metrik KPI
            def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
                # Pastikan nilai tidak NaN sebelum pemrosesan
                if pd.isna(current_val):
                    col.metric(title, "N/A")
                    return

                delta = (current_val - prev_val) / prev_val if prev_val and prev_val > 0 and pd.notna(prev_val) else 0
                val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.2f}".rstrip('0').rstrip('.')
                
                # Hanya tampilkan delta jika ada data bulan sebelumnya
                delta_display = f"{delta:.1%}" if prev_val and pd.notna(prev_val) else None
                help_display = help_text if prev_val and pd.notna(prev_val) else None
                col.metric(title, val_format, delta_display, help=help_display)
            
            help_str = f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""

            # Tampilkan KPI Wajib (Penjualan, Transaksi, AOV)
            display_kpi(kpi_cols[0], "💰 Penjualan", last_month.get('TotalMonthlySales'), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
            display_kpi(kpi_cols[1], "🛒 Transaksi", last_month.get('TotalTransactions'), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
            display_kpi(kpi_cols[2], "💳 AOV", last_month.get('AOV'), prev_month.get('AOV') if prev_month is not None else None, help_str, True)
            
            # Tampilkan KPI Dinamis untuk metrik tambahan
            for i, metric in enumerate(selected_metrics):
                if metric in last_month:
                    display_kpi(kpi_cols[3+i], f"⭐ {metric}", last_month.get(metric), prev_month.get(metric) if prev_month is not None else None, help_str, False)
        
        st.markdown("---")

        # --- Tampilan Visualisasi Dinamis ---
        with st.expander("📈 Lihat Menu Terlaris", expanded=False):
            top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
            top_menus.index = top_menus.index + 1
            st.dataframe(top_menus, use_container_width=True)
        
        tab_titles = ["Penjualan", "Transaksi", "AOV"] + selected_metrics
        tabs = st.tabs([f"**{title}**" for title in tab_titles])

        def create_trend_chart(tab, data, y_col, y_label, color):
            with tab:
                st.subheader(f"Analisis Tren Bulanan: {y_label}")
                fig = px.line(data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
                fig.update_traces(line_color=color)
                
                analysis, trendline, p_val = analyze_trend_v2(data[y_col], data['Bulan'].dt.strftime('%b %Y'))
                if trendline is not None:
                    fig.add_scatter(x=data['Bulan'], y=trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
                
                st.plotly_chart(fig, use_container_width=True)
                display_analysis_with_details(f"Analisis Tren {y_label}", analysis, p_val)

        if not monthly_agg.empty:
            create_trend_chart(tabs[0], monthly_agg, 'TotalMonthlySales', 'Total Penjualan (Rp)', 'royalblue')
            create_trend_chart(tabs[1], monthly_agg, 'TotalTransactions', 'Jumlah Transaksi', 'orange')
            create_trend_chart(tabs[2], monthly_agg, 'AOV', 'Average Order Value (Rp)', 'green')

            # Grafik dinamis untuk metrik tambahan
            for i, metric in enumerate(selected_metrics):
                if metric in monthly_agg.columns and monthly_agg[metric].notna().any():
                    color_palette = px.colors.qualitative.Vivid
                    color = color_palette[i % len(color_palette)]
                    create_trend_chart(tabs[3+i], monthly_agg, metric, metric, color)