import os
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

# --- FUNGSI-FUNGSI UTAMA ---

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

def reset_processing_state():
    """Callback untuk mereset state jika file diubah."""
    st.session_state.data_processed = False
    for key in list(st.session_state.keys()):
        if key.startswith("map_"):
            del st.session_state[key]

# (Anda bisa menempelkan fungsi-fungsi analisis lainnya di sini jika mau, atau biarkan seperti ini)
def analyze_trend_v3(df_monthly, metric_col, metric_label):
    if len(df_monthly.dropna(subset=[metric_col])) < 3: return {'narrative': "Data tidak cukup untuk analisis tren."}
    df = df_monthly.dropna(subset=[metric_col]).copy()
    if len(set(df[metric_col])) <= 1: return {'narrative': "Data konstan, tidak ada tren."}
    df['x_val'] = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** signifikan" if p_value < 0.05 else "stabil/fluktuatif"
    yoy_narrative = ""
    if len(df) >= 13:
        last_val, yoy_val = df.iloc[-1][metric_col], df.iloc[-13][metric_col]
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_narrative = f" Dibandingkan bulan yang sama tahun lalu, performa **{'tumbuh' if yoy_change > 0 else 'menurun'} {abs(yoy_change):.1%}**."
    ma_line, momentum_narrative = None, ""
    if len(df) >= 4:
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        momentum_narrative = " Momentum jangka pendek terlihat **positif**." if ma_line.iloc[-1] > ma_line.iloc[-2] else " Momentum jangka pendek menunjukkan **perlambatan**."
    max_perf_month, min_perf_month = df.loc[df[metric_col].idxmax()], df.loc[df[metric_col].idxmin()]
    extrema_narrative = f" Puncak performa terjadi pada **{max_perf_month['Bulan'].strftime('%B %Y')}** dan titik terendah pada **{min_perf_month['Bulan'].strftime('%B %Y')}**."
    return {'narrative': f"Tren {metric_label} cenderung {trend_type}.{momentum_narrative}{yoy_narrative}{extrema_narrative}", 'trendline': trendline, 'ma_line': ma_line, 'p_value': p_value}

def display_analysis_with_details_v3(title, analysis_result):
    st.info(f"💡 **Analisis Tren {title}:** {analysis_result.get('narrative', 'N/A')}")
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat detail signifikansi statistik"):
            st.markdown(f"**P-value:** `{p_value:.4f}`. Ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05: st.success("✔️ Tren ini **nyata secara statistik**.")
            else: st.warning("⚠️ Tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

def create_channel_analysis(df):
    st.subheader("📊 Analisis Saluran Penjualan (Channel)")
    if 'Visit Purpose' not in df.columns: return st.warning("Kolom 'Visit Purpose' tidak dipetakan.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Saluran", df['Visit Purpose'].nunique())
        channel_sales = df.groupby('Visit Purpose')['Nett Sales'].sum().sort_values(ascending=False)
        fig = px.pie(channel_sales, values='Nett Sales', names=channel_sales.index, title="Kontribusi Penjualan", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        agg_data = df.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'), TotalBills=('Bill Number', 'nunique'))
        agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
        aov_by_channel = agg_data['AOV'].sort_values(ascending=False)
        fig2 = px.bar(aov_by_channel, y=aov_by_channel.index, x=aov_by_channel.values, orientation='h', labels={'x': 'AOV (Rp)', 'y': 'Saluran'}, title="AOV per Saluran")
        st.plotly_chart(fig2, use_container_width=True)

def create_menu_engineering_chart(df):
    st.subheader("🔬 Analisis Performa Menu (Menu Engineering)")
    menu_perf = df.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
    if len(menu_perf) < 2: return st.warning("Data menu tidak cukup untuk analisis.")
    avg_qty, avg_sales = menu_perf['Qty'].mean(), menu_perf['NettSales'].mean()
    show_text = len(menu_perf) < 75
    text_arg = 'Menu' if show_text else None
    fig = px.scatter(menu_perf, x='Qty', y='NettSales', text=text_arg, title="Kuadran Performa Menu", labels={'Qty': 'Total Kuantitas', 'NettSales': 'Total Penjualan (Rp)'}, size='NettSales', color='NettSales', hover_name='Menu')
    fig.add_vline(x=avg_qty, line_dash="dash", line_color="gray")
    fig.add_hline(y=avg_sales, line_dash="dash", line_color="gray")
    if show_text: fig.update_traces(selector=dict(type='scatter'), textposition='top_center')
    else: st.warning("⚠️ Label nama menu disembunyikan karena jumlahnya terlalu banyak. Arahkan mouse ke titik untuk detail.")
    st.plotly_chart(fig, use_container_width=True)
    st.info("""**Cara Membaca Kuadran:**
- **Kanan Atas (STARS 🌟):** Juara Anda! Populer & profit. **Promosikan!**
- **Kanan Bawah (WORKHORSES 🐴):** Populer, profit kurang. **Naikkan harga / buat bundling.**
- **Kiri Atas (PUZZLES 🤔):** Sangat profit, jarang dipesan. **Latih staf / beri diskon coba.**
- **Kiri Bawah (DOGS 🐶):** Kurang populer & profit. **Pertimbangkan untuk hapus.**""")

def create_operational_efficiency_analysis(df):
    st.subheader("⏱️ Analisis Efisiensi Operasional")
    required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time']
    if not all(col in df.columns for col in required_cols): return st.warning("Kolom waktu (In, Out, Order Time) tidak dipetakan.")
    df_eff = df.copy()
    df_eff['Sales Date In'], df_eff['Sales Date Out'] = pd.to_datetime(df_eff['Sales Date In'], errors='coerce'), pd.to_datetime(df_eff['Sales Date Out'], errors='coerce')
    df_eff.dropna(subset=['Sales Date In', 'Sales Date Out'], inplace=True)
    df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
    df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
    if df_eff.empty: return st.warning("Tidak ada data waktu persiapan yang valid.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Waktu Persiapan Rata-rata", f"{df_eff['Prep Time (Seconds)'].mean():.1f} dtk")
    col2.metric("Waktu Persiapan Tercepat", f"{df_eff['Prep Time (Seconds)'].min():.1f} dtk")
    col3.metric("Waktu Persiapan Terlama", f"{df_eff['Prep Time (Seconds)'].max():.1f} dtk")
    df_eff['Hour'] = pd.to_datetime(df_eff['Order Time'], format='%H:%M:%S', errors='coerce').dt.hour
    avg_prep_by_hour = df_eff.groupby('Hour')['Prep Time (Seconds)'].mean().reset_index()
    fig = px.bar(avg_prep_by_hour, x='Hour', y='Prep Time (Seconds)', title="Rata-rata Waktu Persiapan per Jam", labels={'Hour': 'Jam', 'Prep Time (Seconds)': 'Rata-rata Waktu (Detik)'})
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# LOGIKA AUTENTIKASI DAN APLIKASI UTAMA
# ==============================================================================
config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False: st.error("Username atau password salah.")
elif auth_status is None: st.warning("Silakan masukkan username dan password.")
elif auth_status:
    if 'data_processed' not in st.session_state: st.session_state.data_processed = False

    if os.path.exists("logo.png"): st.sidebar.image("logo.png", width=150)
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{name}**")
    st.sidebar.title("📤 Unggah & Konfigurasi")

    uploaded_sales_file = st.sidebar.file_uploader(
        "1. Unggah File Penjualan Detail",
        type=["xlsx", "xls", "csv"],
        key="sales_uploader",
        on_change=reset_processing_state # INI ADALAH PERBAIKAN UTAMA
    )
    
    if uploaded_sales_file is None:
        st.info("👋 Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
        st.stop()

    df_raw = load_raw_data(uploaded_sales_file)
    
    st.sidebar.subheader("🔗 Pemetaan Kolom")
    REQUIRED_COLS_MAP = {'Sales Date': 'Tgl. Transaksi', 'Branch': 'Nama Cabang', 'Bill Number': 'No. Struk/Bill', 'Nett Sales': 'Penjualan Bersih', 'Menu': 'Nama Item/Menu', 'Qty': 'Kuantitas'}
    OPTIONAL_COLS_MAP = {'Visit Purpose': 'Saluran Penjualan', 'Payment Method': 'Metode Pembayaran', 'Sales Date In': 'Waktu Pesanan Masuk', 'Sales Date Out': 'Waktu Pesanan Selesai', 'Order Time': 'Jam Pesanan'}
    user_mapping = {}
    
    def find_best_match(col_list, keywords):
        for col in col_list:
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
        # Validasi 1: Pastikan semua kolom wajib telah dipetakan
        mapped_req_cols = [user_mapping.get(internal_name) for internal_name in REQUIRED_COLS_MAP.keys()]
        if not all(mapped_req_cols):
            st.error("❌ Harap petakan semua kolom WAJIB diisi sebelum memproses data.")
            st.stop()
        
        # Validasi 2: Pastikan pengguna tidak memilih kolom yang sama untuk peran yang berbeda
        chosen_cols = [col for col in user_mapping.values() if col]
        if len(chosen_cols) != len(set(chosen_cols)):
            st.error("❌ Terdeteksi satu kolom dipilih untuk beberapa peran berbeda. Harap periksa kembali pemetaan Anda.")
            st.stop()

        try:
            # --- MULAI: Logika pemrosesan yang hilang ---
            # Membuat DataFrame baru yang bersih, bukan me-rename.
            df = pd.DataFrame()
            for internal_name, source_col in user_mapping.items():
                if source_col: 
                    df[internal_name] = df_raw[source_col]

            # 1. Proses Kolom Teks (String)
            df['Menu'] = df['Menu'].fillna('N/A').astype(str)
            df['Branch'] = df['Branch'].fillna('Tidak Diketahui').astype(str)
            if 'Visit Purpose' in df.columns: df['Visit Purpose'] = df['Visit Purpose'].fillna('N/A').astype(str)
            if 'Payment Method' in df.columns: df['Payment Method'] = df['Payment Method'].fillna('N/A').astype(str)

            # 2. Proses Kolom Tanggal & Waktu
            df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce')
            if 'Sales Date In' in df.columns: df['Sales Date In'] = pd.to_datetime(df['Sales Date In'], errors='coerce')
            if 'Sales Date Out' in df.columns: df['Sales Date Out'] = pd.to_datetime(df['Sales Date Out'], errors='coerce')
            if 'Order Time' in df.columns:
                # Coba konversi ke time, jika gagal biarkan NaT (Not a Time)
                df['Order Time'] = pd.to_datetime(df['Order Time'], errors='coerce').dt.time
            
            # 3. Proses Kolom Numerik
            numeric_cols = ['Qty', 'Nett Sales'] 
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Simpan DataFrame yang sudah jadi dan set status menjadi 'telah diproses'
            st.session_state.df_processed = df 
            st.session_state.data_processed = True
            st.rerun() # Jalankan ulang skrip untuk menampilkan dasbor

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
            st.stop()

    # ==============================================================================
    # BAGIAN UTAMA DASBOR (Tampil setelah data diproses)
    # ==============================================================================
    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        st.sidebar.title("⚙️ Filter Global")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date = df_processed['Sales Date'].dt.date.min()
        max_date = df_processed['Sales Date'].dt.date.max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered = df_processed[
            (df_processed['Branch'] == selected_branch) & 
            (df_processed['Sales Date'].dt.date >= start_date) & 
            (df_processed['Sales Date'].dt.date <= end_date)
        ]
        
        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
            st.stop()
            
        st.title(f"Dashboard Holistik: {selected_branch}")
        st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

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
                help_str = f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""

                def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
                    if pd.isna(current_val):
                        col.metric(title, "N/A"); return
                    delta = (current_val - prev_val) / prev_val if prev_val and prev_val > 0 and pd.notna(prev_val) else 0
                    val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.2f}".rstrip('0').rstrip('.')
                    delta_display = f"{delta:.1%}" if prev_val and pd.notna(prev_val) else None
                    col.metric(title, val_format, delta_display, help=help_text if prev_val else None)

                display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales'), prev_month.get('TotalMonthlySales') if prev_month else None, help_str, True)
                display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions'), prev_month.get('TotalTransactions') if prev_month else None, help_str, False)
                display_kpi(kpi_cols[2], "💳 AOV Bulanan", last_month.get('AOV'), prev_month.get('AOV') if prev_month else None, help_str, True)
                
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