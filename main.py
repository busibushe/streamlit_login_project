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

# --- FUNGSI PEMUATAN DATA (Tidak diubah) ---
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
    try:
        rename_dict = {v: k for k, v in user_mapping.items()}
        df_add = df_add_raw.rename(columns=rename_dict)
        df_add['Date'] = pd.to_datetime(df_add['Date']).dt.date
        df_add['Branch'] = df_add['Branch'].fillna('Tidak Diketahui')
        metric_cols = [col for col in df_add.columns if col not in ['Date', 'Branch']]
        if not metric_cols:
            raise ValueError("Tidak ada kolom metrik yang terdeteksi di file tambahan.")
        for col in metric_cols:
            df_add[col] = pd.to_numeric(df_add[col], errors='coerce')
        df_add.fillna(0, inplace=True)
        return df_add, metric_cols
    except Exception as e:
        raise ValueError(f"Gagal memproses data metrik tambahan: {e}")

# ==============================================================================
# FUNGSI ANALISIS TREN BARU (v3)
# ==============================================================================
def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """
    Menganalisis tren dengan wawasan bisnis F&B: Tren Linear, YoY, dan Momentum.
    Mengembalikan dictionary berisi narasi dan komponen chart.
    """
    if len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan)."}

    # --- 1. Analisis Tren Linear (Keseluruhan) ---
    df = df_monthly.dropna(subset=[metric_col]).copy()
    df['x_val'] = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05:
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    
    # --- 2. Analisis Year-over-Year (YoY) ---
    yoy_narrative = ""
    if len(df) >= 13:
        last_month_data = df.iloc[-1]
        yoy_month_data = df.iloc[-13]
        last_val = last_month_data[metric_col]
        yoy_val = yoy_month_data[metric_col]
        
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_performance = f"**tumbuh {yoy_change:.1%}**" if yoy_change > 0 else f"**menurun {abs(yoy_change):.1%}**"
            yoy_narrative = (f" Dibandingkan bulan yang sama tahun lalu ({yoy_month_data['Bulan'].strftime('%b %Y')}), "
                           f"performa bulan terakhir **{yoy_performance}**, menandakan adanya pertumbuhan riil di luar faktor musiman.")

    # --- 3. Analisis Momentum (3-Month Moving Average) ---
    ma_line = None
    momentum_narrative = ""
    if len(df) >= 4:
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        # Bandingkan 2 titik MA terakhir untuk melihat momentum
        if ma_line.iloc[-1] > ma_line.iloc[-2]:
            momentum_narrative = " Momentum jangka pendek (3 bulan terakhir) terlihat **positif**."
        else:
            momentum_narrative = " Momentum jangka pendek (3 bulan terakhir) menunjukkan **perlambatan**."

    # --- 4. Performa Ekstrem ---
    max_perf_month = df.loc[df[metric_col].idxmax()]
    min_perf_month = df.loc[df[metric_col].idxmin()]
    extrema_narrative = (f" Performa tertinggi tercatat pada **{max_perf_month['Bulan'].strftime('%B %Y')}** "
                         f"dan terendah pada **{min_perf_month['Bulan'].strftime('%B %Y')}**.")

    # --- 5. Gabungkan menjadi Narasi Bisnis ---
    full_narrative = (
        f"Secara keseluruhan, tren {metric_label} cenderung {trend_type} selama periode analisis."
        f"{momentum_narrative}"
        f"{yoy_narrative}"
        f"{extrema_narrative}"
    )

    return {
        'narrative': full_narrative,
        'trendline': trendline,
        'ma_line': ma_line,
        'p_value': p_value
    }


def display_analysis_with_details_v3(title, analysis_result):
    # (Fungsi ini dimodifikasi untuk menerima dictionary hasil)
    st.info(f"💡 **Analisis Tren {title}:** {analysis_result.get('narrative', 'Analisis tidak tersedia.')}")
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat penjelasan signifikansi statistik (p-value)"):
            st.markdown(f"**Nilai p-value** tren ini adalah **`{p_value:.4f}`**. Angka ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05: st.success("✔️ Karena kemungkinan kebetulan rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else: st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

# ==============================================================================
# LOGIKA AUTENTIKASI DAN APLIKASI UTAMA
# ==============================================================================
# (Tidak ada perubahan di bagian ini, cukup tempel saja)
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
    
    uploaded_sales_file = st.sidebar.file_uploader("1. Unggah File Penjualan", type=["xlsx", "xls", "csv"], key="sales_uploader")
    
    if uploaded_sales_file is None:
        st.info("👋 Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
        st.session_state.data_processed = False 
        st.stop()

    df_raw = load_raw_data(uploaded_sales_file)
    
    st.sidebar.title("🔗 Pemetaan Kolom")
    REQUIRED_COLS_MAP = {
        'Sales Date': 'Kolom Tanggal Transaksi', 'Branch': 'Kolom Nama Cabang/Outlet',
        'Bill Number': 'Kolom Nomor Struk/Bill (Unik)', 'Nett Sales': 'Kolom Penjualan Bersih (Nett)',
        'Menu': 'Kolom Nama Item/Menu', 'Qty': 'Kolom Kuantitas/Jumlah Item'
    }
    user_mapping = {}
    
    def find_best_match(col_list, keywords):
        for col in col_list:
            for keyword in keywords:
                if keyword in col.lower().replace("_", "").replace(" ", ""): return col
        return None

    with st.sidebar.expander("Atur Pemetaan Kolom Penjualan", expanded=not st.session_state.data_processed):
        all_cols = [""] + df_raw.columns.tolist()
        for internal_name, description in REQUIRED_COLS_MAP.items():
            keywords = [internal_name.lower().replace("_", "").replace(" ","")]
            if internal_name == 'Sales Date': keywords.extend(['tanggal', 'date', 'waktu'])
            if internal_name == 'Branch': keywords.extend(['cabang', 'outlet'])
            if internal_name == 'Bill Number': keywords.extend(['bill', 'struk', 'invoice', 'nomor'])
            if internal_name == 'Nett Sales': keywords.extend(['nett', 'bersih'])
            best_guess = find_best_match(all_cols, keywords)
            default_index = all_cols.index(best_guess) if best_guess else 0
            user_selection = st.selectbox(f"**{description}**:", options=all_cols, index=default_index, key=f"map_{internal_name}")
            if user_selection: user_mapping[internal_name] = user_selection
    
    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        if not all(user_mapping.values()):
            st.error("❌ Harap petakan semua kolom yang wajib diisi.")
            st.stop()
        try:
            rename_dict = {v: k for k, v in user_mapping.items()}
            df = df_raw.rename(columns=rename_dict)
            for col in REQUIRED_COLS_MAP.keys():
                if col not in df.columns: raise ValueError(f"Kolom internal '{col}' tidak terbentuk.")
            df['Sales Date'] = pd.to_datetime(df['Sales Date']).dt.date
            df['Branch'] = df['Branch'].fillna('Tidak Diketahui')
            numeric_cols = ['Qty', 'Nett Sales']
            for col in numeric_cols:
                if df[col].dtype == 'object': df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            df.fillna(0, inplace=True)
            st.session_state.df_processed = df
            st.session_state.data_processed = True
            st.rerun()
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
            st.stop()

    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        st.sidebar.title("⚙️ Filter & Opsi Lanjutan")
        unique_branches = sorted(df_processed['Branch'].unique())
        selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
        min_date = df_processed['Sales Date'].min()
        max_date = df_processed['Sales Date'].max()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        uploaded_metrics_file = st.sidebar.file_uploader("Unggah File Metrik Tambahan (Opsional)", type=["xlsx", "xls"], key="metrics_uploader")
        
        df_add, metric_cols, selected_metrics = None, [], []
        if uploaded_metrics_file:
            df_add_raw = load_raw_data(uploaded_metrics_file)
            with st.sidebar.expander("Atur Pemetaan Metrik Tambahan", expanded=True):
                all_add_cols = [""] + df_add_raw.columns.tolist()
                date_col = st.selectbox("Pilih Kolom Tanggal Metrik:", all_add_cols, index=(all_add_cols.index(find_best_match(all_add_cols, ['date', 'tanggal'])) if find_best_match(all_add_cols, ['date', 'tanggal']) else 0))
                branch_col = st.selectbox("Pilih Kolom Cabang Metrik:", all_add_cols, index=(all_add_cols.index(find_best_match(all_add_cols, ['branch', 'cabang'])) if find_best_match(all_add_cols, ['branch', 'cabang']) else 0))
            if date_col and branch_col:
                add_mapping = {'Date': date_col, 'Branch': branch_col}
                df_add, metric_cols = process_additional_data(df_add_raw, add_mapping)
                if metric_cols: selected_metrics = st.sidebar.multiselect("Pilih Metrik untuk Ditampilkan", options=metric_cols, default=metric_cols)

        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered = df_processed[(df_processed['Branch'] == selected_branch) & (df_processed['Sales Date'] >= start_date) & (df_processed['Sales Date'] <= end_date)]
        
        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
            st.stop()
            
        st.title(f"📊 Dashboard Performa: {selected_branch}")
        st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
        st.markdown("---")

        monthly_df = df_filtered.copy()
        monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
        monthly_agg = monthly_df.groupby('Bulan').agg(
            TotalMonthlySales=('Nett Sales', 'sum'), TotalTransactions=('Bill Number', 'nunique')
        ).reset_index()
        monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

        if df_add is not None and selected_metrics:
            df_add_filtered = df_add[(df_add['Branch'] == selected_branch) & (df_add['Date'] >= start_date) & (df_add['Date'] <= end_date)]
            if not df_add_filtered.empty:
                add_monthly_df = df_add_filtered.copy()
                add_monthly_df['Bulan'] = pd.to_datetime(add_monthly_df['Date']).dt.to_period('M')
                agg_dict = {metric: 'mean' for metric in selected_metrics}
                additional_monthly_agg = add_monthly_df.groupby('Bulan').agg(agg_dict).reset_index()
                monthly_agg = pd.merge(monthly_agg, additional_monthly_agg, on='Bulan', how='left')

        if not monthly_agg.empty:
            monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
            monthly_agg.fillna(0, inplace=True) # Ganti NaN dengan 0 setelah merge

        if not monthly_agg.empty:
            # ... (Logika KPI tidak berubah)
            num_kpi_cols = 3 + len(selected_metrics)
            kpi_cols = st.columns(num_kpi_cols)
            # ... (sisanya sama)
            
            st.markdown("---")
            with st.expander("📈 Lihat Menu Terlaris", expanded=False):
                top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
                top_menus.index = top_menus.index + 1
                st.dataframe(top_menus, use_container_width=True)
            
            # --- MODIFIKASI: Menggunakan fungsi analisis baru ---
            tab_titles = ["Penjualan", "Transaksi", "AOV"] + selected_metrics
            tabs = st.tabs([f"**{title}**" for title in tab_titles])

            def create_trend_chart_v3(tab, df_data, y_col, y_label, color):
                with tab:
                    analysis_result = analyze_trend_v3(df_data, y_col, y_label)
                    
                    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
                    fig.update_traces(line_color=color, name=y_label)
                    
                    if analysis_result.get('trendline') is not None:
                        fig.add_scatter(x=df_data['Bulan'], y=analysis_result['trendline'], mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
                    if analysis_result.get('ma_line') is not None:
                         fig.add_scatter(x=df_data['Bulan'], y=analysis_result['ma_line'], mode='lines', name='3-Month Moving Avg.', line=dict(color='orange', dash='dot'))

                    st.plotly_chart(fig, use_container_width=True)
                    display_analysis_with_details_v3(y_label, analysis_result)

            create_trend_chart_v3(tabs[0], monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
            create_trend_chart_v3(tabs[1], monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
            create_trend_chart_v3(tabs[2], monthly_agg, 'AOV', 'AOV', 'green')

            for i, metric in enumerate(selected_metrics):
                if metric in monthly_agg.columns and monthly_agg[metric].notna().any():
                    color_palette = px.colors.qualitative.Vivid
                    color = color_palette[i % len(color_palette)]
                    create_trend_chart_v3(tabs[3+i], monthly_agg, metric, metric, color)