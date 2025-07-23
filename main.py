# main.py

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dasbor Analisis Dinamis",
    page_icon="🚀",
    layout="wide"
)

# Inisialisasi session state
if 'mapping_confirmed' not in st.session_state:
    st.session_state.mapping_confirmed = False

# ==============================================================================
# FUNGSI ANALISIS DATA
# ==============================================================================
@st.cache_data
def load_file(uploaded_file):
    """Memuat file Feather yang diunggah."""
    if uploaded_file:
        try:
            return pd.read_feather(uploaded_file)
        except Exception as e:
            st.error(f"Gagal memuat {uploaded_file.name}: {e}")
    return None

def analyze_monthly_trends(df, mapping):
    """Mengagregasi data transaksi menjadi ringkasan bulanan."""
    df['Bulan'] = pd.to_datetime(df[mapping['sales_date']]).dt.to_period('M')
    monthly_agg = df.groupby('Bulan').agg(
        TotalMonthlySales=(mapping['nett_sales'], 'sum'),
        TotalTransactions=(mapping['bill_number'], 'nunique')
    ).reset_index()

    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(
            lambda r: r['TotalMonthlySales'] / r['TotalTransactions'] if r['TotalTransactions'] > 0 else 0,
            axis=1
        )
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
    return monthly_agg

def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """Menganalisis tren dengan regresi linear, YoY, dan momentum."""
    if df_monthly is None or len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': f"Data {metric_label} tidak cukup untuk analisis (min. 3 bulan)."}
    
    df = df_monthly.dropna(subset=[metric_col]).copy()
    if len(set(df[metric_col])) <= 1:
        return {'narrative': f"Data {metric_label} konstan, tidak ada tren."}

    df['x_val'] = np.arange(len(df))
    slope, intercept, _, p_value, _ = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05:
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"

    yoy_narrative = ""
    if len(df) >= 13:
        last_val, yoy_val = df.iloc[-1][metric_col], df.iloc[-13][metric_col]
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_perf = f"**tumbuh {yoy_change:.1%}**" if yoy_change > 0 else f"**menurun {abs(yoy_change):.1%}**"
            yoy_narrative = f" Dibandingkan tahun lalu, performa {yoy_perf}."

    ma_line, momentum_narrative = None, ""
    if len(df) >= 4:
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        if ma_line.iloc[-1] > ma_line.iloc[-2]:
            momentum_narrative = " Momentum jangka pendek (3 bulan) **positif**."
        elif ma_line.iloc[-1] < ma_line.iloc[-2]:
            momentum_narrative = " Momentum jangka pendek **melambat**."

    max_p = df.loc[df[metric_col].idxmax()]
    min_p = df.loc[df[metric_col].idxmin()]
    extrema = f" Puncak performa pada **{max_p['Bulan'].strftime('%B %Y')}** & terendah pada **{min_p['Bulan'].strftime('%B %Y')}**."
    
    full_narrative = f"Secara umum, tren {metric_label} {trend_type}.{momentum_narrative}{yoy_narrative}{extrema}"
    return {'narrative': full_narrative, 'trendline': trendline, 'ma_line': ma_line, 'p_value': p_value}

def calculate_branch_health(df_sales, df_complaints, mapping):
    """Menghitung rasio komplain dan waktu penyelesaian per cabang."""
    sales_agg = df_sales.groupby(mapping['branch']).agg(
        TotalSales=(mapping['nett_sales'], 'sum'),
        TotalTransactions=(mapping['bill_number'], 'nunique')
    ).reset_index()

    if df_complaints.empty or mapping['complaint_branch'] not in df_complaints.columns:
        df_health = sales_agg
        df_health['TotalComplaints'] = 0
        df_health['AvgResolutionTime'] = 0
    else:
        complaints_agg = df_complaints.groupby(mapping['complaint_branch']).agg(
            TotalComplaints=(mapping['complaint_branch'], 'count'),
            AvgResolutionTime=(mapping['resolution_time'], 'mean')
        ).reset_index()
        df_health = pd.merge(sales_agg, complaints_agg, left_on=mapping['branch'], right_on=mapping['complaint_branch'], how='left').fillna(0)

    df_health['ComplaintRatio'] = df_health.apply(lambda r: (r['TotalComplaints']/r['TotalTransactions'])*1000 if r['TotalTransactions']>0 else 0, axis=1)
    return df_health

# --- FUNGSI-FUNGSI UNTUK AGENT AI ---

def analyze_short_term_metric_status(df, date_col, metric_col):
    """Menganalisis tren 7 hari vs 7 hari sebelumnya."""
    if df is None or df.empty or metric_col not in df.columns or len(df.dropna(subset=[metric_col])) < 14:
        return "FLUKTUATIF"
    df = df.sort_values(date_col).dropna(subset=[metric_col])
    last_7_avg = df.tail(7)[metric_col].mean()
    prev_7_avg = df.iloc[-14:-7][metric_col].mean()
    if prev_7_avg == 0:
        return "MENINGKAT TAJAM" if last_7_avg > 0 else "STABIL"
    change = (last_7_avg - prev_7_avg) / prev_7_avg
    if change > 0.20: return "MENINGKAT TAJAM"
    if change > 0.07: return "MENINGKAT"
    if change < -0.20: return "MENURUN TAJAM"
    if change < -0.07: return "MENURUN"
    return "STABIL"

def analyze_long_term_metric_status(df, date_col, metric_col, agg='sum'):
    """Menganalisis tren bulanan (min 4 bulan) dengan regresi linear."""
    if df is None or df.empty or metric_col not in df.columns: return "TIDAK CUKUP DATA"
    resampled = df.set_index(date_col).resample('M')
    if agg == 'sum': monthly = resampled[metric_col].sum().reset_index()
    elif agg == 'mean': monthly = resampled[metric_col].mean().reset_index()
    elif agg == 'nunique': monthly = resampled[metric_col].nunique().reset_index()
    else: monthly = resampled.size().reset_index(name=metric_col)
    if len(monthly) < 4: return "DATA < 4 BULAN"
    monthly['x'] = np.arange(len(monthly))
    slope, _, _, p_val, _ = stats.linregress(monthly['x'], monthly[metric_col].fillna(0))
    trend = "TREN STABIL"
    if p_val < 0.1:
        if slope > 0.05: trend = "TREN MENINGKAT"
        elif slope < -0.05: trend = "TREN MENURUN"
    momentum = ""
    if len(monthly) >= 6:
        last_3 = monthly[metric_col].tail(3).mean()
        prev_3 = monthly[metric_col].iloc[-6:-3].mean()
        if prev_3 > 0:
            mom_change = (last_3 - prev_3) / prev_3
            if mom_change > 0.1: momentum = " | MOMENTUM POSITIF"
            elif mom_change < -0.1: momentum = " | MOMENTUM NEGATIF"
    return f"{trend}{momentum}"

def get_operational_knowledge_base():
    """Aturan untuk masalah jangka pendek."""
    return [
        {"c": lambda s: s["sales"] in ["MENURUN", "MENURUN TAJAM"], "rc": "Penurunan traffic/operasional mendadak minggu ini."},
        {"c": lambda s: s["complaints"] in ["MENINGKAT", "MENINGKAT TAJAM"], "rc": "Insiden layanan/kualitas produk baru-baru ini."},
        {"c": lambda s: s["qa_qc"] == "RENDAH", "rc": "Kegagalan kepatuhan SOP pada audit terakhir."},
        {"c": lambda s: s["sales"] in ["MENINGKAT TAJAM"], "rc": "Program promosi jangka pendek berhasil."},
    ]

def get_strategic_knowledge_base():
    """Aturan untuk masalah jangka panjang."""
    return [
        {"c": lambda s: "TREN MENURUN" in s["sales"] and "TREN MENINGKAT" in s["complaints"], "rc": "Layanan buruk menggerus loyalitas pelanggan."},
        {"c": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"], "rc": "Operasional buruk berdampak negatif pada penjualan."},
        {"c": lambda s: "TREN MENURUN" in s["transactions"], "rc": "Penurunan jumlah pengunjung (traffic)."},
        {"c": lambda s: "TREN MENINGKAT" in s["aov"], "rc": "Strategi harga/promo berhasil."},
        {"c": lambda s: "MOMENTUM NEGATIF" in s["sales"], "rc": "Waspada! Performa melambat dalam 3 bulan terakhir."},
    ]

def run_agents(df_sales, df_complaints, df_qa_qc, mapping):
    """Menjalankan kedua agent (operasional dan strategis) sekaligus."""
    op_results, strat_results = [], []
    op_kb, strat_kb = get_operational_knowledge_base(), get_strategic_knowledge_base()

    for branch in sorted([str(b) for b in df_sales[mapping['branch']].unique() if pd.notna(b)]):
        s_br = df_sales[df_sales[mapping['branch']] == branch]
        c_br = df_complaints[df_complaints[mapping['complaint_branch']] == branch]
        q_br = df_qa_qc[df_qa_qc[mapping['qa_branch']] == branch]

        # Analisis Operasional
        d_sales = s_br.groupby(pd.Grouper(key=mapping['sales_date'], freq='D')).agg(DailySales=(mapping['nett_sales'], 'sum'), Transactions=(mapping['bill_number'], 'nunique')).reset_index()
        d_sales['AOV'] = (d_sales['DailySales'] / d_sales['Transactions']).fillna(0)
        d_complaints = c_br.groupby(pd.Grouper(key=mapping['complaint_date'], freq='D')).size().reset_index(name='Complaints')
        last_qa = "BAIK"
        if not q_br.empty:
            if q_br.sort_values(mapping['qa_date']).iloc[-1][mapping['qa_score']] < 75: last_qa = "RENDAH"
        
        op_status = {"sales": analyze_short_term_metric_status(d_sales, mapping['sales_date'], 'DailySales'),
                     "transactions": analyze_short_term_metric_status(d_sales, mapping['sales_date'], 'Transactions'),
                     "aov": analyze_short_term_metric_status(d_sales, mapping['sales_date'], 'AOV'),
                     "complaints": analyze_short_term_metric_status(d_complaints, mapping['complaint_date'], 'Complaints'),
                     "qa_qc": last_qa}
        op_causes = {rule["rc"] for rule in op_kb if rule["c"](op_status)}
        op_results.append({"Toko": branch, **op_status, "Analisis Operasional": ", ".join(op_causes) or "-"})

        # Analisis Strategis
        bill_agg = s_br.groupby([mapping['bill_number'], pd.Grouper(key=mapping['sales_date'], freq='M')])[mapping['nett_sales']].sum().reset_index()
        m_aov = bill_agg.groupby(mapping['sales_date'])[mapping['nett_sales']].mean().reset_index().rename(columns={mapping['nett_sales']: 'AOV'})
        avg_qa = "N/A"
        if not q_br.empty:
            score = q_br[mapping['qa_score']].mean()
            if score < 75: avg_qa = "RENDAH"
            elif score < 85: avg_qa = "SEDANG"
            else: avg_qa = "TINGGI"

        strat_status = {"sales": analyze_long_term_metric_status(s_br, mapping['sales_date'], mapping['nett_sales'], 'sum'),
                        "transactions": analyze_long_term_metric_status(s_br, mapping['sales_date'], mapping['bill_number'], 'nunique'),
                        "aov": analyze_long_term_metric_status(m_aov, mapping['sales_date'], 'AOV', 'mean'),
                        "complaints": analyze_long_term_metric_status(c_br, mapping['complaint_date'], mapping['complaint_branch'], 'count'),
                        "qa_qc": avg_qa}
        strat_causes = {rule["rc"] for rule in strat_kb if rule["c"](strat_status)}
        strat_results.append({"Toko": branch, **strat_status, "Analisis Strategis": ", ".join(strat_causes) or "-"})

    return pd.DataFrame(op_results), pd.DataFrame(strat_results)

# ==============================================================================
# FUNGSI KOMPONEN UI
# ==============================================================================

def display_column_mapper(df_sales, df_complaints, df_qa_qc):
    """Menampilkan UI untuk memetakan kolom di sidebar."""
    st.sidebar.title("🗺️ Pemetaan Kolom")
    st.sidebar.warning("Harap petakan kolom dari file Anda sebelum melanjutkan.")
    
    CONFIG = {
        'sales_date': {'label': 'Tanggal Transaksi', 'defaults': ['Sales Date', 'date', 'Tanggal']},
        'branch': {'label': 'Nama Cabang/Toko', 'defaults': ['Branch', 'Toko', 'Cabang']},
        'nett_sales': {'label': 'Penjualan (numerik)', 'defaults': ['Nett Sales', 'Sales', 'Revenue']},
        'bill_number': {'label': 'ID Transaksi/Bill', 'defaults': ['Bill Number', 'bill_id']},
        'complaint_date': {'label': 'Tanggal Komplain', 'defaults': ['Sales Date', 'date', 'Tanggal']},
        'complaint_branch': {'label': 'Cabang (Komplain)', 'defaults': ['Branch', 'Toko', 'Cabang']},
        'complaint_category': {'label': 'Kategori Kesalahan', 'defaults': ['kesalahan', 'category']},
        'complaint_priority': {'label': 'Prioritas/Golongan', 'defaults': ['golongan', 'priority']},
        'resolution_time': {'label': 'Waktu Penyelesaian (Jam)', 'defaults': ['Waktu Penyelesaian (Jam)']},
        'qa_date': {'label': 'Tanggal Audit QA/QC', 'defaults': ['Sales Date', 'date', 'Tanggal']},
        'qa_branch': {'label': 'Cabang (QA/QC)', 'defaults': ['Branch', 'Toko', 'Cabang']},
        'qa_score': {'label': 'Skor Kepatuhan (numerik)', 'defaults': ['Skor Kepatuhan', 'Score', 'Nilai']}
    }
    
    cols = {'sales': [""] + list(df_sales.columns), 
            'complaints': [""] + list(df_complaints.columns), 
            'qa_qc': [""] + list(df_qa_qc.columns)}

    def get_default_index(options, defaults):
        for val in defaults:
            if val in options: return options.index(val)
        return 0
    
    mapping = {}
    with st.sidebar.expander("1. Data Penjualan", True):
        for key in ['sales_date', 'branch', 'nett_sales', 'bill_number']:
            mapping[key] = st.selectbox(CONFIG[key]['label'], cols['sales'], index=get_default_index(cols['sales'], CONFIG[key]['defaults']))
    with st.sidebar.expander("2. Data Komplain", True):
        for key in ['complaint_date', 'complaint_branch', 'complaint_category', 'complaint_priority', 'resolution_time']:
            mapping[key] = st.selectbox(CONFIG[key]['label'], cols['complaints'], index=get_default_index(cols['complaints'], CONFIG[key]['defaults']))
    with st.sidebar.expander("3. Data QA/QC", True):
        for key in ['qa_date', 'qa_branch', 'qa_score']:
            mapping[key] = st.selectbox(CONFIG[key]['label'], cols['qa_qc'], index=get_default_index(cols['qa_qc'], CONFIG[key]['defaults']))
            
    if st.sidebar.button("✅ Terapkan Pemetaan", type="primary"):
        if "" in mapping.values():
            st.sidebar.error("Harap isi semua kolom pemetaan yang wajib.")
        else:
            st.session_state.column_mapping = mapping
            st.session_state.mapping_confirmed = True
            st.rerun()

def display_monthly_kpis(monthly_agg):
    """Menampilkan 3 KPI utama: Penjualan, Transaksi, dan AOV bulanan."""
    if len(monthly_agg) < 1: return
    kpi_cols = st.columns(3)
    last = monthly_agg.iloc[-1]
    prev = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None
    
    def display_kpi(col, title, current, prev_val, is_curr=True):
        delta = None
        if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
            delta = (current - prev_val) / prev_val
        val_fmt = f"Rp {current:,.0f}" if is_curr else f"{current:,.0f}"
        help_str = f"vs {prev['Bulan'].strftime('%b %Y')}" if prev is not None else None
        col.metric(title, val_fmt, f"{delta:.1%}" if delta else None, help=help_str)
    
    display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last.get('TotalMonthlySales', 0), prev.get('TotalMonthlySales') if prev is not None else None)
    display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last.get('TotalTransactions', 0), prev.get('TotalTransactions') if prev is not None else None, False)
    display_kpi(kpi_cols[2], "💳 AOV Bulanan", last.get('AOV', 0), prev.get('AOV') if prev is not None else None)

def display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    """Membuat grafik tren lengkap dengan garis tren, moving average, dan narasi analisis."""
    st.subheader(f"📈 Analisis Tren: {y_label}")
    analysis_result = analyze_trend_v3(df_data, y_col, y_label)
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)
    if analysis_result.get('trendline') is not None:
        fig.add_scatter(x=df_data['Bulan'], y=analysis_result['trendline'], mode='lines', name='Garis Tren', line=dict(color='firebrick', dash='dash'))
    if analysis_result.get('ma_line') is not None:
        fig.add_scatter(x=df_data['Bulan'], y=analysis_result['ma_line'], mode='lines', name='Momentum (3-Bulan)', line=dict(color='orange', dash='dot'))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 **Ringkasan Analisis:** {analysis_result.get('narrative', 'Analisis tidak tersedia.')}")
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat penjelasan signifikansi statistik (p-value)"):
            st.markdown(f"**Nilai p-value** tren ini **`{p_value:.4f}`**. Artinya, ada **`{p_value:.2%}`** kemungkinan pola ini terjadi karena kebetulan.")
            if p_value < 0.05: st.success("✔️ Tren ini **nyata secara statistik**.")
            else: st.warning("⚠️ Tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

def display_branch_health(df_health, mapping):
    """Menampilkan grafik kesehatan cabang."""
    st.subheader("Dashboard Kesehatan Cabang")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_health, x=mapping['branch'], y='ComplaintRatio', title="Rasio Komplain per 1000 Transaksi", color='ComplaintRatio', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_health, x=mapping['branch'], y='AvgResolutionTime', title="Rata-rata Waktu Penyelesaian (Jam)", color='AvgResolutionTime', color_continuous_scale='Oranges')
        st.plotly_chart(fig2, use_container_width=True)

def display_complaint_analysis(df_complaints, mapping):
    """Menampilkan analisis detail dari data komplain."""
    st.subheader("Analisis Detail Komplain")
    if df_complaints.empty:
        st.warning("Tidak ada data komplain untuk filter yang dipilih."); return
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df_complaints, names=mapping['complaint_category'], title="Proporsi Kategori Kesalahan", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_complaints[mapping['complaint_priority']].value_counts().reset_index(), x=mapping['complaint_priority'], y='count', title="Jumlah Komplain per Prioritas")
        st.plotly_chart(fig2, use_container_width=True)

def display_qa_qc_analysis(df_qa_qc, mapping):
    """Menampilkan analisis data audit QA/QC."""
    st.subheader("⭐ Dashboard Kepatuhan Standar (QA/QC)")
    if df_qa_qc.empty:
        st.warning("Tidak ada data QA/QC untuk filter yang dipilih."); return
    avg_score = df_qa_qc[mapping['qa_score']].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")
    fig = px.line(df_qa_qc.sort_values(mapping['qa_date']), x=mapping['qa_date'], y=mapping['qa_score'], color=mapping['qa_branch'], markers=True, title="Tren Skor Kepatuhan per Audit")
    st.plotly_chart(fig, use_container_width=True)

def display_agent_analysis(df_analysis, title, info_text):
    """Menampilkan hasil analisis dari AI Agent dalam bentuk tabel."""
    st.header(title)
    st.info(info_text)
    
    def style_status(val):
        color = "grey"
        if isinstance(val, str):
            if any(k in val for k in ["MENINGKAT", "TINGGI", "POSITIF", "BAIK"]): color = "#2ca02c"
            if any(k in val for k in ["MENURUN", "RENDAH", "NEGATIF"]): color = "#d62728"
        return f'color: {color}'
        
    df_display = df_analysis.set_index('Toko')
    last_col = df_display.columns[-1]
    styled = df_display.style.apply(lambda col: col.map(style_status), subset=pd.IndexSlice[:, df_display.columns != last_col])
    st.dataframe(styled, use_container_width=True)

# ==============================================================================
# APLIKASI UTAMA
# ==============================================================================

def main():
    """Fungsi utama yang menjalankan seluruh alur aplikasi."""
    
    # --- 1. UPLOAD FILE ---
    st.sidebar.title("📤 Unggah Data Master")
    sales_file = st.sidebar.file_uploader("1. Penjualan (.feather)", type="feather")
    complaint_file = st.sidebar.file_uploader("2. Komplain (.feather)", type="feather")
    qa_qc_file = st.sidebar.file_uploader("3. QA/QC (.feather)", type="feather")

    if not all([sales_file, complaint_file, qa_qc_file]):
        st.info("👋 Selamat datang! Silakan unggah ketiga file master untuk memulai analisis.")
        return

    df_sales, df_complaints, df_qa_qc = load_file(sales_file), load_file(complaint_file), load_file(qa_qc_file)
    if any(df is None for df in [df_sales, df_complaints, df_qa_qc]): return

    # --- 2. PEMETAAN KOLOM ---
    if not st.session_state.mapping_confirmed:
        display_column_mapper(df_sales, df_complaints, df_qa_qc)
        st.info("👈 Silakan konfigurasikan pemetaan kolom di sidebar, lalu klik 'Terapkan Pemetaan'.")
        return
    
    mapping = st.session_state.column_mapping
    if st.sidebar.button("Ubah Pemetaan Kolom"):
        st.session_state.mapping_confirmed = False
        st.rerun()

    # --- 3. FILTER GLOBAL ---
    st.sidebar.title("⚙️ Filter Global")
    ALL_BRANCHES = "Semua Cabang (Gabungan)"
    
    try:
        branches = [ALL_BRANCHES] + sorted(list(df_sales[mapping['branch']].unique()))
        min_date = pd.to_datetime(df_sales[mapping['sales_date']]).min().date()
        max_date = pd.to_datetime(df_sales[mapping['sales_date']]).max().date()
    except KeyError as e:
        st.error(f"Kolom yang dipetakan ({e}) tidak ditemukan. Harap periksa kembali pemetaan Anda.")
        return
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data. Error: {e}")
        return

    selected_branch = st.sidebar.selectbox("Pilih Cabang", branches)
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) != 2: return
    start_date, end_date = date_range

    # --- 4. PEMFILTERAN DATA ---
    dfs = {'sales': df_sales, 'complaints': df_complaints, 'qa_qc': df_qa_qc}
    filtered_dfs = {}
    for name, df in dfs.items():
        date_col = mapping.get(f'{name}_date', mapping['sales_date'])
        branch_col = mapping.get(f'{name}_branch', mapping['branch'])
        
        df[date_col] = pd.to_datetime(df[date_col])
        mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
        df_filtered = df[mask]
        
        if selected_branch != ALL_BRANCHES:
            df_filtered = df_filtered[df_filtered[branch_col] == selected_branch]
        filtered_dfs[name] = df_filtered

    if filtered_dfs['sales'].empty:
        st.warning("Tidak ada data penjualan untuk filter yang dipilih."); return
        
    # --- 5. TAMPILAN UTAMA ---
    st.title(f"📊 Dasbor Analisis Holistik: {selected_branch}")
    
    with st.spinner("Menganalisis data... 🤖"):
        monthly_agg = analyze_monthly_trends(filtered_dfs['sales'], mapping)
        branch_health = calculate_branch_health(filtered_dfs['sales'], filtered_dfs['complaints'], mapping)
        op_agent_res, strat_agent_res = run_agents(df_sales, df_complaints, df_qa_qc, mapping)

    penjualan_tab, kualitas_tab, agent_tab = st.tabs(["📈 **Penjualan**", "✅ **Kualitas & Kepatuhan**", "🤖 **AI Root Cause Agent**"])

    with penjualan_tab:
        st.header("Performa Penjualan Bulanan")
        display_monthly_kpis(monthly_agg)
        st.markdown("---")
        display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
        display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'darkorange')
        display_trend_chart_and_analysis(monthly_agg, 'AOV', 'Nilai Belanja Rata-rata (AOV)', 'green')

    with kualitas_tab:
        display_branch_health(branch_health, mapping)
        st.markdown("---")
        display_complaint_analysis(filtered_dfs['complaints'], mapping)
        st.markdown("---")
        display_qa_qc_analysis(filtered_dfs['qa_qc'], mapping)
        
    with agent_tab:
        st.header("Analisis Akar Masalah Otomatis oleh AI Agent")
        op_tab, strat_tab = st.tabs(["**⚡ Operasional (Mingguan)**", "**🎯 Strategis (Bulanan)**"])
        with op_tab:
            display_agent_analysis(op_agent_res, "Diagnosis Taktis Jangka Pendek", "Menganalisis performa 7 hari terakhir untuk deteksi cepat.")
        with strat_tab:
            display_agent_analysis(strat_agent_res, "Diagnosis Strategis Jangka Panjang", "Menganalisis tren bulanan untuk evaluasi fundamental.")

if __name__ == "__main__":
    main()