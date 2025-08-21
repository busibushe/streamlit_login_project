import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from supabase.client import Client, create_client # <-- Import baru

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dashboard F&B Holistik",
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# KONEKSI KE SUPABASE
# ==============================================================================
# Inisialisasi koneksi ke Supabase menggunakan st.secrets
@st.cache_resource
def init_supabase_connection():
    """Menginisialisasi koneksi ke Supabase."""
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase_connection()

# ==============================================================================
# FUNGSI PEMUATAN DATA DARI SUPABASE
# ==============================================================================
# @st.cache_data(ttl=600)
def load_data_from_supabase(table_name: str):
    """
    Memuat data dari tabel Supabase dan mengembalikannya sebagai DataFrame.
    """
    try:
        # ==========================================================
        # ▼▼▼ PERUBAHAN ADA DI BARIS DI BAWAH INI ▼▼▼
        # ==========================================================
        # Menambahkan .range() untuk mengambil lebih dari 1000 baris
        response = supabase.table(table_name).select("*").range(0, 2000000).execute() # Ambil hingga 2 juta baris
        # ==========================================================
        
        df = pd.DataFrame(response.data)
        
        # Hapus kode debug yang lama jika Anda mau
        st.write(f"DEBUG: Data mentah dari tabel '{table_name}':")
        st.write(f"- Jumlah baris: {df.shape[0]}")
        
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Sales Date'}, inplace=True)
        
        if 'Sales Date' in df.columns and not df.empty:
            df['Sales Date'] = pd.to_datetime(df['Sales Date'])
            min_date = df['Sales Date'].min()
            max_date = df['Sales Date'].max()
            st.write(f"- Rentang tanggal yang ditarik: {min_date.strftime('%Y-%m-%d')} hingga {max_date.strftime('%Y-%m-%d')}")

        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari tabel '{table_name}': {e}")
        return pd.DataFrame()
# ==============================================================================
# FUNGSI-FUNGSI ANALISIS & VISUALISASI (Tidak ada perubahan di sini)
# ==============================================================================

def analyze_monthly_trends(df_filtered):
    """Menganalisis tren bulanan, termasuk total penjualan, transaksi, AOV, dan rata-rata penjualan harian."""
    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = monthly_df['Sales Date'].dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(
        TotalMonthlySales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()

    if not monthly_agg.empty:
        # Hitung AOV (Average Order Value)
        monthly_agg['AOV'] = monthly_agg.apply(
            lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0,
            axis=1
        )
        
        # Hitung Rata-rata Penjualan Harian
        timestamp_bulan = monthly_agg['Bulan'].dt.to_timestamp()
        monthly_agg['RataRataPenjualanHarian'] = monthly_agg['TotalMonthlySales'] / timestamp_bulan.dt.days_in_month

        # Konversi kolom Bulan ke format timestamp untuk plotting
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()

    return monthly_agg

def display_monthly_kpis(monthly_agg):
    """Menampilkan metriks KPI utama: Penjualan, Rata-rata Harian, Transaksi, dan AOV."""
    if len(monthly_agg) < 1: return
    kpi_cols = st.columns(4) # Menggunakan 4 kolom untuk KPI
    last_month = monthly_agg.iloc[-1]
    prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None

    def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
        delta = None
        if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
            delta = (current_val - prev_val) / prev_val
        val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.0f}"
        col.metric(title, val_format, f"{delta:.1%}" if delta is not None else None, help=help_text if delta is not None else None)

    help_str = f"Dibandingkan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
    
    # 1. KPI Penjualan Bulanan (Total Sales)
    display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales', 0), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
    
    # 2. KPI Rata-rata Penjualan Harian (Daily Sales) - Dipindahkan ke urutan kedua
    display_kpi(kpi_cols[1], "📅 Rata-rata Penjualan Harian", last_month.get('RataRataPenjualanHarian', 0), prev_month.get('RataRataPenjualanHarian') if prev_month is not None else None, help_str, True)
    
    # 3. KPI Transaksi Bulanan
    display_kpi(kpi_cols[2], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions', 0), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
    
    # 4. KPI AOV Bulanan
    display_kpi(kpi_cols[3], "💳 AOV Bulanan", last_month.get('AOV', 0), prev_month.get('AOV') if prev_month is not None else None, help_str, True)
def display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)
    st.plotly_chart(fig, use_container_width=True)

def calculate_branch_health(df_sales, df_complaints):
    sales_agg = df_sales.groupby('Branch').agg(TotalSales=('Nett Sales', 'sum'), TotalTransactions=('Bill Number', 'nunique')).reset_index()
    if df_complaints.empty:
        complaints_agg = pd.DataFrame(columns=['Branch', 'TotalComplaints', 'AvgResolutionTime'])
    else:
        complaints_agg = df_complaints.groupby('Branch').agg(TotalComplaints=('Branch', 'count'), AvgResolutionTime=('Waktu Penyelesaian (Jam)', 'mean')).reset_index()
    df_health = pd.merge(sales_agg, complaints_agg, on='Branch', how='left').fillna(0)
    df_health['ComplaintRatio'] = df_health.apply(lambda row: (row['TotalComplaints'] / row['TotalTransactions']) * 1000 if row['TotalTransactions'] > 0 else 0, axis=1)
    return df_health

def display_branch_health(df_health):
    st.subheader("Dashboard Kesehatan Cabang")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_health, x='Branch', y='ComplaintRatio', title="Rasio Komplain per 1000 Transaksi", color='ComplaintRatio', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_health, x='Branch', y='AvgResolutionTime', title="Rata-rata Waktu Penyelesaian Komplain (Jam)", color='AvgResolutionTime', color_continuous_scale='Oranges')
        st.plotly_chart(fig2, use_container_width=True)

def display_complaint_analysis(df_complaints):
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

def display_qa_qc_analysis(df_qa_qc):
    st.subheader("⭐ Dashboard Kepatuhan Standar (QA/QC)")
    if df_qa_qc.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih.")
        return
    avg_score = df_qa_qc['Skor Kepatuhan'].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")
    st.markdown("##### Tren Skor Kepatuhan per Audit")
    fig1 = px.line(df_qa_qc.sort_values('Sales Date'), x='Sales Date', y='Skor Kepatuhan', color='Branch', markers=True, title="Tren Skor Kepatuhan per Audit")
    fig1.update_yaxes(range=[0, 105])
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("##### Perbandingan Rata-rata Skor per Cabang")
    branch_avg_score = df_qa_qc.groupby('Branch')['Skor Kepatuhan'].mean().reset_index().sort_values('Skor Kepatuhan', ascending=False)
    fig2 = px.bar(branch_avg_score, x='Branch', y='Skor Kepatuhan', title="Perbandingan Rata-rata Skor Kepatuhan antar Cabang", color='Skor Kepatuhan', color_continuous_scale='Greens')
    st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# FUNGSI-FUNGSI UNTUK AGENTIC AI ROOT CAUSE ANALYSIS (Tidak Berubah)
# ==============================================================================
# --- A. FUNGSI UNTUK AGENT OPERASIONAL (JANGKA PENDEK / MINGGUAN) ---
def analyze_short_term_metric_status(df, date_col, metric_col):
    if df is None or df.empty or metric_col not in df.columns or len(df.dropna(subset=[metric_col])) < 14:
        return "FLUKTUATIF"
    df = df.sort_values(date_col).dropna(subset=[metric_col])
    last_7_days_avg = df.tail(7)[metric_col].mean()
    previous_7_days_avg = df.iloc[-14:-7][metric_col].mean()
    if previous_7_days_avg == 0:
        return "MENINGKAT TAJAM" if last_7_days_avg > 0 else "STABIL"
    change_percent = (last_7_days_avg - previous_7_days_avg) / previous_7_days_avg
    if change_percent > 0.20: return "MENINGKAT TAJAM"
    if change_percent > 0.07: return "MENINGKAT"
    if change_percent < -0.20: return "MENURUN TAJAM"
    if change_percent < -0.07: return "MENURUN"
    return "STABIL"

def get_operational_knowledge_base():
    return [
        {"condition": lambda s: s["sales"] in ["MENURUN", "MENURUN TAJAM"], "root_cause": "Penurunan traffic atau masalah operasional mendadak minggu ini."},
        {"condition": lambda s: s["complaints"] in ["MENINGKAT", "MENINGKAT TAJAM"], "root_cause": "Terjadi insiden layanan atau masalah kualitas produk baru-baru ini."},
        {"condition": lambda s: s["qa_qc"] == "RENDAH", "root_cause": "Ditemukan kegagalan kepatuhan SOP signifikan pada audit terakhir."},
        {"condition": lambda s: s["aov"] in ["MENURUN", "MENURUN TAJAM"], "root_cause": "Promo mingguan kurang efektif atau staf gagal melakukan upselling."},
        {"condition": lambda s: s["sales"] in ["MENINGKAT TAJAM"] and s["aov"] in ["MENINGKAT TAJAM"], "root_cause": "Program promosi jangka pendek sangat berhasil."},
    ]

def run_operational_agent(df_sales, df_complaints, df_qa_qc):
    all_branches = sorted([str(b) for b in df_sales['Branch'].unique() if pd.notna(b)])
    knowledge_base = get_operational_knowledge_base()
    results = []
    for branch in all_branches:
        sales_br = df_sales[df_sales['Branch'] == branch]
        complaints_br = df_complaints[df_complaints['Branch'] == branch]
        qa_qc_br = df_qa_qc[df_qa_qc['Branch'] == branch]
        daily_sales = sales_br.groupby(pd.Grouper(key='Sales Date', freq='D')).agg(DailySales=('Nett Sales', 'sum'), Transactions=('Bill Number', 'nunique')).reset_index()
        daily_sales['AOV'] = (daily_sales['DailySales'] / daily_sales['Transactions']).fillna(0)
        daily_complaints = complaints_br.groupby(pd.Grouper(key='Sales Date', freq='D')).size().reset_index(name='Complaints')
        last_qa_qc_status = "BAIK"
        if not qa_qc_br.empty:
            last_audit_score = qa_qc_br.sort_values('Sales Date').iloc[-1]['Skor Kepatuhan']
            if last_audit_score < 75: last_qa_qc_status = "RENDAH"
        status = {
            "sales": analyze_short_term_metric_status(daily_sales, 'Sales Date', 'DailySales'),
            "transactions": analyze_short_term_metric_status(daily_sales, 'Sales Date', 'Transactions'),
            "aov": analyze_short_term_metric_status(daily_sales, 'Sales Date', 'AOV'),
            "complaints": analyze_short_term_metric_status(daily_complaints, 'Sales Date', 'Complaints'),
            "qa_qc": last_qa_qc_status
        }
        causes = {rule["root_cause"] for rule in knowledge_base if rule["condition"](status)}
        results.append({"Toko": branch, "Penjualan (7d)": status["sales"], "Transaksi (7d)": status["transactions"], "AOV (7d)": status["aov"], "Komplain (7d)": status["complaints"], "Audit Terakhir": status["qa_qc"], "Analisis Operasional": ", ".join(causes) if causes else "Performa operasional stabil."})
    return pd.DataFrame(results)

# --- B. FUNGSI UNTUK AGENT STRATEGIS (JANGKA PANJANG / 3-BULANAN) ---
def analyze_long_term_metric_status(df, date_col, metric_col, agg_method='sum'):
    if df is None or df.empty or metric_col not in df.columns: return "TIDAK CUKUP DATA"
    if agg_method == 'sum': df_resampled = df.set_index(date_col).resample('M')[metric_col].sum().reset_index()
    elif agg_method == 'mean': df_resampled = df.set_index(date_col).resample('M')[metric_col].mean().reset_index()
    elif agg_method == 'nunique': df_resampled = df.set_index(date_col).resample('M')[metric_col].nunique().reset_index()
    else: df_resampled = df.set_index(date_col).resample('M').size().reset_index(name=metric_col)
    monthly_df = df_resampled
    if len(monthly_df) < 4: return "DATA < 4 BULAN"
    monthly_df['x'] = np.arange(len(monthly_df))
    slope, _, _, p_value, _ = stats.linregress(monthly_df['x'], monthly_df[metric_col].fillna(0))
    trend_status = "TREN STABIL"
    if p_value < 0.1:
        if slope > 0.05: trend_status = "TREN MENINGKAT"
        elif slope < -0.05: trend_status = "TREN MENURUN"
    momentum_status = ""
    if len(monthly_df) >= 6:
        last_3_months_avg = monthly_df[metric_col].tail(3).mean()
        prev_3_months_avg = monthly_df[metric_col].iloc[-6:-3].mean()
        if prev_3_months_avg > 0:
            momentum_change = (last_3_months_avg - prev_3_months_avg) / prev_3_months_avg
            if momentum_change > 0.1: momentum_status = " | MOMENTUM POSITIF"
            elif momentum_change < -0.1: momentum_status = " | MOMENTUM NEGATIF"
    return f"{trend_status}{momentum_status}"

def get_strategic_knowledge_base():
    return [
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "TREN MENINGKAT" in s["complaints"], "root_cause": "Layanan buruk menggerus loyalitas pelanggan (Bad service causing churn)."},
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"], "root_cause": "Operasional buruk berdampak negatif pada penjualan (Poor operations hurting loyalty)."},
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"] and "MOMENTUM NEGATIF" in s["sales"], "root_cause": "Masalah fundamental pada operasional/rekrutmen (Poor purchasing/hiring loyalty)."},
        {"condition": lambda s: "TREN MENURUN" in s["transactions"] and "TREN MENURUN" in s["sales"], "root_cause": "Penurunan jumlah pengunjung (Reduced visitors/traffic) jadi pendorong utama penurunan penjualan."},
        {"condition": lambda s: "TREN MENINGKAT" in s["aov"] and "TREN MENURUN" not in s["sales"], "root_cause": "Strategi harga/promo berhasil meningkatkan nilai belanja (Good promo / Customers buy more)."},
        {"condition": lambda s: "TREN MENINGKAT" in s["sales"] and ("TINGGI" in s["qa_qc"] or "TREN MENURUN" in s["complaints"]), "root_cause": "Standar operasional yang baik mendorong pertumbuhan (Good operations standard / Good service)."},
        {"condition": lambda s: "MOMENTUM NEGATIF" in s["sales"] or "MOMENTUM NEGATIF" in s["transactions"], "root_cause": "Waspada! Performa melambat dalam 3 bulan terakhir, sinyal awal potensi penurunan tren."},
    ]

def run_strategic_agent(df_sales, df_complaints, df_qa_qc):
    all_branches = sorted([str(b) for b in df_sales['Branch'].unique() if pd.notna(b)])
    knowledge_base = get_strategic_knowledge_base()
    results = []
    for branch in all_branches:
        sales_br = df_sales[df_sales['Branch'] == branch]
        complaints_br = df_complaints[df_complaints['Branch'] == branch]
        qa_qc_br = df_qa_qc[df_qa_qc['Branch'] == branch]
        bill_agg = sales_br.groupby(['Bill Number', pd.Grouper(key='Sales Date', freq='M')])['Nett Sales'].sum().reset_index()
        monthly_aov_df = bill_agg.groupby('Sales Date')['Nett Sales'].mean().reset_index().rename(columns={'Nett Sales': 'AOV'})
        qa_qc_score = "TIDAK ADA DATA"
        if not qa_qc_br.empty:
            avg_score = qa_qc_br['Skor Kepatuhan'].mean()
            if avg_score < 75: qa_qc_score = "RENDAH"
            elif avg_score < 85: qa_qc_score = "SEDANG"
            else: qa_qc_score = "TINGGI"
        status = {
            "sales": analyze_long_term_metric_status(sales_br, 'Sales Date', 'Nett Sales', 'sum'),
            "transactions": analyze_long_term_metric_status(sales_br, 'Sales Date', 'Bill Number', 'nunique'),
            "aov": analyze_long_term_metric_status(monthly_aov_df, 'Sales Date', 'AOV', 'mean'),
            "complaints": analyze_long_term_metric_status(complaints_br, 'Sales Date', 'Branch', 'count'),
            "qa_qc": qa_qc_score
        }
        causes = {rule["root_cause"] for rule in knowledge_base if rule["condition"](status)}
        results.append({"Toko": branch, "Tren Penjualan": status["sales"], "Tren Transaksi": status["transactions"], "Tren AOV": status["aov"], "Skor QA/QC": status["qa_qc"], "Tren Komplain": status["complaints"], "Analisis & Kemungkinan Akar Masalah": ", ".join(causes) if causes else "Tidak ada pola strategis signifikan yang terdeteksi."})
    return pd.DataFrame(results)

# --- C. FUNGSI DISPLAY UNIVERSAL ---
def display_agent_analysis(df_analysis, title, info_text):
    st.header(title)
    st.info(info_text)
    def style_status(val):
        color = "grey"
        if isinstance(val, str):
            if any(keyword in val for keyword in ["MENINGKAT", "TINGGI", "POSITIF", "BAIK"]): color = "#2ca02c"
            if any(keyword in val for keyword in ["MENURUN", "RENDAH", "NEGATIF"]): color = "#d62728"
        return f'color: {color}'
    df_display = df_analysis.set_index('Toko')
    last_column_name = df_display.columns[-1]
    styled_df = df_display.style.apply(lambda col: col.map(style_status), subset=pd.IndexSlice[:, df_display.columns != last_column_name])
    st.dataframe(styled_df, use_container_width=True)

# ==============================================================================
# APLIKASI UTAMA STREAMLIT
# ==============================================================================
def main_app(user_name):
    """Fungsi utama yang menjalankan seluruh aplikasi dashboard."""
    if 'authenticator' in globals() and 'name' in st.session_state and st.session_state.name:
      authenticator.logout("Logout", "sidebar")
      st.sidebar.success(f"Login sebagai: **{user_name}**")

    st.title("📊 Dashboard Analisis Penjualan")
    st.info("Memuat data penjualan dari Supabase... Harap tunggu sebentar.")

    # =====================================================================
    # ▼▼▼ PERUBAHAN UTAMA ADA DI SINI ▼▼▼
    # =====================================================================
    # Muat data penjualan seperti biasa
    df_sales = load_data_from_supabase("sales_data") 
    
    # Buat DataFrame kosong untuk data yang belum ada
    # Pastikan ada kolom-kolom penting agar filter tidak error
    df_complaints = pd.DataFrame(columns=['Branch', 'Sales Date', 'kesalahan', 'golongan', 'Waktu Penyelesaian (Jam)'])
    df_qa_qc = pd.DataFrame(columns=['Branch', 'Sales Date', 'Skor Kepatuhan'])
    
    # Konversi tipe data tanggal agar konsisten
    df_complaints['Sales Date'] = pd.to_datetime(df_complaints['Sales Date'])
    df_qa_qc['Sales Date'] = pd.to_datetime(df_qa_qc['Sales Date'])
    # =====================================================================

    if df_sales.empty:
        st.error("Gagal memuat data penjualan dari Supabase. Aplikasi tidak dapat dilanjutkan.")
        st.stop()
        
    st.sidebar.title("⚙️ Filter Global")
    ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
    unique_branches = sorted([str(b) for b in df_sales['Branch'].unique() if pd.notna(b)])
    branch_options = [ALL_BRANCHES_OPTION] + unique_branches
    selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
    
    min_date, max_date = df_sales['Sales Date'].min().date(), df_sales['Sales Date'].max().date()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2: st.stop()
    start_date, end_date = date_range

    date_mask_sales = (df_sales['Sales Date'].dt.date >= start_date) & (df_sales['Sales Date'].dt.date <= end_date)
    df_sales_filtered = df_sales[date_mask_sales]
    
    # Filter df_complaints dan df_qa_qc (akan menghasilkan DataFrame kosong, tapi tidak error)
    df_complaints_filtered = pd.DataFrame()
    if not df_complaints.empty:
        date_mask_complaints = (df_complaints['Sales Date'].dt.date >= start_date) & (df_complaints['Sales Date'].dt.date <= end_date)
        df_complaints_filtered = df_complaints[date_mask_complaints]

    df_qa_qc_filtered = pd.DataFrame()
    if not df_qa_qc.empty:
        date_mask_qa_qc = (df_qa_qc['Sales Date'].dt.date >= start_date) & (df_qa_qc['Sales Date'].dt.date <= end_date)
        df_qa_qc_filtered = df_qa_qc[date_mask_qa_qc]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered['Branch'] == selected_branch]
        if not df_complaints_filtered.empty:
            df_complaints_filtered = df_complaints_filtered[df_complaints_filtered['Branch'] == selected_branch]
        if not df_qa_qc_filtered.empty:
            df_qa_qc_filtered = df_qa_qc_filtered[df_qa_qc_filtered['Branch'] == selected_branch]

    if df_sales_filtered.empty: st.warning("Tidak ada data penjualan untuk filter yang dipilih."); st.stop()

    st.header(f"Menampilkan Analisis untuk: {selected_branch}")
    st.markdown(f"Periode: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    # Sisa kode tetap sama...
    with st.spinner("Menganalisis data... 🤖"):
        monthly_agg = analyze_monthly_trends(df_sales_filtered)
        df_branch_health = calculate_branch_health(df_sales_filtered, df_complaints_filtered)
        
        if 'op_agent_results' not in st.session_state:
            st.session_state.op_agent_results = run_operational_agent(df_sales, df_complaints, df_qa_qc)
        if 'strat_agent_results' not in st.session_state:
            st.session_state.strat_agent_results = run_strategic_agent(df_sales, df_complaints, df_qa_qc)
    
    penjualan_tab, kualitas_tab, qa_qc_tab, agent_tab = st.tabs(["📈 Penjualan", "✅ Kualitas & Komplain", "⭐ Kepatuhan QA/QC", "🤖 AI Root Cause Agent"])
    
    with penjualan_tab:
        st.header("Analisis Tren Performa Penjualan")
        if monthly_agg is not None and not monthly_agg.empty:
            display_monthly_kpis(monthly_agg)
            st.markdown("---")
            display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Total Penjualan Bulanan', 'royalblue')
            display_trend_chart_and_analysis(monthly_agg, 'RataRataPenjualanHarian', 'Rata-rata Penjualan Harian (per Bulan)', 'purple')
            display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Total Transaksi Bulanan', 'orange')
            display_trend_chart_and_analysis(monthly_agg, 'AOV', 'Rata-rata Nilai Pesanan (AOV)', 'green')
        else:
            st.warning("Data bulanan tidak cukup untuk analisis tren.")
        
    with kualitas_tab:
        st.header("Analisis Kualitas Layanan dan Penanganan Komplain")
        display_branch_health(df_branch_health)
        st.markdown("---")
        display_complaint_analysis(df_complaints_filtered)
        
    with qa_qc_tab:
        st.header("Analisis Kepatuhan Standar Operasional")
        display_qa_qc_analysis(df_qa_qc_filtered)
        
    with agent_tab:
        st.header("Analisis Akar Masalah Otomatis oleh AI Agent")
        
        op_tab, strat_tab = st.tabs(["**⚡ Analisis Operasional (Mingguan)**", "**🎯 Analisis Strategis (3 Bulanan)**"])
        
        with op_tab:
            display_agent_analysis(
                st.session_state.op_agent_results,
                title="Diagnosis Taktis Jangka Pendek",
                info_text="Agent ini menganalisis performa 7 hari terakhir untuk mendeteksi masalah yang membutuhkan reaksi cepat (quick response)."
            )
            
        with strat_tab:
            display_agent_analysis(
                st.session_state.strat_agent_results,
                title="Diagnosis Strategis Jangka Panjang",
                info_text="Agent ini menganalisis tren bulanan (minimal 4 bulan) untuk mengidentifikasi pola fundamental dan mengevaluasi dampak strategi jangka panjang."
            )
# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================
try:
    if 'credentials' not in st.secrets or 'cookie' not in st.secrets:
        main_app("Developer")
    else:
        config = {'credentials': dict(st.secrets['credentials']),'cookie': dict(st.secrets['cookie'])}
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
        name, auth_status, username = authenticator.login("Login", "main")
        if auth_status is False: st.error("Username atau password salah.")
        elif auth_status is None: st.warning("Silakan masukkan username dan password.")
        elif auth_status: main_app(name)
except Exception:
    main_app("Developer")