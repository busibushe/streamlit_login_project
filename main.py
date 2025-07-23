import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dasbor Analisis Dinamis",
    page_icon="🧩",
    layout="wide"
)

# Inisialisasi session state jika belum ada
if 'mapping_confirmed' not in st.session_state:
    st.session_state.mapping_confirmed = False

# ==============================================================================
# FUNGSI PEMUATAN & PEMETAAN DATA
# ==============================================================================
@st.cache_data
def load_feather_file(uploaded_file):
    """Memuat satu file Feather."""
    if uploaded_file is None: return None
    try:
        return pd.read_feather(uploaded_file)
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}")
        return None

def display_column_mapper(df_sales, df_complaints, df_qa_qc):
    """Menampilkan UI untuk memetakan kolom di sidebar."""
    st.sidebar.title("🗺️ Pemetaan Kolom")
    st.sidebar.warning("Harap petakan kolom dari file Anda sebelum melanjutkan.")

    sales_cols = [""] + list(df_sales.columns)
    complaint_cols = [""] + list(df_complaints.columns)
    qa_qc_cols = [""] + list(df_qa_qc.columns)

    def get_default_index(options, default_values):
        for val in default_values:
            if val in options:
                return options.index(val)
        return 0

    mapping = {}
    with st.sidebar.expander("1. Data Penjualan", expanded=True):
        mapping['sales_date'] = st.selectbox("Kolom Tanggal Transaksi", sales_cols, index=get_default_index(sales_cols, ['Sales Date', 'date', 'Date', 'Tanggal']))
        mapping['branch'] = st.selectbox("Kolom Nama Cabang/Toko", sales_cols, index=get_default_index(sales_cols, ['Branch', 'Toko', 'store', 'Cabang']))
        mapping['nett_sales'] = st.selectbox("Kolom Penjualan (numerik)", sales_cols, index=get_default_index(sales_cols, ['Nett Sales', 'Sales', 'Revenue', 'penjualan']))
        mapping['bill_number'] = st.selectbox("Kolom ID Transaksi/Bill", sales_cols, index=get_default_index(sales_cols, ['Bill Number', 'bill_id', 'No Bill']))
        mapping['menu'] = st.selectbox("Kolom Nama Menu/Produk", sales_cols, index=get_default_index(sales_cols, ['Menu', 'Produk', 'Item', 'product_name']))
        mapping['qty'] = st.selectbox("Kolom Kuantitas (numerik)", sales_cols, index=get_default_index(sales_cols, ['Qty', 'quantity', 'jumlah']))

    with st.sidebar.expander("2. Data Komplain", expanded=True):
        mapping['complaint_date'] = st.selectbox("Kolom Tanggal Komplain", complaint_cols, index=get_default_index(complaint_cols, ['Sales Date', 'date', 'Date', 'Tanggal']))
        mapping['complaint_branch'] = st.selectbox("Kolom Cabang (Komplain)", complaint_cols, index=get_default_index(complaint_cols, ['Branch', 'Toko', 'store', 'Cabang']))
        mapping['complaint_category'] = st.selectbox("Kolom Kategori Kesalahan", complaint_cols, index=get_default_index(complaint_cols, ['kesalahan', 'category', 'kategori']))
        mapping['complaint_priority'] = st.selectbox("Kolom Prioritas/Golongan", complaint_cols, index=get_default_index(complaint_cols, ['golongan', 'priority', 'prioritas']))
        mapping['resolution_time'] = st.selectbox("Kolom Waktu Penyelesaian (numerik)", complaint_cols, index=get_default_index(complaint_cols, ['Waktu Penyelesaian (Jam)', 'resolution_hours']))

    with st.sidebar.expander("3. Data QA/QC", expanded=True):
        mapping['qa_date'] = st.selectbox("Kolom Tanggal Audit QA/QC", qa_qc_cols, index=get_default_index(qa_qc_cols, ['Sales Date', 'date', 'Date', 'Tanggal']))
        mapping['qa_branch'] = st.selectbox("Kolom Cabang (QA/QC)", qa_qc_cols, index=get_default_index(qa_qc_cols, ['Branch', 'Toko', 'store', 'Cabang']))
        mapping['qa_score'] = st.selectbox("Kolom Skor Kepatuhan (numerik)", qa_qc_cols, index=get_default_index(qa_qc_cols, ['Skor Kepatuhan', 'Score', 'Nilai']))

    if st.sidebar.button("✅ Terapkan Pemetaan", type="primary"):
        # Validasi bahwa semua kolom penting telah dipilih
        required_cols = [v for k, v in mapping.items() if 'date' in k or 'branch' in k or 'sales' in k or 'bill' in k or 'score' in k]
        if "" in required_cols or None in required_cols:
            st.sidebar.error("Harap isi semua kolom pemetaan yang wajib.")
        else:
            st.session_state.column_mapping = mapping
            st.session_state.mapping_confirmed = True
            st.rerun()

# ==============================================================================
# FUNGSI-FUNGSI ANALISIS & VISUALISASI (TELAH DIMODIFIKASI)
# ==============================================================================
def analyze_monthly_trends(df, mapping):
    df['Bulan'] = df[mapping['sales_date']].dt.to_period('M')
    monthly_agg = df.groupby('Bulan').agg(
        TotalMonthlySales=(mapping['nett_sales'], 'sum'),
        TotalTransactions=(mapping['bill_number'], 'nunique')
    ).reset_index()

    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(lambda r: r['TotalMonthlySales'] / r['TotalTransactions'] if r['TotalTransactions'] > 0 else 0, axis=1)
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
    return monthly_agg

def display_monthly_kpis(monthly_agg):
    # Fungsi ini tidak perlu diubah karena menerima data yang sudah diagregasi
    if len(monthly_agg) < 1: return
    kpi_cols = st.columns(3)
    last_month = monthly_agg.iloc[-1]
    prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None
    def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
        delta = None
        if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
            delta = (current_val - prev_val) / prev_val
        val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.0f}"
        col.metric(title, val_format, f"{delta:.1%}" if delta is not None else None, help=help_text if delta is not None else None)
    help_str = f"Dibandingkan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
    display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales', 0), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
    display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions', 0), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
    display_kpi(kpi_cols[2], "💳 AOV Bulanan", last_month.get('AOV', 0), prev_month.get('AOV') if prev_month is not None else None, help_str, True)

def calculate_price_group_analysis(df, mapping):
    required_cols = [mapping['menu'], mapping['nett_sales'], mapping['qty']]
    if not all(col in df.columns for col in required_cols): return None
    menu_prices = df.groupby(mapping['menu']).agg(TotalSales=(mapping['nett_sales'], 'sum'), TotalQty=(mapping['qty'], 'sum')).reset_index()
    menu_prices = menu_prices[menu_prices['TotalQty'] > 0]
    menu_prices['AvgPrice'] = menu_prices['TotalSales'] / menu_prices['TotalQty']
    if len(menu_prices) < 4: return None
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    menu_prices['PriceGroupLabel'] = kmeans.fit_predict(menu_prices[['AvgPrice']])
    cluster_centers = menu_prices.groupby('PriceGroupLabel')['AvgPrice'].mean().sort_values().index
    label_mapping = {center: f"Kelompok {i+1}" for i, center in enumerate(cluster_centers)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroupLabel'].map(label_mapping)
    price_order = [" (Termurah)", " (Menengah)", " (Mahal)", " (Termahal)"]
    sorted_groups = menu_prices.groupby('PriceGroup')['AvgPrice'].mean().sort_values().index
    final_label_map = {group: group + price_order[i] for i, group in enumerate(sorted_groups)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroup'].map(final_label_map)
    df_with_groups = pd.merge(df, menu_prices[[mapping['menu'], 'PriceGroup']], on=mapping['menu'], how='left')
    group_performance = df_with_groups.groupby('PriceGroup').agg(TotalSales=(mapping['nett_sales'], 'sum'), TotalQty=(mapping['qty'], 'sum')).reset_index()
    group_performance['sort_order'] = group_performance['PriceGroup'].str.extract('(\\d+)').astype(int)
    return group_performance.sort_values('sort_order').drop(columns='sort_order')

def calculate_branch_health(df_sales, df_complaints, mapping):
    sales_agg = df_sales.groupby(mapping['branch']).agg(TotalSales=(mapping['nett_sales'], 'sum'), TotalTransactions=(mapping['bill_number'], 'nunique')).reset_index()
    if df_complaints.empty or not all(c in df_complaints.columns for c in [mapping['complaint_branch'], mapping['resolution_time']]):
        complaints_agg = pd.DataFrame(columns=[mapping['complaint_branch'], 'TotalComplaints', 'AvgResolutionTime'])
    else:
        complaints_agg = df_complaints.groupby(mapping['complaint_branch']).agg(TotalComplaints=(mapping['complaint_branch'], 'count'), AvgResolutionTime=(mapping['resolution_time'], 'mean')).reset_index()
    df_health = pd.merge(sales_agg, complaints_agg, left_on=mapping['branch'], right_on=mapping['complaint_branch'], how='left').fillna(0)
    df_health['ComplaintRatio'] = df_health.apply(lambda r: (r['TotalComplaints'] / r['TotalTransactions']) * 1000 if r['TotalTransactions'] > 0 else 0, axis=1)
    return df_health

def display_complaint_analysis(df_complaints, mapping):
    st.subheader("Analisis Detail Komplain")
    if df_complaints.empty or not all(c in df_complaints.columns for c in [mapping['complaint_category'], mapping['complaint_priority']]):
        st.warning("Data komplain atau kolom yang dibutuhkan tidak tersedia."); return
    col1, col2 = st.columns(2)
    with col1:
        kesalahan_agg = df_complaints[mapping['complaint_category']].value_counts().reset_index()
        fig1 = px.pie(kesalahan_agg, names=mapping['complaint_category'], values='count', title="Proporsi Kategori Kesalahan", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        golongan_agg = df_complaints[mapping['complaint_priority']].value_counts().reset_index()
        fig2 = px.bar(golongan_agg, x=mapping['complaint_priority'], y='count', title="Jumlah Komplain per Golongan Prioritas", color=mapping['complaint_priority'])
        st.plotly_chart(fig2, use_container_width=True)

def display_qa_qc_analysis(df_qa_qc, mapping):
    st.subheader("⭐ Dashboard Kepatuhan Standar (QA/QC)")
    if df_qa_qc.empty or not all(c in df_qa_qc.columns for c in [mapping['qa_score'], mapping['qa_date'], mapping['qa_branch']]):
        st.warning("Data QA/QC atau kolom yang dibutuhkan tidak tersedia."); return
    avg_score = df_qa_qc[mapping['qa_score']].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")
    st.markdown("##### Tren Skor Kepatuhan per Audit")
    fig1 = px.line(df_qa_qc.sort_values(mapping['qa_date']), x=mapping['qa_date'], y=mapping['qa_score'], color=mapping['qa_branch'], markers=True, title="Tren Skor Kepatuhan per Audit")
    st.plotly_chart(fig1, use_container_width=True)

def display_price_group_analysis(analysis_results):
    st.subheader("📊 Analisis Kelompok Harga")
    if analysis_results is None or analysis_results.empty:
        st.warning("Data tidak cukup untuk analisis kelompok harga.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=analysis_results['PriceGroup'], y=analysis_results['TotalSales'], name='Total Penjualan', marker_color='royalblue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=analysis_results['PriceGroup'], y=analysis_results['TotalQty'], name='Total Kuantitas', mode='lines+markers', line=dict(color='darkorange')), secondary_y=True)
    fig.update_layout(title_text="Kontribusi Penjualan vs. Kuantitas per Kelompok Harga", xaxis_title="Kelompok Harga", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="<b>Total Penjualan (Rp)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Kuantitas Terjual</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
def display_branch_health(df_health):
    st.subheader("Dashboard Kesehatan Cabang")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_health, x='Branch', y='ComplaintRatio', title="Rasio Komplain per 1000 Transaksi", color='ComplaintRatio', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_health, x='Branch', y='AvgResolutionTime', title="Rata-rata Waktu Penyelesaian Komplain (Jam)", color='AvgResolutionTime', color_continuous_scale='Oranges')
        st.plotly_chart(fig2, use_container_width=True)
def old_display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)
    st.plotly_chart(fig, use_container_width=True)
def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """Menganalisis tren dengan wawasan bisnis F&B: Tren Linear, YoY, dan Momentum."""
    # Pastikan data cukup untuk analisis
    if df_monthly is None or len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': f"Data {metric_label} tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan)."}
    
    df = df_monthly.dropna(subset=[metric_col]).copy()
    
    # Cek jika data konstan (tidak ada variasi)
    if len(set(df[metric_col])) <= 1:
        return {'narrative': f"Data {metric_label} konstan, tidak ada tren yang bisa dianalisis."}

    # Analisis Tren Linear
    df['x_val'] = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05: # Ambang batas signifikansi
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"

    # Analisis Year-on-Year (YoY)
    yoy_narrative = ""
    if len(df) >= 13: # Butuh setidaknya 13 bulan untuk membandingkan dengan tahun lalu
        last_val = df.iloc[-1][metric_col]
        yoy_val = df.iloc[-13][metric_col]
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_performance = f"**tumbuh {yoy_change:.1%}**" if yoy_change > 0 else f"**menurun {abs(yoy_change):.1%}**"
            yoy_narrative = f" Dibandingkan bulan yang sama tahun lalu, performa bulan terakhir {yoy_performance}."

    # Analisis Momentum (Moving Average)
    ma_line, momentum_narrative = None, ""
    if len(df) >= 4: # Butuh beberapa bulan untuk melihat momentum
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        if ma_line.iloc[-1] > ma_line.iloc[-2]:
            momentum_narrative = " Momentum jangka pendek (3 bulan terakhir) terlihat **positif**."
        elif ma_line.iloc[-1] < ma_line.iloc[-2]:
            momentum_narrative = " Momentum jangka pendek menunjukkan **perlambatan**."

    # Analisis Titik Ekstrem
    max_perf_month = df.loc[df[metric_col].idxmax()]
    min_perf_month = df.loc[df[metric_col].idxmin()]
    extrema_narrative = f" Performa tertinggi tercatat pada **{max_perf_month['Bulan'].strftime('%B %Y')}** dan terendah pada **{min_perf_month['Bulan'].strftime('%B %Y')}**."

    # Gabungkan semua narasi
    full_narrative = (f"Secara keseluruhan, tren {metric_label} cenderung {trend_type}."
                      f"{momentum_narrative}{yoy_narrative}{extrema_narrative}")

    return {
        'narrative': full_narrative,
        'trendline': trendline,
        'ma_line': ma_line,
        'p_value': p_value
    }
def display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    """Membuat grafik tren lengkap dengan garis tren, moving average, dan narasi analisis."""
    st.subheader(f"📈 Analisis Tren: {y_label}")
    
    # Panggil fungsi analisis baru
    analysis_result = analyze_trend_v3(df_data, y_col, y_label)

    # Buat grafik dasar
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)

    # Tambahkan Garis Tren (jika ada)
    if analysis_result.get('trendline') is not None:
        fig.add_scatter(
            x=df_data['Bulan'], 
            y=analysis_result['trendline'], 
            mode='lines', 
            name='Garis Tren', 
            line=dict(color='firebrick', dash='dash')
        )

    # Tambahkan Garis Moving Average (jika ada)
    if analysis_result.get('ma_line') is not None:
        fig.add_scatter(
            x=df_data['Bulan'], 
            y=analysis_result['ma_line'], 
            mode='lines', 
            name='Momentum (3-Bulan)', 
            line=dict(color='orange', dash='dot')
        )
    
    # Tampilkan grafik dan narasi
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 **Ringkasan Analisis:** {analysis_result.get('narrative', 'Analisis tidak tersedia.')}")
    
    # Tambahkan penjelasan statistik
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat penjelasan signifikansi statistik (p-value)"):
            st.markdown(f"**Nilai p-value** untuk garis tren ini adalah **`{p_value:.4f}`**.")
            st.markdown(f"Ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena faktor kebetulan.")
            if p_value < 0.05:
                st.success("✔️ Karena kemungkinan kebetulan rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else:
                st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik** dan mungkin hanya fluktuasi acak.")
    st.markdown("---")
# ==============================================================================
# FUNGSI-FUNGSI AGENT AI (TELAH DIMODIFIKASI)
# ==============================================================================
def run_operational_agent(df_sales, df_complaints, df_qa_qc, mapping):
    all_branches = sorted([str(b) for b in df_sales[mapping['branch']].unique() if pd.notna(b)])
    knowledge_base = get_operational_knowledge_base()
    results = []
    for branch in all_branches:
        sales_br = df_sales[df_sales[mapping['branch']] == branch]
        complaints_br = df_complaints[df_complaints[mapping['complaint_branch']] == branch]
        qa_qc_br = df_qa_qc[df_qa_qc[mapping['qa_branch']] == branch]
        daily_sales = sales_br.groupby(pd.Grouper(key=mapping['sales_date'], freq='D')).agg(DailySales=(mapping['nett_sales'], 'sum'), Transactions=(mapping['bill_number'], 'nunique')).reset_index()
        daily_sales['AOV'] = (daily_sales['DailySales'] / daily_sales['Transactions']).fillna(0)
        daily_complaints = complaints_br.groupby(pd.Grouper(key=mapping['complaint_date'], freq='D')).size().reset_index(name='Complaints')
        last_qa_qc_status = "BAIK"
        if not qa_qc_br.empty:
            last_audit_score = qa_qc_br.sort_values(mapping['qa_date']).iloc[-1][mapping['qa_score']]
            if last_audit_score < 75: last_qa_qc_status = "RENDAH"
        status = {
            "sales": analyze_short_term_metric_status(daily_sales, mapping['sales_date'], 'DailySales'),
            "transactions": analyze_short_term_metric_status(daily_sales, mapping['sales_date'], 'Transactions'),
            "aov": analyze_short_term_metric_status(daily_sales, mapping['sales_date'], 'AOV'),
            "complaints": analyze_short_term_metric_status(daily_complaints, mapping['complaint_date'], 'Complaints'),
            "qa_qc": last_qa_qc_status
        }
        causes = {rule["root_cause"] for rule in knowledge_base if rule["condition"](status)}
        results.append({"Toko": branch, "Penjualan (7d)": status["sales"], "Transaksi (7d)": status["transactions"], "AOV (7d)": status["aov"], "Komplain (7d)": status["complaints"], "Audit Terakhir": status["qa_qc"], "Analisis Operasional": ", ".join(causes) if causes else "Performa operasional stabil."})
    return pd.DataFrame(results)

def run_strategic_agent(df_sales, df_complaints, df_qa_qc, mapping):
    all_branches = sorted([str(b) for b in df_sales[mapping['branch']].unique() if pd.notna(b)])
    knowledge_base = get_strategic_knowledge_base()
    results = []
    for branch in all_branches:
        sales_br = df_sales[df_sales[mapping['branch']] == branch]
        complaints_br = df_complaints[df_complaints[mapping['complaint_branch']] == branch]
        qa_qc_br = df_qa_qc[df_qa_qc[mapping['qa_branch']] == branch]
        bill_agg = sales_br.groupby([mapping['bill_number'], pd.Grouper(key=mapping['sales_date'], freq='M')])[mapping['nett_sales']].sum().reset_index()
        monthly_aov_df = bill_agg.groupby(mapping['sales_date'])[mapping['nett_sales']].mean().reset_index().rename(columns={mapping['nett_sales']: 'AOV'})
        qa_qc_score = "TIDAK ADA DATA"
        if not qa_qc_br.empty:
            avg_score = qa_qc_br[mapping['qa_score']].mean()
            if avg_score < 75: qa_qc_score = "RENDAH"
            elif avg_score < 85: qa_qc_score = "SEDANG"
            else: qa_qc_score = "TINGGI"
        status = {
            "sales": analyze_long_term_metric_status(sales_br, mapping['sales_date'], mapping['nett_sales'], 'sum'),
            "transactions": analyze_long_term_metric_status(sales_br, mapping['sales_date'], mapping['bill_number'], 'nunique'),
            "aov": analyze_long_term_metric_status(monthly_aov_df, mapping['sales_date'], 'AOV', 'mean'),
            "complaints": analyze_long_term_metric_status(complaints_br, mapping['complaint_date'], mapping['complaint_branch'], 'count'),
            "qa_qc": qa_qc_score
        }
        causes = {rule["root_cause"] for rule in knowledge_base if rule["condition"](status)}
        results.append({"Toko": branch, "Tren Penjualan": status["sales"], "Tren Transaksi": status["transactions"], "Tren AOV": status["aov"], "Skor QA/QC": status["qa_qc"], "Tren Komplain": status["complaints"], "Analisis & Kemungkinan Akar Masalah": ", ".join(causes) if causes else "Tidak ada pola strategis signifikan."})
    return pd.DataFrame(results)

def analyze_short_term_metric_status(df, date_col, metric_col):
    """Menganalisis tren 7 hari vs 7 hari sebelumnya untuk deteksi dini."""
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
    """Aturan untuk masalah jangka pendek yang butuh reaksi cepat."""
    return [
        {"condition": lambda s: s["sales"] in ["MENURUN", "MENURUN TAJAM"], "root_cause": "Penurunan traffic atau masalah operasional mendadak minggu ini."},
        {"condition": lambda s: s["complaints"] in ["MENINGKAT", "MENINGKAT TAJAM"], "root_cause": "Terjadi insiden layanan atau masalah kualitas produk baru-baru ini."},
        {"condition": lambda s: s["qa_qc"] == "RENDAH", "root_cause": "Ditemukan kegagalan kepatuhan SOP signifikan pada audit terakhir."},
        {"condition": lambda s: s["aov"] in ["MENURUN", "MENURUN TAJAM"], "root_cause": "Promo mingguan kurang efektif atau staf gagal melakukan upselling."},
        {"condition": lambda s: s["sales"] in ["MENINGKAT TAJAM"] and s["aov"] in ["MENINGKAT TAJAM"], "root_cause": "Program promosi jangka pendek sangat berhasil."},
    ]
def analyze_long_term_metric_status(df, date_col, metric_col, agg_method='sum'):
    """Menganalisis tren bulanan (min 4 bulan) menggunakan regresi linear."""
    if df is None or df.empty or metric_col not in df.columns: return "TIDAK CUKUP DATA"
    df_resampled = df.set_index(date_col).resample('M')
    if agg_method == 'sum': monthly_df = df_resampled[metric_col].sum().reset_index()
    elif agg_method == 'mean': monthly_df = df_resampled[metric_col].mean().reset_index()
    elif agg_method == 'nunique': monthly_df = df_resampled[metric_col].nunique().reset_index()
    else: monthly_df = df_resampled.size().reset_index(name=metric_col)
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
    """Basis pengetahuan dari tabel untuk analisis jangka panjang."""
    return [
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "TREN MENINGKAT" in s["complaints"], "root_cause": "Layanan buruk menggerus loyalitas pelanggan (Bad service causing churn)."},
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"], "root_cause": "Operasional buruk berdampak negatif pada penjualan (Poor operations hurting loyalty)."},
        {"condition": lambda s: "TREN MENURUN" in s["sales"] and "RENDAH" in s["qa_qc"] and "MOMENTUM NEGATIF" in s["sales"], "root_cause": "Masalah fundamental pada operasional/rekrutmen (Poor purchasing/hiring loyalty)."},
        {"condition": lambda s: "TREN MENURUN" in s["transactions"] and "TREN MENURUN" in s["sales"], "root_cause": "Penurunan jumlah pengunjung (Reduced visitors/traffic) jadi pendorong utama penurunan penjualan."},
        {"condition": lambda s: "TREN MENINGKAT" in s["aov"] and "TREN MENURUN" not in s["sales"], "root_cause": "Strategi harga/promo berhasil meningkatkan nilai belanja (Good promo / Customers buy more)."},
        {"condition": lambda s: "TREN MENINGKAT" in s["sales"] and ("TINGGI" in s["qa_qc"] or "TREN MENURUN" in s["complaints"]), "root_cause": "Standar operasional yang baik mendorong pertumbuhan (Good operations standard / Good service)."},
        {"condition": lambda s: "MOMENTUM NEGATIF" in s["sales"] or "MOMENTUM NEGATIF" in s["transactions"], "root_cause": "Waspada! Performa melambat dalam 3 bulan terakhir, sinyal awal potensi penurunan tren."},
    ]
def display_agent_analysis(df_analysis, title, info_text):
    """Menampilkan hasil analisis dari AI Agent dalam bentuk tabel."""
    st.header(title)
    st.info(info_text)
    
    def style_status(val):
        color = "grey"
        if isinstance(val, str):
            if any(keyword in val for keyword in ["MENINGKAT", "TINGGI", "POSITIF", "BAIK"]): color = "#2ca02c"
            if any(keyword in val for keyword in ["MENURUN", "RENDAH", "NEGATIF"]): color = "#d62728"
        return f'color: {color}'
        
    df_display = df_analysis.set_index('Toko')
    # Temukan kolom terakhir untuk dikecualikan dari styling
    last_column_name = df_display.columns[-1]
    styled_df = df_display.style.apply(lambda col: col.map(style_status), subset=pd.IndexSlice[:, df_display.columns != last_column_name])
    st.dataframe(styled_df, use_container_width=True)

# ==============================================================================
# APLIKASI UTAMA STREAMLIT
# ==============================================================================
def main_app(user_name):
    st.sidebar.title("📤 Unggah Data Master")
    sales_file = st.sidebar.file_uploader("1. Unggah Penjualan Master (.feather)", type=["feather"])
    complaint_file = st.sidebar.file_uploader("2. Unggah Komplain Master (.feather)", type=["feather"])
    qa_qc_file = st.sidebar.file_uploader("3. Unggah QA/QC Master (.feather)", type=["feather"])

    if not all([sales_file, complaint_file, qa_qc_file]):
        st.info("👋 Selamat datang! Silakan unggah ketiga file master untuk memulai.")
        st.stop()
        
    df_sales = load_feather_file(sales_file)
    df_complaints = load_feather_file(complaint_file)
    df_qa_qc = load_feather_file(qa_qc_file)
    
    if df_sales is None or df_complaints is None or df_qa_qc is None: st.stop()

    # Tampilkan pemetaan kolom jika belum dikonfirmasi
    if not st.session_state.mapping_confirmed:
        display_column_mapper(df_sales, df_complaints, df_qa_qc)
        st.info("👈 Silakan konfigurasikan pemetaan kolom di sidebar, lalu klik 'Terapkan Pemetaan'.")
        st.stop()

    # Ambil mapping dari session state
    mapping = st.session_state.column_mapping

    # Pra-pemrosesan Data setelah mapping
    try:
        df_sales[mapping['sales_date']] = pd.to_datetime(df_sales[mapping['sales_date']])
        df_complaints[mapping['complaint_date']] = pd.to_datetime(df_complaints[mapping['complaint_date']])
        df_qa_qc[mapping['qa_date']] = pd.to_datetime(df_qa_qc[mapping['qa_date']])
    except Exception as e:
        st.error(f"Gagal mengonversi kolom tanggal. Pastikan kolom tanggal yang dipilih benar. Error: {e}")
        st.stop()
    
    # Reset mapping jika pengguna ingin mengubah
    if st.sidebar.button("Ubah Pemetaan Kolom"):
        st.session_state.mapping_confirmed = False
        st.rerun()

    st.sidebar.title("⚙️ Filter Global")
    ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
    unique_branches = sorted([str(b) for b in df_sales[mapping['branch']].unique() if pd.notna(b)])
    branch_options = [ALL_BRANCHES_OPTION] + unique_branches
    selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
    
    min_date = df_sales[mapping['sales_date']].min().date()
    max_date = df_sales[mapping['sales_date']].max().date()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) != 2: st.stop()
    start_date, end_date = date_range

    # Pemfilteran Data
    date_mask_sales = (df_sales[mapping['sales_date']].dt.date >= start_date) & (df_sales[mapping['sales_date']].dt.date <= end_date)
    df_sales_filtered = df_sales[date_mask_sales]
    date_mask_complaints = (df_complaints[mapping['complaint_date']].dt.date >= start_date) & (df_complaints[mapping['complaint_date']].dt.date <= end_date)
    df_complaints_filtered = df_complaints[date_mask_complaints]
    date_mask_qa_qc = (df_qa_qc[mapping['qa_date']].dt.date >= start_date) & (df_qa_qc[mapping['qa_date']].dt.date <= end_date)
    df_qa_qc_filtered = df_qa_qc[date_mask_qa_qc]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered[mapping['branch']] == selected_branch]
        df_complaints_filtered = df_complaints_filtered[df_complaints_filtered[mapping['complaint_branch']] == selected_branch]
        df_qa_qc_filtered = df_qa_qc_filtered[df_qa_qc_filtered[mapping['qa_branch']] == selected_branch]

    if df_sales_filtered.empty: st.warning("Tidak ada data penjualan untuk filter yang dipilih."); st.stop()

    st.title(f"Dasbor Analisis Holistik: {selected_branch}")
    st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    # Kalkulasi dan Tampilan Dashboard
    with st.spinner("Menganalisis data... 🤖"):
        monthly_agg = analyze_monthly_trends(df_sales_filtered, mapping)
        price_group_results = calculate_price_group_analysis(df_sales_filtered, mapping)
        df_branch_health = calculate_branch_health(df_sales_filtered, df_complaints_filtered, mapping)
        op_agent_results = run_operational_agent(df_sales, df_complaints, df_qa_qc, mapping)
        strat_agent_results = run_strategic_agent(df_sales, df_complaints, df_qa_qc, mapping)
    
    penjualan_tab, kualitas_tab, qa_qc_tab, agent_tab = st.tabs(["📈 Penjualan", "✅ Kualitas & Komplain", "⭐ Kepatuhan QA/QC", "🤖 AI Root Cause Agent"])
    
    with penjualan_tab:
        display_monthly_kpis(monthly_agg) # Fungsi ini tidak perlu mapping
        st.markdown("---")
        # Fungsi ini juga tidak perlu mapping karena menerima data agregat
        display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
        display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
        display_trend_chart_and_analysis(monthly_agg, 'AOV', 'AOV', 'green')
        st.markdown("---")
        display_price_group_analysis(price_group_results) # Tidak perlu mapping

    with kualitas_tab:
        display_branch_health(df_branch_health) # Tidak perlu mapping
        st.markdown("---")
        display_complaint_analysis(df_complaints_filtered, mapping)
        
    with qa_qc_tab:
        display_qa_qc_analysis(df_qa_qc_filtered, mapping)
        
    with agent_tab:
        st.header("Analisis Akar Masalah Otomatis oleh AI Agent")
        op_tab, strat_tab = st.tabs(["**⚡ Analisis Operasional (Mingguan)**", "**🎯 Analisis Strategis (3 Bulanan)**"])
        with op_tab:
            display_agent_analysis(op_agent_results, "Diagnosis Taktis Jangka Pendek", "Menganalisis performa 7 hari terakhir untuk deteksi cepat.")
        with strat_tab:
            display_agent_analysis(strat_agent_results, "Diagnosis Strategis Jangka Panjang", "Menganalisis tren bulanan untuk evaluasi fundamental.")

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