import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ==============================================================================
# 1. KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dashboard F&B Holistik",
    page_icon="🚀",
    layout="wide"
)

# ==============================================================================
# 2. SEMUA FUNGSI HELPER
# (Semua fungsi Anda dikumpulkan di sini)
# ==============================================================================

# --- FUNGSI PEMUATAN DATA ---
@st.cache_data
def load_feather_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        df = pd.read_feather(uploaded_file)
        if 'Sales Date' in df.columns: df['Sales Date'] = pd.to_datetime(df['Sales Date'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}"); return None

@st.cache_data(ttl=600)
def load_qa_qc_from_gsheet(url):
    if not url or "docs.google.com" not in url:
        st.warning("URL Google Sheet QA/QC tidak valid."); return pd.DataFrame()
    try:
        url = url.replace("/edit?gid=", "/export?format=csv&gid=").replace("/edit#gid=", "/export?format=csv&gid=")
        df = pd.read_csv(url, on_bad_lines='warn')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets. Error: {e}"); return pd.DataFrame()

def process_qa_qc_data(df_raw):
    if df_raw.empty: return pd.DataFrame()
    df_processed = df_raw.copy()
    df_processed = df_processed.rename(columns={"Lokasi Store": "Branch", "Tanggal Visit": "Sales Date"})
    try:
        df_processed['Sales Date'] = pd.to_datetime(df_processed['Sales Date'], dayfirst=True, errors='coerce')
    except Exception:
        df_processed['Sales Date'] = pd.to_datetime(df_processed['Sales Date'], errors='coerce')
    try:
        first_question_col_name = [col for col in df_processed.columns if 'KEBERSIHAN' in col][0]
        first_col_index = df_processed.columns.get_loc(first_question_col_name)
        question_cols = df_processed.columns[first_col_index:].tolist()
    except IndexError:
        return pd.DataFrame()
    results = []
    for index, row in df_processed.iterrows():
        answers = row[question_cols].astype(str).str.upper()
        total_y = (answers == 'Y').sum()
        total_n = (answers == 'N').sum()
        total_points = total_y + total_n
        score = (total_y / total_points * 100) if total_points > 0 else 0
        results.append({"Branch": row["Branch"], "Sales Date": row["Sales Date"], "Skor Kepatuhan": score})
    return pd.DataFrame(results).dropna(subset=['Sales Date'])

# --- FUNGSI ANALISIS & VISUALISASI ---
def display_detailed_qa_qc_analysis(df_qa_qc_raw):
    st.subheader("Analisis Mendalam Kepatuhan Standar (Deep Dive)")
    if df_qa_qc_raw.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih."); return
    try:
        first_question_col_name = [col for col in df_qa_qc_raw.columns if 'KEBERSIHAN' in col][0]
        first_col_index = df_qa_qc_raw.columns.get_loc(first_question_col_name)
        question_cols = df_qa_qc_raw.columns[first_col_index:].tolist()
    except (IndexError, KeyError):
        st.error("Format kolom data QA/QC tidak dikenali."); return
    melted_df = df_qa_qc_raw[question_cols].melt()
    counts = melted_df['value'].astype(str).str.upper().value_counts()
    total_y, total_n = counts.get('Y', 0), counts.get('N', 0)
    total_points = total_y + total_n
    overall_score = (total_y / total_points * 100) if total_points > 0 else 0
    st.markdown("##### Ringkasan Audit")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skor Kepatuhan Gabungan", f"{overall_score:.1f}%")
    col2.metric("Total Poin Diperiksa", f"{total_points}")
    col3.metric("✔️ Poin Terpenuhi (Y)", f"{total_y}")
    col4.metric("❌ Poin Pelanggaran (N)", f"{total_n}")
    st.markdown("---")
    st.markdown("##### Rincian Skor per Kategori")
    categories = sorted(list(set([col.split('[')[0].strip() for col in question_cols])))
    cat_col1, cat_col2 = st.columns(2)
    for i, category in enumerate(categories):
        cat_cols = [col for col in question_cols if col.startswith(category)]
        cat_melted = df_qa_qc_raw[cat_cols].melt()
        cat_counts = cat_melted['value'].astype(str).str.upper().value_counts()
        cat_y, cat_n = cat_counts.get('Y', 0), cat_counts.get('N', 0)
        cat_total = cat_y + cat_n
        cat_score = (cat_y / cat_total * 100) if cat_total > 0 else 0
        target_col = cat_col1 if i % 2 == 0 else cat_col2
        with target_col:
            st.text(category); st.progress(int(cat_score)); st.caption(f"Skor: {cat_score:.1f}% ({cat_y}/{cat_total})")
    st.markdown("---")
    st.markdown("##### Daftar Temuan (Poin 'N')")
    findings_df = df_qa_qc_raw[['Lokasi Store', 'Tanggal Visit'] + question_cols].melt(id_vars=['Lokasi Store', 'Tanggal Visit'], var_name='Detail Pemeriksaan', value_name='Hasil')
    findings_df = findings_df[findings_df['Hasil'].astype(str).str.upper() == 'N']
    if findings_df.empty:
        st.success("🎉 Hebat! Tidak ada temuan atau pelanggaran pada data yang dipilih.")
    else:
        findings_df['Kategori'] = findings_df['Detail Pemeriksaan'].apply(lambda x: x.split('[')[0].strip())
        st.dataframe(findings_df[['Lokasi Store', 'Tanggal Visit', 'Kategori', 'Detail Pemeriksaan']], use_container_width=True)

# (Sisa fungsi-fungsi analisis Anda yang lain ditempatkan di sini... saya persingkat agar tidak terlalu panjang)
# ...
# def analyze_monthly_trends(...):
# def display_monthly_kpis(...):
# def display_trend_chart_and_analysis(...):
# ... dan seterusnya ...

# ==============================================================================
# 3. FUNGSI UTAMA APLIKASI (HANYA UNTUK TAMPILAN DASHBOARD)
# ==============================================================================
def main_dashboard():
    """Fungsi ini hanya fokus untuk menampilkan dashboard setelah login berhasil."""
    
    st.sidebar.title("📤 Sumber Data")
    sales_file = st.sidebar.file_uploader("1. Unggah Penjualan Master (.feather)", type=["feather"])
    complaint_file = st.sidebar.file_uploader("2. Unggah Komplain Master (.feather)", type=["feather"])
    qa_qc_url = st.sidebar.text_input("3. URL Google Sheet QA/QC", placeholder="Masukkan URL Google Sheet Anda di sini...")

    if sales_file is None or complaint_file is None:
        st.info("👋 Selamat datang! Silakan unggah file master penjualan dan komplain, lalu masukkan URL Google Sheet QA/QC.")
        st.stop()
        
    df_sales = load_feather_file(sales_file)
    df_complaints = load_feather_file(complaint_file)
    df_qa_qc_raw = load_qa_qc_from_gsheet(qa_qc_url)
    df_qa_qc_processed = process_qa_qc_data(df_qa_qc_raw)
    
    if df_sales is None or df_complaints is None or df_qa_qc_processed.empty:
        st.error("Gagal memuat semua sumber data. Pastikan file Feather valid dan URL Google Sheet benar serta berisi data."); 
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

    # Filter semua dataframe
    df_sales_filtered = df_sales[(df_sales['Sales Date'].dt.date >= start_date) & (df_sales['Sales Date'].dt.date <= end_date)]
    df_complaints_filtered = df_complaints[(df_complaints['Sales Date'].dt.date >= start_date) & (df_complaints['Sales Date'].dt.date <= end_date)]
    df_qa_qc_filtered_processed = df_qa_qc_processed[(df_qa_qc_processed['Sales Date'].dt.date >= start_date) & (df_qa_qc_processed['Sales Date'].dt.date <= end_date)]
    
    # Perlu penanganan khusus untuk memfilter df_qa_qc_raw karena nama kolom tanggalnya berbeda
    df_qa_qc_raw['Sales Date'] = pd.to_datetime(df_qa_qc_raw['Tanggal Visit'], dayfirst=True, errors='coerce')
    df_qa_qc_filtered_raw = df_qa_qc_raw[df_qa_qc_raw['Sales Date'].dt.date.between(start_date, end_date)]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered['Branch'] == selected_branch]
        df_complaints_filtered = df_complaints_filtered[df_complaints_filtered['Branch'] == selected_branch]
        df_qa_qc_filtered_processed = df_qa_qc_filtered_processed[df_qa_qc_filtered_processed['Branch'] == selected_branch]
        df_qa_qc_filtered_raw = df_qa_qc_filtered_raw[df_qa_qc_filtered_raw['Lokasi Store'] == selected_branch]

    if df_sales_filtered.empty: st.warning("Tidak ada data penjualan untuk filter yang dipilih."); st.stop()

    st.title(f"Dashboard Analisis Holistik: {selected_branch}")
    st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    # (Logika tab Anda tidak berubah, saya persingkat juga)
    tabs = st.tabs(["📈 Penjualan", "✅ Kualitas & Komplain", "⭐ Ringkasan QA/QC", "🔍 Detail Audit QA/QC", "🤖 AI Root Cause Agent"])
    with tabs[2]: # Ringkasan QA/QC
        # display_qa_qc_analysis(df_qa_qc_filtered_processed) # panggil fungsi ringkasan
        st.write("Konten Ringkasan QA/QC")
    with tabs[3]: # Detail QA/QC
        display_detailed_qa_qc_analysis(df_qa_qc_filtered_raw)
    # ... Logika untuk tab-tab lain ...


# ==============================================================================
# 4. ALUR UTAMA APLIKASI (MAIN EXECUTION FLOW)
# ==============================================================================

# Cek jika mode development (tanpa login) atau production (dengan login)
if 'credentials' not in st.secrets or 'cookie' not in st.secrets:
    # --- Mode Development (langsung tampilkan dashboard) ---
    st.sidebar.warning("Mode Developer Aktif (tanpa login)")
    main_dashboard()

else:
    # --- Mode Production (dengan proses login) ---
    config = {
        'credentials': dict(st.secrets['credentials']),
        'cookie': dict(st.secrets['cookie'])
    }
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Render form login
    name, authentication_status, username = authenticator.login("Login", "main")

    # Logika setelah proses login
    if authentication_status:
        # --- Jika login BERHASIL ---
        authenticator.logout("Logout", "sidebar", key='unique_logout_button') # Tombol logout ditaruh di sini
        st.sidebar.success(f"Login sebagai **{name}**")
        main_dashboard() # Panggil fungsi untuk menampilkan konten utama
        
    elif authentication_status is False:
        # --- Jika login GAGAL ---
        st.error("Username atau password salah.")
        
    elif authentication_status is None:
        # --- Jika BELUM login ---
        st.warning("Silakan masukkan username dan password Anda.")