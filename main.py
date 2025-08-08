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
# ==============================================================================

# --- FUNGSI PEMUATAN & PEMBERSIHAN DATA ---
@st.cache_data
def load_feather_file(uploaded_file):
    """Memuat satu file Feather dan memastikan tipe data tanggal."""
    if uploaded_file is None: return None
    try:
        df = pd.read_feather(uploaded_file)
        if 'Sales Date' in df.columns: df['Sales Date'] = pd.to_datetime(df['Sales Date'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}"); return None

@st.cache_data(ttl=600)
def load_qa_qc_from_gsheet(url):
    """Hanya memuat data mentah dari Google Sheet."""
    if not url or "docs.google.com" not in url:
        st.warning("URL Google Sheet QA/QC tidak valid."); return pd.DataFrame()
    try:
        url = url.replace("/edit?gid=", "/export?format=csv&gid=").replace("/edit#gid=", "/export?format=csv&gid=")
        df = pd.read_csv(url, on_bad_lines='warn')
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets. Error: {e}"); return pd.DataFrame()

def clean_qa_qc_data(df_raw):
    """
    Fungsi sentral untuk membersihkan dan menstandarisasi data QA/QC.
    Semua proses cleaning terjadi di sini.
    """
    if df_raw.empty: return pd.DataFrame()
    
    df_clean = df_raw.copy()
    # 1. Bersihkan nama kolom
    df_clean.columns = df_clean.columns.str.strip().str.lower()
    
    # 2. Rename kolom utama
    rename_map = {"lokasi store": "branch", "tanggal visit": "sales date"}
    df_clean = df_clean.rename(columns=rename_map)
    
    # 3. Validasi dan konversi tipe data tanggal
    if 'sales date' not in df_clean.columns:
        # Tidak menampilkan error di sini agar tidak duplikat jika URL kosong
        return pd.DataFrame()
    df_clean['sales date'] = pd.to_datetime(df_clean['sales date'], dayfirst=True, errors='coerce')
    
    return df_clean.dropna(subset=['sales date'])

def process_qa_qc_for_summary(df_clean):
    """
    Mengolah data yang SUDAH BERSIH menjadi format ringkas untuk ringkasan/summary.
    Fungsi ini tidak lagi melakukan cleaning.
    """
    if df_clean.empty: return pd.DataFrame()
    
    try:
        first_question_col_name = [col for col in df_clean.columns if 'kebersihan' in col][0]
        first_col_index = df_clean.columns.get_loc(first_question_col_name)
        question_cols = df_clean.columns[first_col_index:].tolist()
    except IndexError:
        st.error("Format kolom data QA/QC tidak dikenali."); return pd.DataFrame()

    results = []
    for index, row in df_clean.iterrows():
        answers = row[question_cols].astype(str).str.upper()
        total_y = (answers == 'Y').sum()
        total_n = (answers == 'N').sum()
        total_points = total_y + total_n
        score = (total_y / total_points * 100) if total_points > 0 else 0
        results.append({"Branch": row["branch"], "Sales Date": row["sales date"], "Skor Kepatuhan": score})
        
    return pd.DataFrame(results)

# --- FUNGSI ANALISIS & VISUALISASI ---
def display_qa_qc_summary(df_processed):
    """Menampilkan ringkasan strategis QA/QC."""
    st.subheader("Ringkasan Kepatuhan Standar (QA/QC)")
    if df_processed.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih."); return
    
    avg_score = df_processed['Skor Kepatuhan'].mean()
    st.metric("Rata-rata Skor Kepatuhan (Compliance Score)", f"{avg_score:.1f}%")
    
    st.markdown("##### Tren Skor Kepatuhan per Audit")
    fig1 = px.line(df_processed.sort_values('Sales Date'), x='Sales Date', y='Skor Kepatuhan', color='Branch', markers=True, title="Tren Skor Kepatuhan per Audit")
    fig1.update_yaxes(range=[0, 105])
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("##### Perbandingan Rata-rata Skor per Cabang")
    branch_avg_score = df_processed.groupby('Branch')['Skor Kepatuhan'].mean().reset_index().sort_values('Skor Kepatuhan', ascending=False)
    fig2 = px.bar(branch_avg_score, x='Branch', y='Skor Kepatuhan', title="Perbandingan Rata-rata Skor Kepatuhan antar Cabang", color='Skor Kepatuhan', color_continuous_scale='Greens')
    st.plotly_chart(fig2, use_container_width=True)

def display_detailed_qa_qc_analysis(df_clean):
    """Menampilkan analisis mendalam dari data QA/QC yang SUDAH BERSIH."""
    st.subheader("Analisis Mendalam Kepatuhan Standar (Deep Dive)")
    if df_clean.empty:
        st.warning("Tidak ada data audit QA/QC untuk periode dan cabang yang dipilih."); return
        
    try:
        first_question_col_name = [col for col in df_clean.columns if 'kebersihan' in col][0]
        first_col_index = df_clean.columns.get_loc(first_question_col_name)
        question_cols = df_clean.columns[first_col_index:].tolist()
    except (IndexError, KeyError):
        st.error("Format kolom data QA/QC tidak dikenali."); return
        
    melted_df = df_clean[question_cols].melt()
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
        cat_melted = df_clean[cat_cols].melt()
        cat_counts = cat_melted['value'].astype(str).str.upper().value_counts()
        cat_y, cat_n = cat_counts.get('Y', 0), cat_counts.get('N', 0)
        cat_total = cat_y + cat_n
        cat_score = (cat_y / cat_total * 100) if cat_total > 0 else 0
        target_col = cat_col1 if i % 2 == 0 else cat_col2
        with target_col:
            st.text(category); st.progress(int(cat_score)); st.caption(f"Skor: {cat_score:.1f}% ({cat_y}/{cat_total})")
    st.markdown("---")
    
    st.markdown("##### Daftar Temuan (Poin 'N')")
    findings_df = df_clean[['branch', 'sales date'] + question_cols].melt(id_vars=['branch', 'sales date'], var_name='Detail Pemeriksaan', value_name='Hasil')
    findings_df = findings_df[findings_df['Hasil'].astype(str).str.upper() == 'N']
    if findings_df.empty:
        st.success("🎉 Hebat! Tidak ada temuan atau pelanggaran pada data yang dipilih.")
    else:
        findings_df['Kategori'] = findings_df['Detail Pemeriksaan'].apply(lambda x: x.split('[')[0].strip())
        st.dataframe(findings_df[['branch', 'sales date', 'Kategori', 'Detail Pemeriksaan']], use_container_width=True)

# (Sisa fungsi-fungsi analisis lainnya bisa ditambahkan di sini jika ada)

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
    
    # --- ALUR BARU YANG LEBIH BAIK UNTUK QA/QC ---
    df_qa_qc_raw = load_qa_qc_from_gsheet(qa_qc_url)
    df_qa_qc_clean = clean_qa_qc_data(df_qa_qc_raw)
    df_qa_qc_processed = process_qa_qc_for_summary(df_qa_qc_clean)
    
    if df_sales is None or df_complaints is None or df_qa_qc_clean.empty:
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
    df_sales_filtered = df_sales[df_sales['Sales Date'].dt.date.between(start_date, end_date)]
    df_complaints_filtered = df_complaints[df_complaints['Sales Date'].dt.date.between(start_date, end_date)]
    df_qa_qc_filtered_clean = df_qa_qc_clean[df_qa_qc_clean['sales date'].dt.date.between(start_date, end_date)]
    df_qa_qc_filtered_processed = df_qa_qc_processed[df_qa_qc_processed['Sales Date'].dt.date.between(start_date, end_date)]

    if selected_branch != ALL_BRANCHES_OPTION:
        df_sales_filtered = df_sales_filtered[df_sales_filtered['Branch'] == selected_branch]
        df_complaints_filtered = df_complaints_filtered[df_complaints_filtered['Branch'] == selected_branch]
        df_qa_qc_filtered_clean = df_qa_qc_filtered_clean[df_qa_qc_filtered_clean['branch'] == selected_branch]
        df_qa_qc_filtered_processed = df_qa_qc_filtered_processed[df_qa_qc_filtered_processed['Branch'] == selected_branch]

    if df_sales_filtered.empty: st.warning("Tidak ada data penjualan untuk filter yang dipilih."); st.stop()

    st.title(f"Dashboard Analisis Holistik: {selected_branch}")
    st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

    tabs = st.tabs(["📈 Penjualan", "✅ Kualitas & Komplain", "⭐ Ringkasan QA/QC", "🔍 Detail Audit QA/QC"])
    
    with tabs[0]: # Penjualan
        st.write("Konten Analisis Penjualan di sini...")
        # Ganti dengan pemanggilan fungsi analisis penjualan Anda, contoh:
        # display_sales_analysis(df_sales_filtered)

    with tabs[1]: # Kualitas & Komplain
        st.write("Konten Kualitas & Komplain di sini...")
        # Ganti dengan pemanggilan fungsi analisis komplain Anda, contoh:
        # display_complaint_analysis(df_complaints_filtered)

    with tabs[2]: # Ringkasan QA/QC
        display_qa_qc_summary(df_qa_qc_filtered_processed)
        
    with tabs[3]: # Detail QA/QC
        display_detailed_qa_qc_analysis(df_qa_qc_filtered_clean)
        
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
    config = {'credentials': dict(st.secrets['credentials']),'cookie': dict(st.secrets['cookie'])}
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status:
        authenticator.logout("Logout", "sidebar", key='unique_logout_button')
        st.sidebar.success(f"Login sebagai **{name}**")
        main_dashboard()
    elif authentication_status is False:
        st.error("Username atau password salah.")
    elif authentication_status is None:
        st.warning("Silakan masukkan username dan password Anda.")