import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Customer Insight Dashboard")

# --- JUDUL APLIKASI ---
st.title("📊 Customer Deep Insight Dashboard")
st.markdown("Unggah file **CSV** atau **XLSX** data pelanggan Anda untuk mendapatkan analisis mendalam.")

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_data
def load_data(uploaded_file):
    """
    Fungsi cerdas untuk memuat data dari file CSV atau XLSX.
    Secara otomatis menstandardisasi nama kolom.
    """
    df = None
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            # Logika untuk membaca file CSV
            try:
                df = pd.read_csv(uploaded_file, sep=',', on_bad_lines='skip', encoding='utf-8')
            except pd.errors.ParserError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip', encoding='utf-8')
        
        elif file_extension in ['xlsx', 'xls']:
            # Logika untuk membaca file Excel
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        else:
            st.error("Format file tidak didukung. Harap unggah file CSV atau XLSX.")
            return None

    except Exception as e:
        st.error(f"Gagal memuat file. Pastikan file Anda valid. Error: {e}")
        return None

    if df is not None:
        # Membersihkan dan menstandardisasi nama kolom (berlaku untuk CSV dan XLSX)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
        
        st.sidebar.subheader("Nama Kolom Terdeteksi:")
        st.sidebar.code(df.columns.tolist())

        date_cols = ['dob', 'join_date', 'first_transaction_date', 'last_transaction_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(0, inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna('Unknown', inplace=True)
            
        return df
    return None

def calculate_rfm(df):
    """Fungsi untuk menghitung Recency, Frequency, Monetary dan membuat segmentasi."""
    required_cols = ['last_transaction_date', 'lifetime_transaction', 'lifetime_spend', 'member_code']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Kolom yang dibutuhkan untuk Analisis RFM tidak ditemukan. Dibutuhkan: {', '.join(required_cols)}")
        return None

    snapshot_date = df['last_transaction_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('member_code').agg({
        'last_transaction_date': lambda date: (snapshot_date - date.max()).days,
        'lifetime_transaction': 'sum',
        'lifetime_spend': 'sum'
    }).rename(columns={
        'last_transaction_date': 'Recency',
        'lifetime_transaction': 'Frequency',
        'lifetime_spend': 'Monetary'
    })

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    segment_map = {
        r'[3-4][3-4][3-4]': '👑 Champions', r'[2-4][3-4][1-4]': '🫂 Loyal Customers',
        r'[3-4][1-2][1-4]': '🌱 Potential Loyalist', r'[4][1][1]': '👋 New Customers',
        r'[3][1][1]': 'Promising', r'[2-3][2-3][2-3]': '⚠️ Needs Attention',
        r'[2-3][1-2][1-2]': 'About to Sleep', r'[1][2-4][2-4]': '😥 At Risk',
        r'[1][1][1-4]': 'Hibernating', r'[1-2][1-2][1-2]': '💔 Lost'
    }
    rfm['Segment'] = rfm['RFM_Score'].replace(segment_map, regex=True)
    return rfm

# --- SIDEBAR UNTUK UPLOAD FILE ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Pilih file CSV atau XLSX", type=["csv", "xlsx"])

# --- KONTEN UTAMA APLIKASI ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Ringkasan Umum", "👑 Segmentasi RFM", "🎂 Demografi Pelanggan",
            "🍝 Produk & Cabang", "📈 Akuisisi Member"
        ])

        with tab1:
            st.header("Ringkasan Umum Bisnis")
            # ... (kode di tab ini tidak berubah) ...
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pelanggan", f"{df['member_code'].nunique():,}")
            col2.metric("Total Transaksi", f"{df['lifetime_transaction'].sum():,}")
            col3.metric("Total Pendapatan", f"Rp {df['lifetime_spend'].sum():,}")
            col4.metric("Status Member Aktif", f"{df[df['member_status'] == 'Active'].shape[0]:,}")
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribusi Status Member")
                status_counts = df['member_status'].value_counts()
                fig_status = px.pie(status_counts, values=status_counts.values, names=status_counts.index, title="Proporsi Member Aktif vs Inaktif")
                st.plotly_chart(fig_status, use_container_width=True)
            with c2:
                st.subheader("Distribusi Gender")
                gender_counts = df['gender'].value_counts()
                fig_gender = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title="Proporsi Gender Pelanggan")
                st.plotly_chart(fig_gender, use_container_width=True)

        with tab2:
            st.header("👑 Segmentasi Pelanggan (RFM Analysis)")
            # ... (kode di tab ini tidak berubah) ...
            st.markdown("RFM adalah metode segmentasi berdasarkan Recency, Frequency, dan Monetary.")
            rfm_df = calculate_rfm(df)
            if rfm_df is not None:
                segment_counts = rfm_df['Segment'].value_counts()
                st.subheader("Jumlah Pelanggan per Segmen")
                fig_rfm = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, title="Distribusi Pelanggan Berdasarkan Segmen RFM", labels={'x': 'Segmen', 'y': 'Jumlah Pelanggan'})
                fig_rfm.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_rfm, use_container_width=True)
                st.subheader("Detail Data per Segmen")
                st.dataframe(rfm_df.sort_values(by='Monetary', ascending=False))

        with tab3:
            st.header("🎂 Analisis Demografi Pelanggan")
            if 'dob' in df.columns:
                
                # --- PERBAIKAN DI SINI ---
                st.subheader("Perhitungan Umur (Age Calculation)")
                # Hitung selisih waktu. Hasilnya akan mengandung NaT jika 'dob' kosong/invalid
                time_diff = datetime.now() - df['dob']

                # Konversi ke tahun. Bagi dengan 365.25 untuk akurasi. 
                # NaT akan otomatis menjadi NaN (Not a Number) yang bisa di-handle
                age_float = time_diff.dt.days / 365.25

                # Ganti nilai NaN (umur yang tidak bisa dihitung) dengan 0 atau nilai lain
                # Lalu konversi semua umur menjadi angka bulat (integer)
                df['age'] = age_float.fillna(0).astype(int)
                # --- AKHIR PERBAIKAN ---

                # Buat Kelompok Umur
                bins = [0, 17, 24, 34, 44, 54, 64, 150] # Batas atas dinaikkan
                labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Total Belanja Berdasarkan Kelompok Umur")
                    # Filter data dimana age_group tidak kosong
                    age_spend = df[df['age_group'].notna()].groupby('age_group')['lifetime_spend'].sum().reset_index()
                    fig_age = px.bar(age_spend, x='age_group', y='lifetime_spend', title="Total Belanja vs Kelompok Umur")
                    st.plotly_chart(fig_age, use_container_width=True)
                with c2:
                    st.subheader("Total Belanja Berdasarkan Gender")
                    gender_spend = df.groupby('gender')['lifetime_spend'].sum().reset_index()
                    fig_gender_spend = px.bar(gender_spend, x='gender', y='lifetime_spend', title="Total Belanja vs Gender")
                    st.plotly_chart(fig_gender_spend, use_container_width=True)
            else:
                st.warning("Kolom 'dob' tidak ditemukan untuk analisis demografi.")

        with tab4:
            st.header("🍝 Kinerja Produk (Menu) dan Cabang")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("10 Menu Terfavorit")
                fav_menu = df['favorite_menu'].value_counts().nlargest(10)
                fig_menu = px.bar(fav_menu, x=fav_menu.values, y=fav_menu.index, orientation='h', title="Top 10 Menu Favorit", labels={'x': 'Jumlah Pelanggan', 'y': 'Menu'})
                fig_menu.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_menu, use_container_width=True)
            with c2:
                st.subheader("10 Cabang Terfavorit")
                fav_branch = df['favorite_branch'].value_counts().nlargest(10)
                fig_branch = px.bar(fav_branch, x=fav_branch.values, y=fav_branch.index, orientation='h', title="Top 10 Cabang Favorit", labels={'x': 'Jumlah Pelanggan', 'y': 'Cabang'})
                fig_branch.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_branch, use_container_width=True)
        
        with tab5:
            st.header("📈 Analisis Akuisisi dan Pertumbuhan Member")
            if 'join_date' in df.columns:
                df_join = df.set_index('join_date')
                monthly_joins = df_join.resample('M').size().rename('Jumlah Member Baru')
                fig_join = px.line(monthly_joins, x=monthly_joins.index, y=monthly_joins.values, title="Pertumbuhan Member Baru per Bulan", labels={'x': 'Bulan', 'y': 'Jumlah Member Baru'})
                st.plotly_chart(fig_join, use_container_width=True)
            else:
                st.warning("Kolom 'join_date' tidak ditemukan untuk analisis akuisisi.")

else:
    st.info("☝️ Silakan unggah file CSV atau XLSX Anda pada sidebar untuk memulai analisis.")