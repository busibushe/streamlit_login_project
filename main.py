import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Customer Insight Dashboard")

# --- JUDUL APLIKASI ---
st.title("📊 Customer Deep Insight Dashboard")
st.markdown("Unggah file CSV data pelanggan Anda untuk mendapatkan analisis mendalam secara otomatis.")

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_data
def load_data(file):
    """
    Fungsi cerdas untuk memuat data dari file CSV.
    Secara otomatis mencoba separator koma dan titik koma.
    """
    try:
        # 1. Coba baca dengan separator koma (standar)
        # on_bad_lines='skip' akan melewati baris yang bermasalah
        df = pd.read_csv(file, sep=',', on_bad_lines='skip', encoding='utf-8')
    except (pd.errors.ParserError, UnicodeDecodeError):
        try:
            # 2. Jika gagal, kembalikan pointer file ke awal dan coba dengan titik koma
            file.seek(0)
            df = pd.read_csv(file, sep=';', on_bad_lines='skip', encoding='utf-8')
        except (pd.errors.ParserError, UnicodeDecodeError):
            try:
                # 3. Jika masih gagal, coba dengan encoding 'latin1'
                file.seek(0)
                df = pd.read_csv(file, sep=',', on_bad_lines='skip', encoding='latin1')
            except Exception as e:
                st.error(f"Gagal total memuat file. Pastikan file Anda adalah CSV yang valid. Error: {e}")
                return None # Kembalikan None jika semua usaha gagal

    # Pastikan df berhasil dibuat sebelum melanjutkan
    if 'df' in locals():
        # Konversi kolom tanggal, 'coerce' akan mengubah error menjadi NaT (Not a Time)
        date_cols = ['DOB', 'Join Date', 'First Transaction Date', 'Last Transaction Date']
        for col in date_cols:
            # Cek apakah kolom ada sebelum konversi
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Membersihkan nama kolom dari spasi yang tidak perlu
        df.columns = df.columns.str.strip()

        # Mengisi nilai kosong pada kolom numerik dengan 0 dan kategorikal dengan 'Unknown'
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(0, inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna('Unknown', inplace=True)
            
        return df
    return None
def calculate_rfm(df):
    """Fungsi untuk menghitung Recency, Frequency, Monetary dan membuat segmentasi."""
    # Pastikan tidak ada tanggal transaksi di masa depan
    snapshot_date = df['Last Transaction Date'].max() + pd.Timedelta(days=1)
    
    # Hitung RFM
    rfm = df.groupby('Member Code').agg({
        'Last Transaction Date': lambda date: (snapshot_date - date.max()).days,
        'Lifetime Transaction': 'sum',
        'Lifetime Spend': 'sum'
    }).rename(columns={
        'Last Transaction Date': 'Recency',
        'Lifetime Transaction': 'Frequency',
        'Lifetime Spend': 'Monetary'
    })

    # Membuat skor RFM (1-4, di mana 4 adalah yang terbaik)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]) # Recency lebih rendah lebih baik
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Pemetaan skor ke segmen (bisa disesuaikan)
    segment_map = {
        r'[3-4][3-4][3-4]': '👑 Champions',
        r'[2-4][3-4][1-4]': '🫂 Loyal Customers',
        r'[3-4][1-2][1-4]': '🌱 Potential Loyalist',
        r'[4][1][1]': '👋 New Customers',
        r'[3][1][1]': 'Promising',
        r'[2-3][2-3][2-3]': '⚠️ Needs Attention',
        r'[2-3][1-2][1-2]': 'About to Sleep',
        r'[1][2-4][2-4]': '😥 At Risk',
        r'[1][1][1-4]': 'Hibernating',
        r'[1-2][1-2][1-2]': '💔 Lost'
    }

    rfm['Segment'] = rfm['RFM_Score'].replace(segment_map, regex=True)
    return rfm


# --- SIDEBAR UNTUK UPLOAD FILE ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # TAMBAHKAN PENGECEKAN INI
    if df is not None:
        # --- Membuat Tab untuk Navigasi ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Ringkasan Umum", 
            "👑 Segmentasi RFM", 
            "🎂 Demografi Pelanggan",
            "🍝 Produk & Cabang",
            "📈 Akuisisi Member"
        ])

        # --- TAB 1: RINGKASAN UMUM ---
        with tab1:
            st.header("Ringkasan Umum Bisnis")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pelanggan", f"{df['Member Code'].nunique():,}")
            col2.metric("Total Transaksi", f"{df['Lifetime Transaction'].sum():,}")
            col3.metric("Total Pendapatan", f"Rp {df['Lifetime Spend'].sum():,}")
            col4.metric("Status Member Aktif", f"{df[df['Member Status'] == 'Active'].shape[0]:,}")

            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribusi Status Member")
                status_counts = df['Member Status'].value_counts()
                fig_status = px.pie(status_counts, values=status_counts.values, names=status_counts.index, title="Proporsi Member Aktif vs Inaktif")
                st.plotly_chart(fig_status, use_container_width=True)
            
            with c2:
                st.subheader("Distribusi Gender")
                gender_counts = df['Gender'].value_counts()
                fig_gender = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title="Proporsi Gender Pelanggan")
                st.plotly_chart(fig_gender, use_container_width=True)

        # --- TAB 2: SEGMENTASI RFM ---
        with tab2:
            st.header("👑 Segmentasi Pelanggan (RFM Analysis)")
            st.markdown("""
            **RFM** adalah metode segmentasi berdasarkan 3 metrik utama:
            - **Recency (R):** Kapan terakhir kali pelanggan bertransaksi? (Semakin baru, semakin baik)
            - **Frequency (F):** Seberapa sering pelanggan bertransaksi? (Semakin sering, semakin baik)
            - **Monetary (M):** Berapa banyak uang yang dihabiskan pelanggan? (Semakin banyak, semakin baik)
            
            Dengan ini, kita bisa mengidentifikasi pelanggan terbaik (Champions) hingga pelanggan yang sudah lama tidak aktif (Lost).
            """)
            
            rfm_df = calculate_rfm(df)
            segment_counts = rfm_df['Segment'].value_counts()

            st.subheader("Jumlah Pelanggan per Segmen")
            fig_rfm = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, 
                            title="Distribusi Pelanggan Berdasarkan Segmen RFM",
                            labels={'x': 'Segmen', 'y': 'Jumlah Pelanggan'})
            fig_rfm.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_rfm, use_container_width=True)

            st.subheader("Detail Data per Segmen")
            st.dataframe(rfm_df.sort_values(by='Monetary', ascending=False))


        # --- TAB 3: DEMOGRAFI ---
        with tab3:
            st.header("🎂 Analisis Demografi Pelanggan")
            
            # Hitung Umur
            df['Age'] = (datetime.now() - df['DOB']).astype('<m8[Y]')
            # Buat Kelompok Umur
            bins = [0, 17, 24, 34, 44, 54, 64, 100]
            labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Total Belanja Berdasarkan Kelompok Umur")
                age_spend = df.groupby('Age Group')['Lifetime Spend'].sum().reset_index()
                fig_age = px.bar(age_spend, x='Age Group', y='Lifetime Spend', title="Total Belanja vs Kelompok Umur")
                st.plotly_chart(fig_age, use_container_width=True)
                
            with c2:
                st.subheader("Total Belanja Berdasarkan Gender")
                gender_spend = df.groupby('Gender')['Lifetime Spend'].sum().reset_index()
                fig_gender_spend = px.bar(gender_spend, x='Gender', y='Lifetime Spend', title="Total Belanja vs Gender")
                st.plotly_chart(fig_gender_spend, use_container_width=True)

        # --- TAB 4: PRODUK & CABANG ---
        with tab4:
            st.header("🍝 Kinerja Produk (Menu) dan Cabang")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("10 Menu Terfavorit")
                fav_menu = df['Favorite Menu'].value_counts().nlargest(10)
                fig_menu = px.bar(fav_menu, x=fav_menu.values, y=fav_menu.index, orientation='h', 
                                title="Top 10 Menu Favorit", labels={'x': 'Jumlah Pelanggan', 'y': 'Menu'})
                fig_menu.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_menu, use_container_width=True)
                
            with c2:
                st.subheader("10 Cabang Terfavorit")
                fav_branch = df['Favorite Branch'].value_counts().nlargest(10)
                fig_branch = px.bar(fav_branch, x=fav_branch.values, y=fav_branch.index, orientation='h',
                                    title="Top 10 Cabang Favorit", labels={'x': 'Jumlah Pelanggan', 'y': 'Cabang'})
                fig_branch.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_branch, use_container_width=True)

        # --- TAB 5: AKUISISI MEMBER ---
        with tab5:
            st.header("📈 Analisis Akuisisi dan Pertumbuhan Member")
            
            # Resample data berdasarkan bulan join
            df_join = df.set_index('Join Date')
            monthly_joins = df_join.resample('M').size().rename('Jumlah Member Baru')
            
            fig_join = px.line(monthly_joins, x=monthly_joins.index, y=monthly_joins.values,
                            title="Pertumbuhan Member Baru per Bulan",
                            labels={'x': 'Bulan', 'y': 'Jumlah Member Baru'})
            st.plotly_chart(fig_join, use_container_width=True)

else:
    st.info("☝️ Silakan unggah file CSV Anda pada sidebar untuk memulai analisis.")
    st.image("https://i.imgur.com/3ySJs1j.png", caption="Contoh Dashboard yang Akan Dihasilkan")