import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ==============================================================================
# Konfigurasi Halaman Dashboard
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Perbandingan Tim X",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# Fungsi Helper
# ==============================================================================

@st.cache_data
def transform_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan dan mentransformasi data penjualan mentah."""
    df = df.copy()
    if "Sales Date" in df.columns:
        df.rename(columns={"Sales Date": "Date"}, inplace=True)
    
    # Pastikan kolom-kolom penting ada dan bertipe string
    for col in ['Branch', 'Menu Category Detail', 'Menu', 'Payment Method', 'Visit Purpose', 'Bill Number']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            # Jika kolom tidak ada, buat kolom kosong untuk menghindari error
            df[col] = ''

    df['Branch'] = df['Branch'].str.strip().replace({
        "Bandung Cheesecuit, Setiabudhi": "Setiabudi", 
        "Bandung Cheesecuit Cirebon": "Cipto Cirebon"
    }).str.replace(r"^Bandung Cheesecuit ", "", regex=True)

    conditions = [
        df['Menu'].str.contains('Bongsor Brownies Longsor', na=False),
        df['Menu'].str.contains('Kardus|Totebag', na=False, regex=True),
        df['Menu'].str.contains('Hampers', na=False),
        df['Menu'].str.startswith('Cheesecuit Bogel', na=False),
        df['Menu'].str.startswith('Cheesecuit', na=False),
        df['Menu'].str.contains('Lasagna|Puding|Brownies', na=False, regex=True),
    ]
    choices = ['Bongsor', 'Merchandise', 'Hampers', 'Bogel', 'Reguler', 'Dessert']
    df['Menu Category Detail'] = np.select(conditions, choices, default='Lainnya')
    
    # Konversi tipe data numerik dan tanggal
    numeric_cols = ["Qty", "Total After Bill Discount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    date_cols = ["Date", "Order Time", "Sales Date Out"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df

@st.cache_data
def load_raw_data(uploaded_files):
    """Membaca dan menggabungkan file-file data mentah yang diunggah."""
    try:
        list_of_dfs = []
        for file in uploaded_files:
            df_single = pd.read_excel(file, header=11)
            if len(df_single) > 4:
                df_single = df_single.iloc[:-4]
            list_of_dfs.append(df_single)
        
        if not list_of_dfs:
            return None

        df_raw = pd.concat(list_of_dfs, ignore_index=True)
        return transform_sales_data(df_raw)
    except Exception as e:
        st.error(f"Error: Gagal membaca file data mentah. Detail: {e}")
        return None

@st.cache_data
def load_kpi_data(uploaded_kpi_file):
    """Membaca file data KPI tambahan."""
    try:
        df = pd.read_excel(uploaded_kpi_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error: Gagal membaca file KPI tambahan. Detail: {e}")
        return None

def format_rupiah(value):
    """Mengubah angka menjadi format Rupiah."""
    return f"Rp {value:,.0f}".replace(',', '.')

# ==============================================================================
# Tampilan Utama Dashboard
# ==============================================================================

st.title("📊 Dashboard Monitoring & Perbandingan Kinerja Tim X")

# --- Bagian Upload File ---
col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    uploaded_files = st.file_uploader(
        "1. Unggah Data Transaksi Mentah (`Sales Recapitulation...xlsx`)",
        type=['xlsx'],
        accept_multiple_files=True
    )
with col_upload2:
    uploaded_kpi_file = st.file_uploader(
        "2. Unggah Data KPI Tambahan (`data_kpi_tambahan.xlsx`)",
        type=['xlsx']
    )

if uploaded_files:
    raw_data = load_raw_data(uploaded_files)

    if raw_data is not None:
        # ======================================================================
        # Sidebar untuk Filter Tanggal
        # ======================================================================
        st.sidebar.header("⚙️ Filter Data")
        min_date = raw_data['Date'].min().date()
        max_date = raw_data['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal Analisis",
            value=(max_date - timedelta(days=6), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) != 2:
            st.warning("Silakan pilih tanggal awal dan tanggal akhir pada sidebar.")
            st.stop()
            
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

        # ==================================================================
        # BAGIAN KPI TIM X
        # ==================================================================
        st.header("Kontrol KPI Utama Tim X (Periode Intervensi)")
        if uploaded_kpi_file:
            # (Logika KPI tidak berubah, tetap menggunakan file terpisah)
            kpi_data = load_kpi_data(uploaded_kpi_file)
            intervention_date = pd.to_datetime('2025-08-05')
            
            def display_kpi_metrics(branch_name, sales_data, kpi_extra_data):
                st.subheader(f"KPI: {branch_name}")
                before_sales = sales_data[sales_data['Date'] < intervention_date]
                after_sales = sales_data[sales_data['Date'] >= intervention_date]
                before_kpi = kpi_extra_data[kpi_extra_data['Date'] < intervention_date]
                after_kpi = kpi_extra_data[kpi_extra_data['Date'] >= intervention_date]
                avg_sales_before = before_sales['Total After Bill Discount'].sum() / before_sales['Date'].nunique() if not before_sales.empty else 0
                avg_sales_after = after_sales['Total After Bill Discount'].sum() / after_sales['Date'].nunique() if not after_sales.empty else 0
                sales_increase = ((avg_sales_after - avg_sales_before) / avg_sales_before) if avg_sales_before > 0 else 0
                avg_google_score_before = before_kpi['Google_Review_Score'].mean() if not before_kpi.empty else 0
                if branch_name == "Gabungan":
                    latest_scores = after_kpi.sort_values('Date', ascending=False).drop_duplicates(subset=['Branch'], keep='first')
                    latest_google_score_after = latest_scores['Google_Review_Score'].mean() if not latest_scores.empty else 0
                else:
                    latest_google_score_after = after_kpi.sort_values('Date', ascending=False)['Google_Review_Score'].iloc[0] if not after_kpi.empty else 0
                total_complaints_after = after_kpi['Jumlah_Complaints'].sum()
                total_complaints_before = before_kpi['Jumlah_Complaints'].sum()
                latest_qaqc_after = after_kpi.dropna(subset=['Skor_QAQC']).sort_values('Date', ascending=False)['Skor_QAQC'].iloc[0] if not after_kpi.dropna(subset=['Skor_QAQC']).empty else 0
                compliance_after = after_kpi['Upload_Kebersihan'].mean() if not after_kpi.empty else 0
                compliance_before = before_kpi['Upload_Kebersihan'].mean() if not before_kpi.empty else 0
                st.metric(label="Peningkatan Penjualan (Rata-rata Harian)", value=format_rupiah(avg_sales_after), delta=f"{sales_increase:.1%} vs {format_rupiah(avg_sales_before)}")
                st.metric(label="Skor Google Review (Terbaru)", value=f"{latest_google_score_after:.2f} ★", delta=f"dari rata-rata {avg_google_score_before:.2f} ★")
                st.metric(label="Total Keluhan (Setelah Intervensi)", value=f"{total_complaints_after}", delta=f"{total_complaints_after - total_complaints_before} vs periode sebelumnya", delta_color="inverse")
                qaqc_delta_text = "Target: ≥ 90"
                if latest_qaqc_after > 0: qaqc_delta_text = "✅ Lulus Target" if latest_qaqc_after >= 90 else f"❌ Gagal Target ({latest_qaqc_after - 90})"
                st.metric(label="Skor QAQC Terakhir", value=f"{latest_qaqc_after:.0f}", delta=qaqc_delta_text)
                st.metric(label="Kepatuhan Upload Kebersihan", value=f"{compliance_after:.1%}", delta=f"{compliance_after - compliance_before:.1%}")

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1: display_kpi_metrics("Bintara", raw_data[raw_data['Branch'] == 'Bintara'], kpi_data[kpi_data['Branch'] == 'Bintara'])
            with kpi_col2: display_kpi_metrics("Jatiwaringin", raw_data[raw_data['Branch'] == 'Jatiwaringin'], kpi_data[kpi_data['Branch'] == 'Jatiwaringin'])
            with kpi_col3: display_kpi_metrics("Gabungan", raw_data, kpi_data)
        else:
            st.warning("Unggah file `data_kpi_tambahan.xlsx` untuk melihat progres KPI Tim X.")
        st.divider()

        # ==================================================================
        # Proses Filtering Data untuk Bagian Bawah
        # ==================================================================
        filtered_df = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)]
        bintara_df = filtered_df[filtered_df['Branch'] == 'Bintara']
        jatiwaringin_df = filtered_df[filtered_df['Branch'] == 'Jatiwaringin']
        
        # ==================================================================
        # Layout Perbandingan Dua Kolom
        # ==================================================================
        col_bintara, col_jatiwaringin = st.columns(2, gap="large")

        for branch_name, branch_df in [("Bintara", bintara_df), ("Jatiwaringin", jatiwaringin_df)]:
            
            column_to_fill = col_bintara if branch_name == "Bintara" else col_jatiwaringin
            
            with column_to_fill:
                st.header(f"📍 {branch_name}")

                # --- KPI Harian (dihitung on-the-fly) ---
                kpi_harian = branch_df.groupby(branch_df['Date'].dt.date).agg(
                    Total_Penjualan_Rp=('Total After Bill Discount', 'sum'),
                    Jumlah_Transaksi=('Bill Number', 'nunique'),
                    Total_Item_Terjual=('Qty', 'sum')
                ).reset_index()
                kpi_harian['ATV (Nilai Transaksi Rata2)'] = (kpi_harian['Total_Penjualan_Rp'] / kpi_harian['Jumlah_Transaksi']).fillna(0)
                kpi_harian['UPT (Item per Transaksi)'] = (kpi_harian['Total_Item_Terjual'] / kpi_harian['Jumlah_Transaksi']).fillna(0)

                # --- Analisis per Jam (dihitung on-the-fly) ---
                branch_df['Hour'] = branch_df['Order Time'].dt.hour
                analisis_jam = branch_df.groupby('Hour').agg(
                    Jumlah_Pengunjung_Transaksi=('Bill Number', 'nunique')
                ).reset_index()
                
                # --- Lanjutan Tampilan ---
                # (Kode untuk menampilkan metrik, tren, dan grafik lainnya tetap sama,
                #  tapi sekarang sumbernya adalah data mentah yang sudah difilter 'branch_df')
                
                # ... (Kode untuk metrik, tren, analisis jam, menu, payment, waiter)
                # ... (Ini akan menjadi panjang, jadi saya singkat. Logikanya adalah
                # ... melakukan groupby pada 'branch_df' untuk setiap grafik)
                st.subheader("Tren Kinerja Harian")
                fig_tren = px.line(kpi_harian, x='Date', y=['Total_Penjualan_Rp', 'Jumlah_Transaksi', 'ATV (Nilai Transaksi Rata2)', 'UPT (Item per Transaksi)'],
                                   facet_row="variable", labels={"variable": "Metrik", "value": "Nilai", "Date": "Tanggal"},
                                   height=600, markers=True)
                fig_tren.update_yaxes(matches=None, title_text="")
                st.plotly_chart(fig_tren, use_container_width=True)

                st.subheader("Analisis Aktivitas per Jam")
                fig_jam = px.bar(analisis_jam, x='Hour', y='Jumlah_Pengunjung_Transaksi', labels={'Hour': 'Jam', 'Jumlah_Pengunjung_Transaksi': 'Jumlah Transaksi'})
                st.plotly_chart(fig_jam, use_container_width=True)

else:
    st.info("Silakan unggah file data transaksi mentah untuk memulai analisis.")
