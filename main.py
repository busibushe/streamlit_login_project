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
def load_data(uploaded_file):
    """Membaca semua sheet dari file Excel yang diunggah dengan key yang unik."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_map = {
            'KPI Harian': 'kpi_harian',
            'Analisis per Jam': 'analisis_jam',
            'Waktu Pelayanan per Jam': 'waktu_pelayanan',
            'Rekap Kategori & Menu': 'rekap_menu',
            'Rekap Metode Pembayaran': 'rekap_payment',
            'Analisis Kinerja Waiter': 'kinerja_waiter',
            'Laporan Stok Harian': 'laporan_stok'
        }
        
        data = {}
        for sheet_name, key in sheet_map.items():
            data[key] = pd.read_excel(xls, sheet_name)
        
        # --- Pre-processing Data ---
        data['kpi_harian']['Date'] = pd.to_datetime(data['kpi_harian']['Date'])
        data['laporan_stok']['Date'] = pd.to_datetime(data['laporan_stok']['Date'])
        
        return data
    except Exception as e:
        st.error(f"Error: Gagal membaca file Excel. Pastikan semua sheet yang dibutuhkan ada. Detail: {e}")
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
    uploaded_file = st.file_uploader(
        "1. Unggah Laporan Analisis Toko (`laporan_analisis_toko...xlsx`)",
        type=['xlsx']
    )
with col_upload2:
    uploaded_kpi_file = st.file_uploader(
        "2. Unggah Data KPI Tambahan (`data_kpi_tambahan.xlsx`)",
        type=['xlsx']
    )

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data:
        # ======================================================================
        # Sidebar untuk Filter Tanggal
        # ======================================================================
        st.sidebar.header("⚙️ Filter Data")
        min_date = data['kpi_harian']['Date'].min().date()
        max_date = data['kpi_harian']['Date'].max().date()
        
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
        # BAGIAN BARU: Monitoring KPI Tim X
        # ==================================================================
        st.header("Kontrol KPI Utama Tim X (Periode Intervensi)")
        if uploaded_kpi_file:
            kpi_data = load_kpi_data(uploaded_kpi_file)
            intervention_date = pd.to_datetime('2025-08-05')

            # --- Fungsi untuk menghitung dan menampilkan KPI ---
            def display_kpi_metrics(branch_name, sales_data, kpi_extra_data):
                st.subheader(f"KPI: {branch_name}")

                # Filter data sebelum dan sesudah intervensi
                before_sales = sales_data[sales_data['Date'] < intervention_date]
                after_sales = sales_data[sales_data['Date'] >= intervention_date]
                before_kpi = kpi_extra_data[kpi_extra_data['Date'] < intervention_date]
                after_kpi = kpi_extra_data[kpi_extra_data['Date'] >= intervention_date]

                # --- Hitung Metrik KPI ---
                avg_sales_before = before_sales['Total_Penjualan_Rp'].mean() if not before_sales.empty else 0
                avg_sales_after = after_sales['Total_Penjualan_Rp'].mean() if not after_sales.empty else 0
                sales_increase = ((avg_sales_after - avg_sales_before) / avg_sales_before) if avg_sales_before > 0 else 0
                
                avg_google_score_before = before_kpi['Google_Review_Score'].mean() if not before_kpi.empty else 0
                
                # ✅ PERBAIKAN: Logika perhitungan skor Google Review untuk gabungan
                if branch_name == "Gabungan":
                    # Cari skor terbaru untuk setiap cabang, lalu hitung rata-ratanya
                    latest_scores = after_kpi.sort_values('Date', ascending=False).drop_duplicates(subset=['Branch'], keep='first')
                    latest_google_score_after = latest_scores['Google_Review_Score'].mean() if not latest_scores.empty else 0
                else:
                    # Logika asli untuk cabang individu (sudah benar)
                    latest_google_score_after = after_kpi.sort_values('Date', ascending=False)['Google_Review_Score'].iloc[0] if not after_kpi.empty else 0

                total_complaints_after = after_kpi['Jumlah_Complaints'].sum()
                total_complaints_before = before_kpi['Jumlah_Complaints'].sum()
                
                latest_qaqc_after = after_kpi.dropna(subset=['Skor_QAQC']).sort_values('Date', ascending=False)['Skor_QAQC'].iloc[0] if not after_kpi.dropna(subset=['Skor_QAQC']).empty else 0
                
                compliance_after = after_kpi['Upload_Kebersihan'].mean() if not after_kpi.empty else 0
                compliance_before = before_kpi['Upload_Kebersihan'].mean() if not before_kpi.empty else 0
                
                # --- Tampilkan Metrik KPI ---
                st.metric(
                    label="Peningkatan Penjualan (Rata-rata Harian)",
                    value=format_rupiah(avg_sales_after),
                    delta=f"{sales_increase:.1%} vs {format_rupiah(avg_sales_before)}"
                )
                st.metric(
                    label="Skor Google Review (Terbaru)",
                    value=f"{latest_google_score_after:.2f} ★",
                    delta=f"dari rata-rata {avg_google_score_before:.2f} ★"
                )
                st.metric(
                    label="Total Keluhan (Setelah Intervensi)",
                    value=f"{total_complaints_after}",
                    delta=f"{total_complaints_after - total_complaints_before} vs periode sebelumnya",
                    delta_color="inverse"
                )
                
                # Logika warna untuk Skor QAQC
                qaqc_delta_text = "Target: ≥ 90"
                if latest_qaqc_after > 0:
                    qaqc_delta_text = "✅ Lulus Target" if latest_qaqc_after >= 90 else f"❌ Gagal Target ({latest_qaqc_after - 90})"
                
                st.metric(
                    label="Skor QAQC Terakhir",
                    value=f"{latest_qaqc_after:.0f}",
                    delta=qaqc_delta_text
                )
                st.metric(
                    label="Kepatuhan Upload Kebersihan",
                    value=f"{compliance_after:.1%}",
                    delta=f"{compliance_after - compliance_before:.1%}"
                )

            # --- Tampilkan KPI per Kolom ---
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                display_kpi_metrics("Bintara", data['kpi_harian'][data['kpi_harian']['Branch'] == 'Bintara'], kpi_data[kpi_data['Branch'] == 'Bintara'])
            with kpi_col2:
                display_kpi_metrics("Jatiwaringin", data['kpi_harian'][data['kpi_harian']['Branch'] == 'Jatiwaringin'], kpi_data[kpi_data['Branch'] == 'Jatiwaringin'])
            with kpi_col3:
                display_kpi_metrics("Gabungan", data['kpi_harian'], kpi_data)

        else:
            st.warning("Unggah file `data_kpi_tambahan.xlsx` untuk melihat progres KPI Tim X.")
        st.divider()

        # ==================================================================
        # Proses Filtering Data untuk Bagian Bawah
        # ==================================================================
        filtered_data = {}
        for key, df in data.items():
            if 'Date' in df.columns:
                filtered_data[key] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            else:
                filtered_data[key] = df

        bintara_data = {key: df[df['Branch'] == 'Bintara'] for key, df in filtered_data.items()}
        jatiwaringin_data = {key: df[df['Branch'] == 'Jatiwaringin'] for key, df in filtered_data.items()}
        
        # ==================================================================
        # Layout Perbandingan Dua Kolom
        # ==================================================================
        col_bintara, col_jatiwaringin = st.columns(2, gap="large")

        for branch_name, branch_data in [("Bintara", bintara_data), ("Jatiwaringin", jatiwaringin_data)]:
            
            column_to_fill = col_bintara if branch_name == "Bintara" else col_jatiwaringin
            
            with column_to_fill:
                st.header(f"📍 {branch_name}")

                # --- KPI Individu ---
                branch_kpi = branch_data['kpi_harian']
                branch_total_penjualan = branch_kpi['Total_Penjualan_Rp'].sum()
                branch_total_transaksi = branch_kpi['Jumlah_Transaksi'].sum()
                
                latest_kpi_day = branch_kpi.sort_values('Date', ascending=False).iloc[0] if not branch_kpi.empty else None
                latest_atv = latest_kpi_day['ATV (Nilai Transaksi Rata2)'] if latest_kpi_day is not None else 0
                latest_upt = latest_kpi_day['UPT (Item per Transaksi)'] if latest_kpi_day is not None else 0
                
                row1_cols = st.columns(2)
                row1_cols[0].metric("Total Penjualan", format_rupiah(branch_total_penjualan))
                row1_cols[1].metric("Total Transaksi", f"{branch_total_transaksi:,}")
                
                row2_cols = st.columns(2)
                row2_cols[0].metric("ATV (Hari Terakhir)", format_rupiah(latest_atv))
                row2_cols[1].metric("UPT (Hari Terakhir)", f"{latest_upt:.2f}")

                # --- Monitoring Stok Terbaru ---
                st.subheader("Stok Produk Utama (Terbaru)")
                latest_stok = branch_data['laporan_stok'].sort_values('Date').drop_duplicates('Product Name', keep='last')
                if not latest_stok.empty:
                    stock_cols = st.columns(len(latest_stok))
                    for i, (_, row) in enumerate(latest_stok.iterrows()):
                        with stock_cols[i]:
                            st.metric(
                                label=f"{row['Product Name']} ({row['Capacity']})",
                                value=f"{row['Stock']}",
                                delta=f"{row['Percentage']:.0%} terisi",
                                delta_color="off" 
                            )
                else:
                    st.warning("Data stok tidak tersedia pada rentang tanggal ini.")

                # --- Tren Harian ---
                st.subheader("Tren Kinerja Harian")
                fig_tren = px.line(branch_kpi, x='Date', y=['Total_Penjualan_Rp', 'Jumlah_Transaksi', 'ATV (Nilai Transaksi Rata2)', 'UPT (Item per Transaksi)'],
                                   facet_row="variable", labels={"variable": "Metrik", "value": "Nilai", "Date": "Tanggal"},
                                   height=600, markers=True)
                fig_tren.update_yaxes(matches=None, title_text="")
                st.plotly_chart(fig_tren, use_container_width=True)

                # --- Analisis per Jam ---
                st.subheader("Analisis Aktivitas per Jam")
                st.caption("Catatan: Data ini merupakan rangkuman dari keseluruhan periode file yang diunggah.")
                hourly_perf = pd.merge(branch_data['analisis_jam'], branch_data['waktu_pelayanan'], on=['Branch', 'Hour'], how='outer').sort_values('Hour')
                fig_jam = px.bar(hourly_perf, x='Hour', y='Jumlah_Pengunjung_Transaksi', labels={'Hour': 'Jam', 'Jumlah_Pengunjung_Transaksi': 'Jumlah Transaksi'})
                fig_jam.add_scatter(x=hourly_perf['Hour'], y=hourly_perf['Rata2_Waktu_Pelayanan_Menit'], mode='lines', name='Waktu Layanan (Menit)', yaxis='y2')
                fig_jam.update_layout(yaxis2=dict(title='Waktu Layanan (Menit)', overlaying='y', side='right', showgrid=False))
                st.plotly_chart(fig_jam, use_container_width=True)

                # --- Analisis Performa Menu & Kategori ---
                st.subheader("Analisis Performa Menu & Kategori")
                st.caption("Catatan: Data ini merupakan rangkuman dari keseluruhan periode file yang diunggah.")

                sort_options = {
                    'Berdasarkan Penjualan (Rp)': 'Total_Penjualan_Rp',
                    'Berdasarkan Kuantitas (Qty)': 'Total_Item_Terjual'
                }
                sort_choice = st.radio(
                    "Tampilkan data berdasarkan:",
                    options=sort_options.keys(),
                    key=f'sort_{branch_name}',
                    horizontal=True
                )
                sort_by_column = sort_options[sort_choice]
                
                if sort_choice == 'Berdasarkan Penjualan (Rp)':
                    sales_by_cat = branch_data['rekap_menu'].groupby('Menu Category Detail')['Total_Penjualan_Rp'].sum().reset_index()
                    fig_donut = px.pie(sales_by_cat, names='Menu Category Detail', values='Total_Penjualan_Rp', hole=0.5, title="Distribusi Penjualan (Rp)")
                else:
                    qty_by_cat = branch_data['rekap_menu'].groupby('Menu Category Detail')['Total_Item_Terjual'].sum().reset_index()
                    fig_donut = px.pie(qty_by_cat, names='Menu Category Detail', values='Total_Item_Terjual', hole=0.5, title="Distribusi Kuantitas (Qty)")
                st.plotly_chart(fig_donut, use_container_width=True)
                
                menu_perf = branch_data['rekap_menu'].sort_values(sort_by_column, ascending=False)
                top5 = menu_perf.head(5)
                bottom5 = menu_perf.tail(5).sort_values(sort_by_column, ascending=True)
                
                fig_top5 = px.bar(top5.iloc[::-1], y='Menu', x=sort_by_column, orientation='h', title="Top 5 Menu")
                st.plotly_chart(fig_top5, use_container_width=True)
                
                fig_bottom5 = px.bar(bottom5, y='Menu', x=sort_by_column, orientation='h', title="Bottom 5 Menu")
                st.plotly_chart(fig_bottom5, use_container_width=True)

                # --- Rekap Metode Pembayaran ---
                st.subheader("Metode Pembayaran")
                st.caption("Catatan: Data ini merupakan rangkuman dari keseluruhan periode file yang diunggah.")
                fig_payment = px.pie(branch_data['rekap_payment'], names='Payment Method', values='Jumlah_Transaksi', hole=0.4, title="Distribusi Metode Pembayaran")
                st.plotly_chart(fig_payment, use_container_width=True)
                
                # --- Kinerja Waiter ---
                st.subheader("Kinerja Waiter")
                st.caption("Catatan: Data ini merupakan rangkuman dari keseluruhan periode file yang diunggah.")
                fig_waiter = px.bar(branch_data['kinerja_waiter'], x='Waiter', y='ATV_per_Waiter', title="Rata-rata Nilai Transaksi (ATV) per Waiter",
                                    labels={'ATV_per_Waiter': 'ATV (Rp)'})
                st.plotly_chart(fig_waiter, use_container_width=True)

else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")
