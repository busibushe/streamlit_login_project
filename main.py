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
    """Membaca semua sheet dari file Excel yang diunggah."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        # Membaca semua sheet yang dibutuhkan
        sheet_names = ['KPI Harian', 'Analisis per Jam', 'Waktu Pelayanan per Jam', 'Rekap Kategori & Menu', 'Rekap Metode Pembayaran', 'Analisis Kinerja Waiter', 'Laporan Stok Harian']
        data = {name.split(' ')[0].lower(): pd.read_excel(xls, name) for name in sheet_names}
        
        # --- Pre-processing Data ---
        data['kpi']['Date'] = pd.to_datetime(data['kpi']['Date'])
        data['laporan']['Date'] = pd.to_datetime(data['laporan']['Date'])
        
        return data
    except Exception as e:
        st.error(f"Error: Gagal membaca file Excel. Pastikan semua sheet yang dibutuhkan ada. Detail: {e}")
        return None

def format_rupiah(value):
    """Mengubah angka menjadi format Rupiah."""
    return f"Rp {value:,.0f}".replace(',', '.')

# ==============================================================================
# Tampilan Utama Dashboard
# ==============================================================================

st.title("📊 Dashboard Perbandingan Kinerja: Bintara vs Jatiwaringin")

# --- Bagian Upload File ---
uploaded_file = st.file_uploader(
    "Unggah file laporan Excel terbaru (`laporan_analisis_toko...xlsx`)",
    type=['xlsx']
)

st.info(
    """
    **Info:** Versi ini menggunakan upload file manual. Versi selanjutnya akan terhubung langsung ke Google Sheets 
    untuk pembaruan data otomatis setiap hari.
    """,
    icon="💡"
)

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data:
        # ======================================================================
        # Sidebar untuk Filter Tanggal
        # ======================================================================
        st.sidebar.header("⚙️ Filter Data")
        min_date = data['kpi']['Date'].min().date()
        max_date = data['kpi']['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal",
            value=(max_date - timedelta(days=6), max_date),
            min_value=min_date,
            max_value=max_date
        )
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

        # ==================================================================
        # Proses Filtering Data
        # ==================================================================
        kpi_filtered = data['kpi'][(data['kpi']['Date'] >= start_date) & (data['kpi']['Date'] <= end_date)]
        stok_filtered = data['laporan'][(data['laporan']['Date'] >= start_date) & (data['laporan']['Date'] <= end_date)]
        
        # Pisahkan data per cabang
        bintara_data = {key: df[df['Branch'] == 'Bintara'] for key, df in data.items()}
        jatiwaringin_data = {key: df[df['Branch'] == 'Jatiwaringin'] for key, df in data.items()}
        
        kpi_bintara = kpi_filtered[kpi_filtered['Branch'] == 'Bintara']
        kpi_jatiwaringin = kpi_filtered[kpi_filtered['Branch'] == 'Jatiwaringin']

        # ==================================================================
        # Tampilan Metrik Utama (Summary Tengah)
        # ==================================================================
        st.header(f"Ringkasan Kinerja Gabungan ({start_date.strftime('%d %b')} - {end_date.strftime('%d %b')})")
        
        total_penjualan = kpi_filtered['Total_Penjualan_Rp'].sum()
        total_transaksi = kpi_filtered['Jumlah_Transaksi'].sum()
        avg_atv = total_penjualan / total_transaksi if total_transaksi > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Penjualan Gabungan", format_rupiah(total_penjualan))
        col2.metric("Total Transaksi Gabungan", f"{total_transaksi:,}")
        col3.metric("Rata-rata ATV Gabungan", format_rupiah(avg_atv))
        st.divider()

        # ==================================================================
        # Layout Perbandingan Dua Kolom
        # ==================================================================
        col_bintara, col_jatiwaringin = st.columns(2, gap="large")

        for branch_name, branch_kpi, branch_all_data in [("Bintara", kpi_bintara, bintara_data), ("Jatiwaringin", kpi_jatiwaringin, jatiwaringin_data)]:
            
            # Tentukan kolom mana yang akan diisi
            column_to_fill = col_bintara if branch_name == "Bintara" else col_jatiwaringin
            
            with column_to_fill:
                st.header(f"📍 {branch_name}")

                # --- KPI Individu ---
                branch_total_penjualan = branch_kpi['Total_Penjualan_Rp'].sum()
                branch_total_transaksi = branch_kpi['Jumlah_Transaksi'].sum()
                branch_avg_atv = branch_total_penjualan / branch_total_transaksi if branch_total_transaksi > 0 else 0
                
                c1, c2 = st.columns(2)
                c1.metric("Total Penjualan", format_rupiah(branch_total_penjualan))
                c2.metric("Total Transaksi", f"{branch_total_transaksi:,}")

                # --- Monitoring Stok Terbaru ---
                st.subheader("Stok Produk Utama (Terbaru)")
                latest_stok = stok_filtered[stok_filtered['Branch'] == branch_name].sort_values('Date').drop_duplicates('Product Name', keep='last')
                if not latest_stok.empty:
                    st.dataframe(latest_stok[['Product Name', 'Stock', 'Capacity', 'Percentage']], use_container_width=True)
                else:
                    st.warning("Data stok tidak tersedia pada rentang tanggal ini.")

                # --- Tren Harian ---
                st.subheader("Tren Kinerja Harian")
                fig_tren = px.line(branch_kpi, x='Date', y=['Total_Penjualan_Rp', 'Jumlah_Transaksi', 'ATV (Nilai Transaksi Rata2)', 'UPT (Item per Transaksi)'],
                                   facet_row="variable", labels={"variable": "Metrik", "value": "Nilai", "Date": "Tanggal"},
                                   height=600, markers=True)
                fig_tren.update_yaxes(matches=None) # Agar skala Y tiap grafik independen
                st.plotly_chart(fig_tren, use_container_width=True)

                # --- Analisis per Jam ---
                st.subheader("Analisis Aktivitas per Jam")
                hourly_perf = pd.merge(branch_all_data['jam'], branch_all_data['waktu'], on='Hour', how='outer').sort_values('Hour')
                fig_jam = px.bar(hourly_perf, x='Hour', y='Jumlah_Pengunjung_Transaksi', labels={'Hour': 'Jam', 'Jumlah_Pengunjung_Transaksi': 'Jumlah Transaksi'})
                fig_jam.add_scatter(x=hourly_perf['Hour'], y=hourly_perf['Rata2_Waktu_Pelayanan_Menit'], mode='lines', name='Waktu Layanan (Menit)', yaxis='y2')
                fig_jam.update_layout(yaxis2=dict(title='Waktu Layanan (Menit)', overlaying='y', side='right'))
                st.plotly_chart(fig_jam, use_container_width=True)

                # --- Top & Bottom 5 Menu ---
                st.subheader("Performa Menu Teratas & Terbawah")
                menu_perf = branch_all_data['menu'].sort_values('Total_Penjualan_Rp', ascending=False)
                top5 = menu_perf.head(5)
                bottom5 = menu_perf.tail(5)
                
                fig_top5 = px.bar(top5, y='Menu', x='Total_Penjualan_Rp', orientation='h', title="Top 5 Menu Terlaris")
                st.plotly_chart(fig_top5, use_container_width=True)
                
                fig_bottom5 = px.bar(bottom5, y='Menu', x='Total_Penjualan_Rp', orientation='h', title="Bottom 5 Menu Kurang Laris")
                st.plotly_chart(fig_bottom5, use_container_width=True)

                # --- Rekap Metode Pembayaran ---
                st.subheader("Metode Pembayaran")
                fig_payment = px.pie(branch_all_data['rekap'], names='Payment Method', values='Jumlah_Transaksi', hole=0.4, title="Distribusi Metode Pembayaran")
                st.plotly_chart(fig_payment, use_container_width=True)
                
                # --- Kinerja Waiter ---
                st.subheader("Kinerja Waiter")
                fig_waiter = px.bar(branch_all_data['analisis'], x='Waiter', y='Total_Penjualan_Diambil', title="Total Penjualan per Waiter")
                st.plotly_chart(fig_waiter, use_container_width=True)

else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")
