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
        min_date = data['kpi_harian']['Date'].min().date()
        max_date = data['kpi_harian']['Date'].max().date()
        
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
        filtered_data = {}
        for key, df in data.items():
            if 'Date' in df.columns:
                filtered_data[key] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            else:
                filtered_data[key] = df

        bintara_data = {key: df[df['Branch'] == 'Bintara'] for key, df in filtered_data.items()}
        jatiwaringin_data = {key: df[df['Branch'] == 'Jatiwaringin'] for key, df in filtered_data.items()}
        
        # ==================================================================
        # Tampilan Metrik Utama (Summary Tengah)
        # ==================================================================
        st.header(f"Ringkasan Kinerja Gabungan ({start_date.strftime('%d %b')} - {end_date.strftime('%d %b')})")
        
        kpi_gabungan = filtered_data['kpi_harian']
        total_penjualan = kpi_gabungan['Total_Penjualan_Rp'].sum()
        total_transaksi = kpi_gabungan['Jumlah_Transaksi'].sum()
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
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Penjualan", format_rupiah(branch_total_penjualan))
                c2.metric("Total Transaksi", f"{branch_total_transaksi:,}")
                c3.metric("ATV (Hari Terakhir)", format_rupiah(latest_atv))
                c4.metric("UPT (Hari Terakhir)", f"{latest_upt:.2f}")

                # --- Monitoring Stok Terbaru ---
                st.subheader("Stok Produk Utama (Terbaru)")
                latest_stok = branch_data['laporan_stok'].sort_values('Date').drop_duplicates('Product Name', keep='last')
                if not latest_stok.empty:
                    stock_cols = st.columns(len(latest_stok))
                    for i, (_, row) in enumerate(latest_stok.iterrows()):
                        with stock_cols[i]:
                            st.metric(
                                label=f"{row['Product Name']} (Kapasitas: {row['Capacity']})",
                                value=f"{row['Stock']}",
                                delta=f"{row['Percentage']:.0%}",
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
                hourly_perf = pd.merge(branch_data['analisis_jam'], branch_data['waktu_pelayanan'], on=['Branch', 'Hour'], how='outer').sort_values('Hour')
                fig_jam = px.bar(hourly_perf, x='Hour', y='Jumlah_Pengunjung_Transaksi', labels={'Hour': 'Jam', 'Jumlah_Pengunjung_Transaksi': 'Jumlah Transaksi'})
                fig_jam.add_scatter(x=hourly_perf['Hour'], y=hourly_perf['Rata2_Waktu_Pelayanan_Menit'], mode='lines', name='Waktu Layanan (Menit)', yaxis='y2')
                fig_jam.update_layout(yaxis2=dict(title='Waktu Layanan (Menit)', overlaying='y', side='right', showgrid=False))
                st.plotly_chart(fig_jam, use_container_width=True)

                # --- Top & Bottom 5 Menu ---
                st.subheader("Performa Menu Teratas & Terbawah")
                sort_by = st.radio(
                    "Urutkan berdasarkan:",
                    ('Total Penjualan (Rp)', 'Total Item Terjual'),
                    key=f'sort_{branch_name}',
                    horizontal=True
                )
                
                menu_perf = branch_data['rekap_menu'].sort_values(sort_by, ascending=False)
                top5 = menu_perf.head(5)
                bottom5 = menu_perf.tail(5).sort_values(sort_by, ascending=True)
                
                fig_top5 = px.bar(top5, y='Menu', x=sort_by, orientation='h', title="Top 5 Menu")
                st.plotly_chart(fig_top5, use_container_width=True)
                
                fig_bottom5 = px.bar(bottom5, y='Menu', x=sort_by, orientation='h', title="Bottom 5 Menu")
                st.plotly_chart(fig_bottom5, use_container_width=True)

                # --- Distribusi Kategori Menu ---
                st.subheader("Distribusi Kategori Menu")
                donut1, donut2 = st.columns(2)
                with donut1:
                    sales_by_cat = branch_data['rekap_menu'].groupby('Menu Category Detail')['Total_Penjualan_Rp'].sum().reset_index()
                    fig_donut_sales = px.pie(sales_by_cat, names='Menu Category Detail', values='Total_Penjualan_Rp', hole=0.5, title="Berdasarkan Penjualan (Rp)")
                    st.plotly_chart(fig_donut_sales, use_container_width=True)
                with donut2:
                    qty_by_cat = branch_data['rekap_menu'].groupby('Menu Category Detail')['Total_Item_Terjual'].sum().reset_index()
                    fig_donut_qty = px.pie(qty_by_cat, names='Menu Category Detail', values='Total_Item_Terjual', hole=0.5, title="Berdasarkan Kuantitas (Qty)")
                    st.plotly_chart(fig_donut_qty, use_container_width=True)

                # --- Rekap Metode Pembayaran ---
                st.subheader("Metode Pembayaran")
                fig_payment = px.pie(branch_data['rekap_payment'], names='Payment Method', values='Jumlah_Transaksi', hole=0.4, title="Distribusi Metode Pembayaran")
                st.plotly_chart(fig_payment, use_container_width=True)
                
                # --- Kinerja Waiter ---
                st.subheader("Kinerja Waiter")
                fig_waiter = px.bar(branch_data['kinerja_waiter'], x='Waiter', y='ATV_per_Waiter', title="Rata-rata Nilai Transaksi (ATV) per Waiter",
                                    labels={'ATV_per_Waiter': 'ATV (Rp)'})
                st.plotly_chart(fig_waiter, use_container_width=True)

else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")
