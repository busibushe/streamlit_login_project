import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ==============================================================================
# Konfigurasi Halaman Dashboard
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Monitoring Tim X",
    page_icon="🎯",
    layout="wide"
)

# ==============================================================================
# Fungsi Helper
# ==============================================================================

@st.cache_data # Cache data agar tidak perlu load ulang setiap kali filter diubah
def load_data(uploaded_file):
    """Membaca semua sheet dari file Excel yang diunggah."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        data = {
            'kpi': pd.read_excel(xls, 'KPI Harian'),
            'jam': pd.read_excel(xls, 'Analisis per Jam'),
            'menu': pd.read_excel(xls, 'Rekap Kategori & Menu'),
            'stok': pd.read_excel(xls, 'Laporan Stok Harian')
        }
        
        # --- Pre-processing Data ---
        # Ubah kolom tanggal menjadi tipe datetime
        data['kpi']['Date'] = pd.to_datetime(data['kpi']['Date'])
        data['stok']['Date'] = pd.to_datetime(data['stok']['Date'])
        
        return data
    except Exception as e:
        st.error(f"Error: Gagal membaca file Excel. Pastikan sheet berikut ada: 'KPI Harian', 'Analisis per Jam', 'Rekap Kategori & Menu', 'Laporan Stok Harian'. Detail: {e}")
        return None

# ==============================================================================
# Tampilan Utama Dashboard
# ==============================================================================

st.title("🎯 Dashboard Monitoring Tim X")
st.markdown("Fokus Analisis: **Cabang Bintara & Jatiwaringin**")

# --- Bagian Upload File ---
uploaded_file = st.file_uploader(
    "Unggah file laporan Excel terbaru (`laporan_analisis_toko...xlsx`)",
    type=['xlsx']
)

# --- Pesan untuk Integrasi Google Sheets di Masa Depan ---
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
        # Sidebar untuk Filter
        # ======================================================================
        st.sidebar.header("⚙️ Filter Data")
        
        # Filter Tanggal
        min_date = data['kpi']['Date'].min().date()
        max_date = data['kpi']['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal",
            value=(max_date - timedelta(days=6), max_date), # Default 7 hari terakhir
            min_value=min_date,
            max_value=max_date,
            help="Pilih periode analisis yang diinginkan."
        )

        # Konversi kembali ke datetime untuk filtering
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        # Filter Cabang
        selected_branches = st.sidebar.multiselect(
            "Pilih Cabang",
            options=["Bintara", "Jatiwaringin"],
            default=["Bintara", "Jatiwaringin"],
            help="Pilih satu atau lebih cabang untuk ditampilkan."
        )

        if not selected_branches:
            st.warning("Pilih minimal satu cabang untuk menampilkan data.")
        else:
            # ==================================================================
            # Proses Filtering Data
            # ==================================================================
            kpi_filtered = data['kpi'][
                (data['kpi']['Date'] >= start_date) & 
                (data['kpi']['Date'] <= end_date) &
                (data['kpi']['Branch'].isin(selected_branches))
            ]
            
            jam_filtered = data['jam'][data['jam']['Branch'].isin(selected_branches)]
            menu_filtered = data['menu'][data['menu']['Branch'].isin(selected_branches)]
            
            stok_filtered = data['stok'][
                (data['stok']['Date'] >= start_date) & 
                (data['stok']['Date'] <= end_date) &
                (data['stok']['Branch'].isin(selected_branches))
            ]

            # ==================================================================
            # Tampilan Metrik Utama (KPI)
            # ==================================================================
            st.header("Ringkasan Kinerja Utama")
            
            total_penjualan = kpi_filtered['Total_Penjualan_Rp'].sum()
            total_transaksi = kpi_filtered['Jumlah_Transaksi'].sum()
            avg_atv = total_penjualan / total_transaksi if total_transaksi > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Penjualan", f"Rp {total_penjualan:,.0f}")
            col2.metric("Jumlah Transaksi", f"{total_transaksi:,}")
            col3.metric("Nilai Transaksi Rata-rata (ATV)", f"Rp {avg_atv:,.0f}")

            st.divider()

            # ==================================================================
            # Visualisasi Data
            # ==================================================================
            st.header("Visualisasi Data")
            
            col_a, col_b = st.columns(2)

            with col_a:
                # --- Grafik 1: Tren Penjualan Harian ---
                st.subheader("Tren Penjualan Harian")
                fig_tren = px.line(
                    kpi_filtered, 
                    x='Date', 
                    y='Total_Penjualan_Rp', 
                    color='Branch',
                    title="Pergerakan Penjualan per Hari",
                    labels={'Total_Penjualan_Rp': 'Total Penjualan (Rp)', 'Date': 'Tanggal'},
                    markers=True
                )
                st.plotly_chart(fig_tren, use_container_width=True)

            with col_b:
                # --- Grafik 2: Komposisi Kategori Menu ---
                st.subheader("Komposisi Penjualan per Kategori")
                menu_agg = menu_filtered.groupby('Menu Category Detail')['Total_Penjualan_Rp'].sum().reset_index()
                fig_pie = px.pie(
                    menu_agg, 
                    names='Menu Category Detail', 
                    values='Total_Penjualan_Rp',
                    title="Kontribusi Penjualan Berdasarkan Kategori Menu",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- Grafik 3: Analisis per Jam ---
            st.subheader("Analisis Trafik & Penjualan per Jam")
            fig_jam = px.bar(
                jam_filtered, 
                x='Hour', 
                y='Jumlah_Pengunjung_Transaksi',
                color='Branch',
                barmode='group',
                title="Jumlah Transaksi per Jam",
                labels={'Hour': 'Jam', 'Jumlah_Pengunjung_Transaksi': 'Jumlah Transaksi'}
            )
            st.plotly_chart(fig_jam, use_container_width=True)

            # --- Grafik 4: Laporan Stok Harian ---
            st.subheader("Monitoring Stok Harian")
            fig_stok = px.line(
                stok_filtered,
                x='Date',
                y='Stock',
                color='Product Name',
                facet_row='Branch', # Pisahkan grafik per cabang
                title="Pergerakan Stok Produk Utama",
                labels={'Stock': 'Jumlah Stok', 'Date': 'Tanggal'},
                height=500
            )
            st.plotly_chart(fig_stok, use_container_width=True)

            st.divider()
            
            # ==================================================================
            # Tampilan Data Detail (dalam Expander)
            # ==================================================================
            st.header("Data Detail")
            
            with st.expander("Lihat Data KPI Harian"):
                st.dataframe(kpi_filtered)
            
            with st.expander("Lihat Data Rekap Kategori & Menu"):
                st.dataframe(menu_filtered)
            
            with st.expander("Lihat Data Laporan Stok"):
                st.dataframe(stok_filtered)
else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")
