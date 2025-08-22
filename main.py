import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase.client import Client, create_client

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Sales Summary Dashboard",
    page_icon="🧀",
    layout="wide"
)

# ==============================================================================
# KONEKSI & PEMUATAN DATA DARI SUPABASE
# ==============================================================================

# Inisialisasi koneksi ke Supabase
@st.cache_resource
def init_supabase_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase_connection()

# Memuat data dari tabel 'sales_data' di Supabase
@st.cache_data(ttl=600) # Cache data selama 10 menit
def load_sales_data():
    try:
        # Ambil semua data, atasi limit 1000 baris
        response = supabase.table("sales_data").select("*").range(0, 2000000).execute()
        df = pd.DataFrame(response.data)

        # Konversi tipe data & penyesuaian kolom
        if df.empty:
            return df
        
        # Kolom tanggal
        date_cols = ['Date', 'Sales Date In', 'Sales Date Out', 'Order Time']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Kolom angka (pastikan tidak ada error)
        numeric_cols = ['Qty', 'Price', 'Subtotal', 'Discount', 'Service Charge', 'Tax', 'VAT', 'Total', 'Nett Sales', 'Bill Discount', 'Total After Bill Discount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ganti nama kolom untuk kemudahan
        if 'Total After Bill Discount' in df.columns:
            df.rename(columns={'Total After Bill Discount': 'Sales'}, inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari Supabase: {e}")
        return pd.DataFrame()

# ==============================================================================
# APLIKASI UTAMA STREAMLIT
# ==============================================================================

# Muat data utama
df = load_sales_data()

if df.empty:
    st.warning("Tidak ada data penjualan yang dapat dimuat dari database.")
    st.stop()

# --- SIDEBAR & FILTER ---
st.sidebar.image("https://cheesecuit.com/wp-content/uploads/2023/12/Logo-Cheesecuit-Brown-300x156.png", width=150)
st.sidebar.header("Filter Dasbor")

# Filter Waktu (Harian, Mingguan, Bulanan)
# Untuk saat ini, kita fokus pada filter harian seperti di gambar
# time_view = st.sidebar.radio("Pilih Waktu", ('Harian', 'Mingguan', 'Bulanan'))

# Filter Tanggal
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter lain dengan opsi 'Semua'
all_option = "Semua"
branch_options = [all_option] + sorted(df['Branch'].unique().tolist())
selected_branch = st.sidebar.selectbox("Branch", branch_options)

category_options = [all_option] + sorted(df['Menu Category Detail'].unique().tolist())
selected_category = st.sidebar.selectbox("Menu Category Detail", category_options)

area_options = [all_option] + sorted(df['Area'].unique().tolist())
selected_area = st.sidebar.selectbox("Area", area_options)

# --- FILTERING DATA LOGIC ---
df_filtered = df[
    (df['Date'].dt.date >= start_date) & 
    (df['Date'].dt.date <= end_date)
]

if selected_branch != all_option:
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_category != all_option:
    df_filtered = df_filtered[df_filtered['Menu Category Detail'] == selected_category]
if selected_area != all_option:
    df_filtered = df_filtered[df_filtered['Area'] == selected_area]

if df_filtered.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
    st.stop()

# --- PERHITUNGAN KPI UTAMA ---
total_sales = df_filtered['Sales'].sum()
total_transactions = df_filtered['Bill Number'].nunique()
total_qty = df_filtered['Qty'].sum()
aov = total_sales / total_transactions if total_transactions > 0 else 0
aqt = total_qty / total_transactions if total_transactions > 0 else 0

sales_target_monthly = 1_500_000_000 # Contoh Target Bulanan
sales_percentage = (total_sales / sales_target_monthly) if sales_target_monthly > 0 else 0

# --- LAYOUT UTAMA ---
st.title("SALES SUMMARY DASHBOARD")

# --- Baris KPI ---
kpi_cols = st.columns(6)
with kpi_cols[0]:
    st.metric("Sales", f"Rp {total_sales:,.0f}")
with kpi_cols[1]:
    st.metric("Jumlah Transaksi", f"{total_transactions:,}")
with kpi_cols[2]:
    st.metric("Qty Produk", f"{total_qty:,}")
with kpi_cols[3]:
    st.metric("AOV", f"Rp {aov:,.0f}")
with kpi_cols[4]:
    st.metric("AQT", f"{aqt:,.1f}")
with kpi_cols[5]:
    st.metric("Target Realisasi", f"{sales_percentage:.2%}")

st.markdown("---")

# --- Baris Grafik Tren Waktu ---
chart_cols = st.columns(3)
with chart_cols[0]:
    sales_trend = df_filtered.groupby(df_filtered['Date'].dt.date)['Sales'].sum().reset_index()
    fig_sales = px.bar(sales_trend, x='Date', y='Sales', title="Sales")
    fig_sales.update_layout(yaxis_title="Omset (Rp)", xaxis_title=None)
    st.plotly_chart(fig_sales, use_container_width=True)
with chart_cols[1]:
    trans_trend = df_filtered.groupby(df_filtered['Date'].dt.date)['Bill Number'].nunique().reset_index()
    fig_trans = px.bar(trans_trend, x='Date', y='Bill Number', title="Jumlah Transaksi")
    fig_trans.update_layout(yaxis_title="Jumlah Transaksi", xaxis_title=None)
    st.plotly_chart(fig_trans, use_container_width=True)
with chart_cols[2]:
    qty_trend = df_filtered.groupby(df_filtered['Date'].dt.date)['Qty'].sum().reset_index()
    fig_qty = px.bar(qty_trend, x='Date', y='Qty', title="Qty Produk")
    fig_qty.update_layout(yaxis_title="Jumlah Qty", xaxis_title=None)
    st.plotly_chart(fig_qty, use_container_width=True)

st.markdown("---")

# --- Baris Analisis Detail ---
detail_cols = st.columns([2, 1]) # Kolom kiri lebih besar
with detail_cols[0]:
    # Sales by Store
    store_sales = df_filtered.groupby('Branch')['Sales'].sum().sort_values(ascending=False).reset_index()
    fig_store = px.bar(store_sales, x='Sales', y='Branch', orientation='h', title="Sales by Store")
    fig_store.update_layout(yaxis_title=None, xaxis_title="Omset (Rp)")
    st.plotly_chart(fig_store, use_container_width=True)
    
    # Top 10 Sales by Varian
    varian_sales = df_filtered.groupby('Menu')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_varian = px.bar(varian_sales.sort_values('Sales', ascending=True), x='Sales', y='Menu', orientation='h', title="Top 10 Sales by Varian")
    fig_varian.update_layout(yaxis_title=None, xaxis_title="Omset (Rp)")
    st.plotly_chart(fig_varian, use_container_width=True)

with detail_cols[1]:
    # Sales by Channel
    channel_sales = df_filtered.groupby('Area')['Sales'].sum().reset_index()
    fig_channel = px.pie(channel_sales, names='Area', values='Sales', title="Sales by Channel", hole=0.3)
    st.plotly_chart(fig_channel, use_container_width=True)

    # Sales by Product Category
    category_sales = df_filtered.groupby('Menu Category Detail')['Sales'].sum().sort_values(ascending=False).reset_index()
    fig_category = px.bar(category_sales, x='Sales', y='Menu Category Detail', orientation='h', title="Sales by Product")
    fig_category.update_layout(yaxis_title=None, xaxis_title="Omset (Rp)")
    st.plotly_chart(fig_category, use_container_width=True)