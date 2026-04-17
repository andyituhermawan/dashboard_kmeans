import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Executive Credit Dashboard", layout="wide")

# CSS Sederhana untuk memastikan ruang di bagian atas
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv('kmeans_dashboard.csv')

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat file CSV: {e}")
    st.stop()

# --- 3. CURRENCY FORMATTER ---
def format_currency(x):
    if x >= 1_000_000:
        return f'Rp {x/1_000_000:.2f} M'
    elif x >= 1_000:
        return f'Rp {x/1_000:.1f} k'
    return f'Rp {x:,.0f}'

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    all_segments = sorted(df['segment_name'].unique().tolist())
    selected_segment = st.selectbox("Pilih Segmen (untuk Deep Dive):", all_segments)
    
    st.markdown("---")
    st.write("Dashboard ini menampilkan analisis portofolio kredit berdasarkan segmentasi nasabah.")

# Filter data untuk Tab 2
segment_data = df[df['segment_name'] == selected_segment]

# --- 5. MAIN CONTENT (TABS) ---
# Navigasi utama diletakkan di luar kontainer apapun agar terlihat di atas
tab1, tab2 = st.tabs(["📊 PORTFOLIO OVERVIEW", "👤 SEGMENT DEEP DIVE"])

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tab1:
    st.header("🏦 Executive Credit Portfolio Dashboard")
    
    # Scorecards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Outstanding", format_currency(df['loan_amount'].sum()))
    m2.metric("Avg Default Prob", f"{df['default_probability'].mean()*100:.2f}%")
    high_risk = len(df[df['risk_category'].astype(str).str.contains('1|High', case=False, na=False)])
    m3.metric("High Risk Clients", f"{high_risk:,}")
    m4.metric("Avg DTI Ratio", f"{df['dti_ratio'].mean()*100:.1f}%")

    st.markdown("---")

    # Row 1: Distribution Charts
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.plotly_chart(px.pie(df, names='segment_name', hole=0.5, title="Market Share per Segment"), use_container_width=True)
    with c2:
        seg_dist = df['segment_name'].value_counts().reset_index()
        st.plotly_chart(px.bar(seg_dist, x='segment_name', y='count', text_auto=True, title="Volume Nasabah", color='segment_name'), use_container_width=True)

    # Row 2: Aggregate Table
    st.subheader("📍 Final Cluster Profiling")
    features = ['loan_amount', 'borrower_age', 'monthly_income', 'dti_ratio', 'dpd']
    summary = df.groupby('segment_name')[features].mean().reset_index()
    
    st.table(summary.style.format({
        'loan_amount': format_currency, 'monthly_income': format_currency,
        'borrower_age': '{:.0f} thn', 'dti_ratio': lambda x: f'{x*100:.1f}%', 'dpd': '{:.2f} hari'
    }).background_gradient(cmap='YlGnBu', subset=['loan_amount', 'monthly_income', 'dti_ratio']))

# ==============================
# TAB 2: DEEP DIVE
# ==============================
with tab2:
    if not segment_data.empty:
        cluster_id = int(segment_data['cluster'].iloc[0])
        st.header(f"Persona Profile: {selected_segment} 🌱")
        
        row_main = st.columns([1.2, 2, 1.5])
        
        with row_main[0]:
            img_path = f"{cluster_id}.png"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x300.png?text=Avatar", use_container_width=True)
            
            p_size = (len(segment_data) / len(df)) * 100
            st.metric("Portfolio Contribution", f"{p_size:.1f}%")
            st.progress(p_size / 100)

        with row_main[1]:
            st.markdown("#### 📊 Key Statistics")
            sc1, sc2 = st.columns(2)
            sc1.metric("Avg Age", f"{segment_data['borrower_age'].mean():.0f} yrs")
            sc2.metric("Avg Income", format_currency(segment_data['monthly_income'].mean()))
            sc1.metric("DTI Ratio", f"{segment_data['dti_ratio'].mean()*100:.1f}%")
            sc2.metric("Max DPD", f"{segment_data['dpd'].max()} days")
            st.markdown("---")
            avg_p = segment_data['default_probability'].mean()
            st.write(f"**Risk Level (PD):** {avg_p:.4f}")
            st.progress(min(float(avg_p), 1.0))

        with row_main[2]:
            st.markdown("#### 👤 Character Profile")
            st.info(f"Profil nasabah dalam segmen **{selected_segment}**.")
            st.success("**Behavior:** Histori pembayaran stabil.")
            st.warning("**Recommendation:** Pantau rasio DTI.")

        st.markdown("---")
        
        # Recommendations & Histogram
        c_low1, c_low2 = st.columns(2)
        with c_low1:
            st.subheader("⚙️ Strategic Recommendations")
            st.markdown("- Perketat monitoring untuk DPD > 0\n- Penawaran produk limit rendah")
        with c_low2:
            st.plotly_chart(px.histogram(segment_data, x="monthly_income", title="Income Distribution (Within Segment)"), use_container_width=True)
    else:
        st.error("Data segmen tidak ditemukan.")
