import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Executive Credit Dashboard", layout="wide")

# CSS untuk memastikan dashboard penuh dan Tab terlihat jelas
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: bold;
        font-size: 16px;
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

# --- 3. FIX CURRENCY FORMATTER ---
def format_currency(x):
    if x >= 1_000_000_000:
        return f'Rp {x/1_000_000_000:.2f} M'
    elif x >= 1_000:
        return f'Rp {x/1_000:.1f} k'
    return f'Rp {x:,.0f}'

# --- 4. SIDEBAR GLOBAL FILTER ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    all_segments = sorted(df['segment_name'].unique().tolist())
    selected_segment = st.selectbox("Pilih Segmen (untuk Deep Dive):", all_segments)

segment_data = df[df['segment_name'] == selected_segment]

# --- 5. MAIN NAVIGATION TABS ---
tab1, tab2 = st.tabs(["📊 PORTFOLIO OVERVIEW", "👤 SEGMENT DEEP DIVE"])

# ==============================
# TAB 1: PORTFOLIO OVERVIEW
# ==============================
with tab1:
    st.title("🏦 Executive Credit Portfolio Dashboard")
    
    # SCORECARD
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Outstanding", format_currency(df['loan_amount'].sum()))
    m2.metric("Avg Default Prob", f"{df['default_probability'].mean()*100:.2f}%")
    high_risk = len(df[df['risk_category'].astype(str).str.contains('1|High', case=False, na=False)])
    m3.metric("High Risk Clients", f"{high_risk:,}")
    m4.metric("Avg DTI Ratio", f"{df['dti_ratio'].mean()*100:.1f}%")

    st.markdown("---")

    # ROW 1: DISTRIBUTION
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.plotly_chart(px.pie(df, names='segment_name', hole=0.5, title="Market Share per Segment"), use_container_width=True)
    with col2:
        seg_dist = df['segment_name'].value_counts().reset_index()
        st.plotly_chart(px.bar(seg_dist, x='segment_name', y='count', text_auto=True, title="Volume Nasabah", color='segment_name'), use_container_width=True)

    # ROW 2: TABLE PROFILING
    st.subheader("📍 Final Cluster Profiling")
    features = ['loan_amount', 'borrower_age', 'monthly_income', 'dti_ratio', 'dpd']
    summary = df.groupby('segment_name')[features].mean().reset_index()
    
    st.table(summary.style.format({
        'loan_amount': format_currency, 'monthly_income': format_currency,
        'borrower_age': '{:.0f} thn', 'dti_ratio': lambda x: f'{x*100:.1f}%', 'dpd': '{:.2f} hari'
    }).background_gradient(cmap='YlGnBu', subset=['loan_amount', 'monthly_income', 'dti_ratio']))

    # ROW 3: RADAR (SPIDER) & SNAKE PLOT (YANG SEMPAT HILANG)
    st.subheader("🧠 Segment Personality Analysis")
    c3, c4 = st.columns(2)
    
    # Normalisasi Data
    scaler = MinMaxScaler()
    df_norm = summary.copy()
    df_norm[features] = scaler.fit_transform(df_norm[features])

    with c3:
        fig_radar = go.Figure()
        for i, row in df_norm.iterrows():
            fig_radar.add_trace(go.Scatterpolar(r=row[features].values, theta=features, fill='toself', name=row['segment_name']))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), title="Radar Chart (Normalized)")
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with c4:
        df_snake = df_norm.melt(id_vars='segment_name', var_name='Metric', value_name='Score')
        fig_snake = px.line(df_snake, x='Metric', y='Score', color='segment_name', markers=True, title="Snake Plot")
        st.plotly_chart(fig_snake, use_container_width=True)

    # ROW 4: RISK HEATMAP
    st.subheader("⚖️ Risk Exposure by Segment")
    risk_pivot = pd.crosstab(df['segment_name'], df['risk_category'])
    st.plotly_chart(px.imshow(risk_pivot, text_auto=True, color_continuous_scale='Reds', aspect="auto"), use_container_width=True)

# ==============================
# TAB 2: SEGMENT DEEP DIVE
# ==============================
with tab2:
    if not segment_data.empty:
        cluster_id = int(segment_data['cluster'].iloc[0])
        st.header(f"Persona Profile: {selected_segment} 🌱")
        
        r1, r2, r3 = st.columns([1.2, 2, 1.5])
        with r1:
            img_path = f"{cluster_id}.png"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x300.png?text=Avatar", use_container_width=True)
            
            p_size = (len(segment_data) / len(df)) * 100
            st.metric("Portfolio Contribution", f"{p_size:.1f}%")
            st.progress(p_size / 100)

        with r2:
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

        with r3:
            st.markdown("#### 👤 Character Profile")
            st.info(f"Analisis karakteristik untuk segmen **{selected_segment}**.")
            st.success("**Pay Behavior:** Stabil dan tepat waktu.")
            st.warning("**Advisory:** Monitoring rasio hutang secara berkala.")

        st.markdown("---")
        
        c_low1, c_low2 = st.columns(2)
        with c_low1:
            st.subheader("⚙️ Strategic Recommendations")
            st.markdown("- Optimalisasi cross-selling produk tabungan\n- Peninjauan limit otomatis")
        with c_low2:
            st.plotly_chart(px.histogram(segment_data, x="monthly_income", title="Income Distribution (Within Segment)"), use_container_width=True)
