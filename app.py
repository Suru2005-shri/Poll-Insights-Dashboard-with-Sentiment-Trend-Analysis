"""
app.py  —  Streamlit Dashboard for Poll Results Visualizer
-----------------------------------------------------------
Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.generate_data import generate_poll_data
from src.analysis import (
    load_and_clean, overall_product_votes, region_wise_analysis,
    age_wise_analysis, satisfaction_analysis, nps_analysis,
    monthly_trend, avg_rating_by_product, generate_insights
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Poll Results Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    h1 { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    csv = "data/poll_data.csv"
    if not os.path.exists(csv):
        os.makedirs("data", exist_ok=True)
        generate_poll_data().to_csv(csv, index=False)
    return load_and_clean(csv)

df_full = get_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bar-chart.png", width=72)
st.sidebar.title("🔍 Filters")

regions    = st.sidebar.multiselect("Region",    df_full["region"].unique(),    default=list(df_full["region"].unique()))
age_groups = st.sidebar.multiselect("Age Group", df_full["age_group"].unique(), default=list(df_full["age_group"].unique()))
genders    = st.sidebar.multiselect("Gender",    df_full["gender"].unique(),    default=list(df_full["gender"].unique()))

df = df_full[
    df_full["region"].isin(regions) &
    df_full["age_group"].isin(age_groups) &
    df_full["gender"].isin(genders)
]

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Poll Results Visualizer")
st.markdown("**A complete product preference survey analysis dashboard** — filter by region, age, and gender.")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
votes       = overall_product_votes(df)
_, nps_score = nps_analysis(df)
winner      = votes.iloc[0]["product"]
winner_pct  = votes.iloc[0]["percentage"]

k1.metric("👥 Total Respondents", f"{len(df):,}")
k2.metric("🏆 Top Product",       f"{winner} ({winner_pct}%)")
k3.metric("⭐ Avg Rating",        f"{df['rating'].mean():.2f} / 10")
k4.metric("📊 NPS Score",         nps_score)

st.markdown("---")

# ── Row 1: Overall vote share + Pie ───────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Vote Share by Product")
    fig = px.bar(votes, x="percentage", y="product", orientation="h",
                 color="product", color_discrete_sequence=PALETTE,
                 text="percentage", labels={"percentage": "Vote Share (%)", "product": "Product"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, height=350, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Distribution")
    fig = px.pie(votes, names="product", values="votes",
                 color_discrete_sequence=PALETTE, hole=0.38)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Region heatmap + Monthly trend ─────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Region × Product Heatmap (%)")
    _, rpct = region_wise_analysis(df)
    fig = px.imshow(rpct, text_auto=".1f", color_continuous_scale="Blues",
                    labels=dict(color="Vote %"))
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Monthly Voting Trend")
    trend = monthly_trend(df).reset_index()
    trend_melt = trend.melt(id_vars="month", var_name="Product", value_name="Votes")
    fig = px.line(trend_melt, x="month", y="Votes", color="Product",
                  color_discrete_sequence=PALETTE, markers=True)
    fig.update_layout(height=370, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Age group + Satisfaction ──────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Age Group Preference")
    _, apct = age_wise_analysis(df)
    apct_melt = apct.reset_index().melt(id_vars="age_group", var_name="Product", value_name="Pct")
    fig = px.bar(apct_melt, x="age_group", y="Pct", color="Product",
                 barmode="group", color_discrete_sequence=PALETTE,
                 labels={"Pct": "Vote Share (%)", "age_group": "Age Group"})
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Satisfaction Distribution")
    sat = satisfaction_analysis(df)
    sat_colors = ["#16A34A", "#84CC16", "#EAB308", "#F97316", "#DC2626"]
    fig = px.bar(sat, x="satisfaction", y="percentage", color="satisfaction",
                 color_discrete_sequence=sat_colors,
                 labels={"percentage": "Respondents (%)", "satisfaction": ""},
                 text="percentage")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", showlegend=False)
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

# ── Avg Rating ────────────────────────────────────────────────────────────────
st.subheader("Average Rating by Product")
ratings = avg_rating_by_product(df).sort_values("rating", ascending=False)
fig = px.bar(ratings, x="preferred_product", y="rating",
             color="preferred_product", color_discrete_sequence=PALETTE,
             text="rating", labels={"preferred_product": "Product", "rating": "Avg Rating"})
fig.add_hline(y=df["rating"].mean(), line_dash="dash", line_color="#64748B",
              annotation_text=f"Overall Avg: {df['rating'].mean():.2f}")
fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", showlegend=False)
fig.update_layout(height=380, yaxis_range=[0, 12])
st.plotly_chart(fig, use_container_width=True)

# ── Insights ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Key Insights")
for insight in generate_insights(df):
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data"):
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "poll_data_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Poll Results Visualizer · Built with Python, Pandas, Plotly & Streamlit · GitHub Portfolio Project")
