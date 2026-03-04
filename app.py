import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maximum Likelihood Estimation – Exercise",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f1b35 0%, #1a2a50 100%); }

.hero {
    background: linear-gradient(135deg, #1e3a6e 0%, #2d5fa6 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero h1 { color: #ffffff; font-size: 2.4rem; font-weight: 700; margin: 0 0 6px 0; }
.hero p  { color: #a8c4e8; font-size: 1.05rem; margin: 0; }

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.card h3 { color: #63a4ff; font-size: 1.15rem; font-weight: 600; margin: 0 0 14px 0; }

.metric-box {
    background: linear-gradient(135deg, #1a3a6b, #2457a4);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.30);
}
.metric-box .label { color: #a8c4e8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-box .value { color: #ffffff; font-size: 2.0rem; font-weight: 700; margin-top: 6px; }
.metric-box .sub   { color: #63a4ff; font-size: 0.8rem; margin-top: 4px; }

.formula-box {
    background: rgba(99,164,255,0.08);
    border: 1px solid rgba(99,164,255,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    color: #e0eeff;
    font-size: 1.05rem;
    font-family: 'Georgia', serif;
    letter-spacing: 0.5px;
}

.step-badge {
    display: inline-block;
    background: #2457a4;
    color: #fff;
    border-radius: 50%;
    width: 26px; height: 26px;
    text-align: center;
    line-height: 26px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 10px;
}
.step-row { color: #cfe0ff; margin: 10px 0; font-size: 0.97rem; }

stDataFrame { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
x_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8])
v_vals = np.array([2, 4, 4, 4, 5, 5, 7, 9])
N = len(v_vals)

# MLE computations
mu = v_vals.mean()                                   # population mean
sigma2_pop = np.sum((v_vals - mu)**2) / N           # population variance
sigma2_sam = np.sum((v_vals - mu)**2) / (N - 1)    # sample variance
sigma_pop = np.sqrt(sigma2_pop)
sigma_sam = np.sqrt(sigma2_sam)

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# PDF at each data point (using population sigma)
pdf_points = gaussian_pdf(v_vals, mu, sigma_pop)

# Smooth Gaussian curve
x_smooth = np.linspace(0, 11, 500)
pdf_smooth = gaussian_pdf(x_smooth, mu, sigma_pop)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📊 Maximum Likelihood Estimation</h1>
  <p>Gaussian Probability Density — Interactive Exercise Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Row 1 : Metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-box">
        <div class="label">Data Points (N)</div>
        <div class="value">{N}</div>
        <div class="sub">x = 1 … 8</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-box">
        <div class="label">Mean (μ)</div>
        <div class="value">{mu:.1f}</div>
        <div class="sub">Sum = {v_vals.sum()} / {N}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-box">
        <div class="label">Variance (σ²) — Pop.</div>
        <div class="value">{sigma2_pop:.2f}</div>
        <div class="sub">σ = {sigma_pop:.4f}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-box">
        <div class="label">Variance (σ²) — Sample</div>
        <div class="value">{sigma2_sam:.4f}</div>
        <div class="sub">σ = {sigma_sam:.4f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2 : Data table  +  Steps ──────────────────────────────────────────────
left, right = st.columns([1, 1.6])

with left:
    st.markdown('<div class="card"><h3>📋 Dataset & Probability Density</h3>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Data Point (x)": x_vals,
        "Value  x(i)": v_vals,
        "P(x ; μ, σ)": [f"{p:.8e}" if p < 0.001 else f"{p:.8f}" for p in pdf_points],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>🔢 Step-by-Step Calculation</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="step-row"><span class="step-badge">1</span><b>Collect Data</b> → x values: {list(v_vals)}</div>
    <div class="step-row"><span class="step-badge">2</span><b>Compute Mean (μ)</b> → ({' + '.join(map(str, v_vals))}) / {N} = <b style="color:#63a4ff">{mu}</b></div>
    <div class="step-row"><span class="step-badge">3</span><b>Compute Variance σ² (population)</b> → Σ(xᵢ − μ)² / N = <b style="color:#63a4ff">{sigma2_pop}</b></div>
    <div class="step-row"><span class="step-badge">4</span><b>Compute Variance σ² (sample)</b> → Σ(xᵢ − μ)² / (N−1) ≈ <b style="color:#63a4ff">{sigma2_sam:.4f}</b></div>
    <div class="step-row"><span class="step-badge">5</span><b>Apply Gaussian PDF</b> for each data point:</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
        P(x ; μ, σ) = 1 / (σ√2π) · exp( −(x − μ)² / 2σ² )
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3 : Plots ─────────────────────────────────────────────────────────────
st.markdown('<div class="card"><h3>📈 Gaussian Probability Density Plots</h3>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  P(x) vs Data Points  ", "  Gaussian Curve P(x) vs x  ", "  Raw Value(x) vs x  "])

BLUE      = "#4a90d9"
LIGHT_BLUE= "#63a4ff"
ACCENT    = "#ff8c42"
BG        = "rgba(0,0,0,0)"
GRID      = "rgba(255,255,255,0.08)"
TEXT      = "#cfe0ff"

layout_common = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
    margin=dict(l=50, r=30, t=50, b=50),
    height=420,
)

# — Tab 1 : PDF at discrete data points ——
with tab1:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=v_vals, y=pdf_points,
        marker=dict(
            color=pdf_points,
            colorscale=[[0, "#1a3a6b"], [1, "#63a4ff"]],
            showscale=False,
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        name="P(x ; μ, σ)",
        hovertemplate="x = %{x}<br>P(x) = %{y:.6f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=v_vals, y=pdf_points,
        mode="markers+lines",
        line=dict(color=LIGHT_BLUE, width=2, dash="dot"),
        marker=dict(size=9, color=ACCENT, line=dict(color="white", width=1.5)),
        name="P(x)",
        hovertemplate="x = %{x}<br>P(x) = %{y:.6f}<extra></extra>",
    ))
    fig1.update_layout(
        **layout_common,
        title=dict(text="Probability Density P(x) at Each Data Point", font=dict(size=16, color=TEXT)),
        xaxis_title="Value x(i)",
        yaxis_title="P(x ; μ, σ)",
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

# — Tab 2 : Smooth Gaussian + mean line ——
with tab2:
    fig2 = go.Figure()
    # shaded fill
    fig2.add_trace(go.Scatter(
        x=x_smooth, y=pdf_smooth,
        fill="tozeroy",
        fillcolor="rgba(74,144,217,0.15)",
        line=dict(color=BLUE, width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=x_smooth, y=pdf_smooth,
        mode="lines",
        line=dict(color=LIGHT_BLUE, width=3),
        name="Gaussian PDF",
        hovertemplate="x = %{x:.2f}<br>P(x) = %{y:.6f}<extra></extra>",
    ))
    # scatter data points on curve
    fig2.add_trace(go.Scatter(
        x=v_vals, y=pdf_points,
        mode="markers",
        marker=dict(size=11, color=ACCENT, symbol="circle",
                    line=dict(color="white", width=2)),
        name="Data Points",
        hovertemplate="x = %{x}<br>P(x) = %{y:.6f}<extra></extra>",
    ))
    # mean vertical line
    fig2.add_vline(x=mu, line=dict(color=ACCENT, dash="dash", width=2))
    fig2.add_annotation(x=mu, y=gaussian_pdf(mu, mu, sigma_pop)*1.08,
                        text=f"μ = {mu}", showarrow=False,
                        font=dict(color=ACCENT, size=13))
    fig2.update_layout(
        **layout_common,
        title=dict(text="Gaussian PDF Curve — P(x) vs x", font=dict(size=16, color=TEXT)),
        xaxis_title="x",
        yaxis_title="P(x ; μ, σ)",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

# — Tab 3 : Raw values ——
with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=x_vals, y=v_vals,
        mode="lines+markers",
        line=dict(color=LIGHT_BLUE, width=2.5),
        marker=dict(size=10, color=ACCENT, line=dict(color="white", width=1.5)),
        name="Value(x)",
        hovertemplate="Data Point = %{x}<br>Value = %{y}<extra></extra>",
    ))
    fig3.update_layout(
        **layout_common,
        title=dict(text="Raw Data: Value(x) vs Data Point Index", font=dict(size=16, color=TEXT)),
        xaxis_title="Data Point Index (x)",
        yaxis_title="Value (x)",
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#4a6fa5; font-size:0.8rem; margin-top:20px; padding:10px;">
    RDMU · Maximum Likelihood Estimation Exercise · SP Jain School of Global Management
</div>
""", unsafe_allow_html=True)
