import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BoostArena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0d0f14;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {
    color: #8b92a8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1px dashed #2a2f45 !important;
    border-radius: 10px !important;
    background: #0d0f14 !important;
    padding: 1rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0d0f14 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Slider styling ── */
[data-testid="stSlider"] .stSlider > div > div > div > div {
    background: #534AB7 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 0 24px rgba(83, 74, 183, 0.45) !important;
}

/* ── Main content text ── */
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li, span { color: #8b92a8; }

/* ── Metric cards (custom) ── */
.metric-card {
    background: #111318;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.ada::before  { background: #D85A30; }
.metric-card.gb::before   { background: #378ADD; }
.metric-card.xgb::before  { background: #1D9E75; }
.metric-card.winner::before { background: linear-gradient(90deg, #534AB7, #1D9E75); }

.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5168;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #e2e8f0;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #4a5168;
    margin-top: 0.4rem;
}
.metric-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    margin-top: 0.5rem;
}
.badge-ada  { background: rgba(216,90,48,0.15);  color: #D85A30; }
.badge-gb   { background: rgba(55,138,221,0.15); color: #378ADD; }
.badge-xgb  { background: rgba(29,158,117,0.15); color: #1D9E75; }
.badge-win  { background: rgba(83,74,183,0.15);  color: #7F77DD; }

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e2130;
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #534AB7;
    flex-shrink: 0;
}
.section-title {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #4a5168 !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* ── Dataset preview table ── */
[data-testid="stDataFrame"] {
    background: #111318 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
}

/* ── Info / warning boxes ── */
.stAlert {
    background: #111318 !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #8b92a8 !important;
}

/* ── Plotly chart container ── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Stat bar ── */
.stat-bar-wrap {
    background: #1a1d27;
    border-radius: 6px;
    height: 6px;
    width: 100%;
    margin-top: 0.5rem;
    overflow: hidden;
}
.stat-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

/* ── Model legend strip ── */
.legend-strip {
    display: flex;
    gap: 1.5rem;
    padding: 0.75rem 1rem;
    background: #111318;
    border: 1px solid #1e2130;
    border-radius: 8px;
    margin-bottom: 1rem;
    align-items: center;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #8b92a8;
    font-weight: 500;
}
.legend-line {
    width: 24px;
    height: 2.5px;
    border-radius: 2px;
}

/* ── Tab styling override ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2130 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5168 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    padding: 0.6rem 1.25rem !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #534AB7 !important;
}

/* ── Spinner ── */
.stSpinner > div { color: #534AB7 !important; }

/* ── Hero headline ── */
.hero-wrap {
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 0.5rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(83,74,183,0.12);
    border: 1px solid rgba(83,74,183,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7F77DD;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    line-height: 1.15 !important;
    margin: 0 0 0.75rem !important;
    letter-spacing: -0.02em !important;
}
.hero-title span { color: #7F77DD; }
.hero-sub {
    font-size: 0.95rem !important;
    color: #4a5168 !important;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Comparison table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.compare-table th {
    text-align: left;
    padding: 0.5rem 1rem;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5168;
    font-weight: 600;
    border-bottom: 1px solid #1e2130;
}
.compare-table td {
    padding: 0.65rem 1rem;
    color: #8b92a8;
    border-bottom: 1px solid #0d0f14;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table .highlight-row td { color: #e2e8f0; background: #1a1d27; }

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.3;
}
.empty-title {
    font-size: 1.1rem !important;
    color: #4a5168 !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}
.empty-sub {
    font-size: 0.85rem !important;
    color: #2a2f45 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem;">
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; letter-spacing:-0.01em;">⚡ BoostArena</div>
        <div style="font-size:0.7rem; color:#2a2f45; margin-top:0.2rem; letter-spacing:0.06em; text-transform:uppercase;">Model Race · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#4a5168; font-weight:600; margin-bottom:0.5rem;">Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()

    st.markdown('<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#4a5168; font-weight:600; margin-bottom:0.5rem; margin-top:0.5rem;">Hyperparameters</p>', unsafe_allow_html=True)

    n_estimators = st.slider("Trees per model", 10, 200, 100, step=10,
                              help="More trees = slower but potentially better")
    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01,
                               help="Step size shrinkage for each boosting round")
    test_size = st.slider("Test split %", 10, 40, 20, step=5,
                           help="Percentage of data held out for validation")

    st.divider()

    run_btn = st.button("⚡  Launch Race", use_container_width=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">⚡ Live Boosting Benchmark</div>
    <h1 class="hero-title">Watch three algorithms<br><span>race to the minimum.</span></h1>
    <p class="hero-sub">Upload a classification dataset, configure the parameters, and see AdaBoost, Gradient Boosting, and XGBoost compete iteration by iteration.</p>
</div>
""", unsafe_allow_html=True)

# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No dataset uploaded yet</div>
        <div class="empty-sub">Drop a CSV file in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & preview data ───────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)

# Dataset stats row
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rows</div>
        <div class="metric-value">{df_raw.shape[0]:,}</div>
        <div class="metric-sub">total samples</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Columns</div>
        <div class="metric-value">{df_raw.shape[1]}</div>
        <div class="metric-sub">features + target</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    missing = df_raw.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Missing values</div>
        <div class="metric-value">{missing:,}</div>
        <div class="metric-sub">auto-imputed</div>
    </div>""", unsafe_allow_html=True)
with col_d:
    cat_cols = df_raw.select_dtypes(include=['object', 'category']).shape[1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Categorical cols</div>
        <div class="metric-value">{cat_cols}</div>
        <div class="metric-sub">one-hot encoded</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# Target selector + preview
st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Dataset preview</p></div>', unsafe_allow_html=True)

default_index = list(df_raw.columns).index('ProdTaken') if 'ProdTaken' in df_raw.columns else 0

col_left, col_right = st.columns([3, 1])
with col_right:
    target_col = st.selectbox("Target column", df_raw.columns, index=default_index, label_visibility="collapsed")
    st.caption(f"**{target_col}** — unique values: {df_raw[target_col].nunique()}")
with col_left:
    st.dataframe(
        df_raw.head(8),
        use_container_width=True,
        height=240,
    )

# ── Training ──────────────────────────────────────────────────────────────────
if run_btn:
    progress_bar = st.progress(0, text="Preprocessing data…")
    status_text  = st.empty()

    with st.spinner(""):

        # ── Preprocessing ──
        df = df_raw.copy()
        id_cols = ['CustomerID', 'id', 'ID', 'Index']
        df = df.drop(columns=[c for c in id_cols if c in df.columns and c != target_col], errors='ignore')

        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y  = le.fit_transform(y)

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=y
        )
        n_classes = len(np.unique(y))

        progress_bar.progress(15, text="Training AdaBoost…")

        # ── AdaBoost ──
        ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        ada.fit(X_train, y_train)
        ada_loss = [log_loss(y_test, p) for p in ada.staged_predict_proba(X_test)]
        ada_acc  = accuracy_score(y_test, ada.predict(X_test))
        try:
            ada_auc = roc_auc_score(y_test, ada.predict_proba(X_test)[:, 1] if n_classes == 2
                                    else ada.predict_proba(X_test), multi_class='ovr')
        except Exception:
            ada_auc = None

        progress_bar.progress(45, text="Training Gradient Boosting…")

        # ── Gradient Boosting ──
        gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        gb.fit(X_train, y_train)
        gb_loss = [log_loss(y_test, p) for p in gb.staged_predict_proba(X_test)]
        gb_acc  = accuracy_score(y_test, gb.predict(X_test))
        try:
            gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1] if n_classes == 2
                                   else gb.predict_proba(X_test), multi_class='ovr')
        except Exception:
            gb_auc = None

        progress_bar.progress(75, text="Training XGBoost…")

        # ── XGBoost ──
        xgb = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_loss = xgb.evals_result()['validation_0']['logloss']
        xgb_acc  = accuracy_score(y_test, xgb.predict(X_test))
        try:
            xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1] if n_classes == 2
                                    else xgb.predict_proba(X_test), multi_class='ovr')
        except Exception:
            xgb_auc = None

        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    iterations = list(range(1, n_estimators + 1))
    final = {
        'AdaBoost':          ada_loss[-1],
        'Gradient Boosting': gb_loss[-1],
        'XGBoost':           xgb_loss[-1],
    }
    winner = min(final, key=final.get)
    winner_map = {'AdaBoost': 'ada', 'Gradient Boosting': 'gb', 'XGBoost': 'xgb'}

    # ── Section: Learning curves ──────────────────────────────────────────────
    st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Validation loss · iteration by iteration</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-strip">
        <div class="legend-item"><div class="legend-line" style="background:#D85A30;"></div>AdaBoost</div>
        <div class="legend-item"><div class="legend-line" style="background:#378ADD;"></div>Gradient Boosting</div>
        <div class="legend-item"><div class="legend-line" style="background:#1D9E75;"></div>XGBoost</div>
        <div style="flex:1;"></div>
        <div style="font-size:0.7rem; color:#2a2f45;">Lower = better</div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Subtle area fills
    fig.add_trace(go.Scatter(
        x=iterations, y=ada_loss, fill='tozeroy',
        fillcolor='rgba(216,90,48,0.04)', line=dict(color='rgba(216,90,48,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=gb_loss, fill='tozeroy',
        fillcolor='rgba(55,138,221,0.04)', line=dict(color='rgba(55,138,221,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=xgb_loss, fill='tozeroy',
        fillcolor='rgba(29,158,117,0.04)', line=dict(color='rgba(29,158,117,0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Main lines
    fig.add_trace(go.Scatter(
        x=iterations, y=ada_loss,
        mode='lines', name='AdaBoost',
        line=dict(color='#D85A30', width=2),
        hovertemplate='<b>AdaBoost</b><br>Tree %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=gb_loss,
        mode='lines', name='Gradient Boosting',
        line=dict(color='#378ADD', width=2),
        hovertemplate='<b>Gradient Boosting</b><br>Tree %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=xgb_loss,
        mode='lines', name='XGBoost',
        line=dict(color='#1D9E75', width=2),
        hovertemplate='<b>XGBoost</b><br>Tree %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))

    # Winner annotation
    winner_loss = final[winner]
    winner_color = {'AdaBoost': '#D85A30', 'Gradient Boosting': '#378ADD', 'XGBoost': '#1D9E75'}[winner]
    fig.add_annotation(
        x=n_estimators, y=winner_loss,
        text=f"  {winner} wins  {winner_loss:.4f}",
        showarrow=True, arrowhead=2, arrowcolor=winner_color,
        font=dict(color=winner_color, size=11, family='JetBrains Mono'),
        ax=-80, ay=-30, arrowwidth=1.5,
        bgcolor='#111318', bordercolor=winner_color, borderwidth=1,
    )

    fig.update_layout(
        plot_bgcolor='#111318',
        paper_bgcolor='#111318',
        font=dict(family='Inter', color='#4a5168', size=12),
        xaxis=dict(
            title='Trees built',
            title_font=dict(size=11, color='#2a2f45'),
            tickfont=dict(size=10, color='#2a2f45'),
            gridcolor='#1a1d27', gridwidth=1,
            zeroline=False, showline=False,
        ),
        yaxis=dict(
            title='Log-loss',
            title_font=dict(size=11, color='#2a2f45'),
            tickfont=dict(size=10, color='#2a2f45'),
            gridcolor='#1a1d27', gridwidth=1,
            zeroline=False, showline=False,
        ),
        legend=dict(
            font=dict(size=11, color='#8b92a8'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            orientation='h',
            x=0, y=1.08,
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1a1d27', font_color='#e2e8f0', bordercolor='#2a2f45'),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Section: Scoreboard ───────────────────────────────────────────────────
    st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Final scoreboard</p></div>', unsafe_allow_html=True)

    cards = [
        ('AdaBoost',          'ada',  ada_loss[-1],  ada_acc,  ada_auc),
        ('Gradient Boosting', 'gb',   gb_loss[-1],   gb_acc,   gb_auc),
        ('XGBoost',           'xgb',  xgb_loss[-1],  xgb_acc,  xgb_auc),
    ]
    worst_loss = max(c[2] for c in cards)
    c1, c2, c3 = st.columns(3)
    for col, (name, cls, loss, acc, auc) in zip([c1, c2, c3], cards):
        is_winner = (name == winner)
        badge_html = f'<span class="metric-badge badge-win">🏆 Winner</span>' if is_winner else f'<span class="metric-badge badge-{cls}">{name}</span>'
        bar_pct = max(0, 100 - int((loss / worst_loss) * 100)) if worst_loss > 0 else 50
        bar_color = {'ada': '#D85A30', 'gb': '#378ADD', 'xgb': '#1D9E75'}[cls]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
                {badge_html}
                <div style="margin-top:0.75rem;">
                    <div class="metric-label">Log-loss</div>
                    <div class="metric-value">{loss:.4f}</div>
                    <div class="stat-bar-wrap">
                        <div class="stat-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-top:1rem; border-top:1px solid #1e2130; padding-top:0.75rem;">
                    <div>
                        <div class="metric-label">Accuracy</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:500; color:#e2e8f0;">{acc:.1%}</div>
                    </div>
                    <div>
                        <div class="metric-label">ROC-AUC</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:500; color:#e2e8f0;">{auc_str}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section: Comparison table ─────────────────────────────────────────────
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Side-by-side comparison</p></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Performance metrics", "Hyperparameter config"])

    with tab1:
        rows = []
        for name, cls, loss, acc, auc in cards:
            win_marker = " 🏆" if name == winner else ""
            rows.append({
                "Model":       name + win_marker,
                "Log-loss ↓":  f"{loss:.4f}",
                "Accuracy ↑":  f"{acc:.4%}",
                "ROC-AUC ↑":   f"{auc:.4f}" if auc is not None else "N/A",
                "Δ from best": f"+{(loss - final[winner]):.4f}" if name != winner else "—",
            })
        compare_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(compare_df, use_container_width=True)

    with tab2:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Trees (n_estimators) | `{n_estimators}` |
        | Learning rate | `{learning_rate}` |
        | Test split | `{test_size}%` |
        | Training samples | `{X_train.shape[0]:,}` |
        | Test samples | `{X_test.shape[0]:,}` |
        | Features after encoding | `{X_train.shape[1]}` |
        | Classes | `{n_classes}` |
        """)

    # ── Section: Convergence insight ─────────────────────────────────────────
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-dot"></div><p class="section-title">Convergence insight</p></div>', unsafe_allow_html=True)

    # Find when each model hits 95% of its final loss improvement
    def convergence_step(losses):
        start, end = losses[0], losses[-1]
        target = start - 0.95 * (start - end)
        for i, l in enumerate(losses):
            if l <= target:
                return i + 1
        return len(losses)

    ada_conv = convergence_step(ada_loss)
    gb_conv  = convergence_step(gb_loss)
    xgb_conv = convergence_step(xgb_loss)

    ci1, ci2, ci3 = st.columns(3)
    for col, (name, cls, conv) in zip([ci1, ci2, ci3], [
        ('AdaBoost', '#D85A30', ada_conv),
        ('Gradient Boosting', '#378ADD', gb_conv),
        ('XGBoost', '#1D9E75', xgb_conv),
    ]):
        pct = int(conv / n_estimators * 100)
        with col:
            st.markdown(f"""
            <div style="background:#111318; border:1px solid #1e2130; border-radius:10px; padding:1rem 1.25rem;">
                <div class="metric-label">{name}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:500; color:{cls};">
                    Tree #{conv}
                </div>
                <div class="metric-sub">95% convergence at {pct}% of budget</div>
                <div class="stat-bar-wrap">
                    <div class="stat-bar-fill" style="width:{pct}%; background:{cls};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)