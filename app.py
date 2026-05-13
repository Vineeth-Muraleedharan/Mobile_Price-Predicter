import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -1px;
    }
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #64748b;
        font-size: 1.05rem;
        margin-top: -10px;
    }
    .price-box {
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .price-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .price-value {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        margin: 20px 0 12px 0;
    }
    .stSelectbox label, .stSlider label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        color: #475569 !important;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# ── Load & Preprocess Data ────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('Cellphone.csv')
    target_col = 'Price'

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    # Skewness — log1p transform
    skewness = df[num_cols].skew()
    highly_skewed = skewness[abs(skewness) > 1].index.tolist()
    df_t = df.copy()
    transformed_cols = []
    for col in highly_skewed:
        if df_t[col].min() >= 0:
            df_t[col] = np.log1p(df_t[col])
            transformed_cols.append(col)

    # Outlier capping
    df_clean = df_t.copy()
    outlier_cols = []
    for col in num_cols + [target_col]:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        if n_out > 0:
            outlier_cols.append(col)
            df_clean[col] = df_clean[col].clip(
                df_clean[col].quantile(0.01),
                df_clean[col].quantile(0.99)
            )

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Alphas
    alphas = np.logspace(-3, 4, 50)

    # Models
    lr = LinearRegression().fit(X_train_sc, y_train)

    ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_sc, y_train)
    ridge = Ridge(alpha=ridge_cv.alpha_).fit(X_train_sc, y_train)

    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train_sc, y_train)
    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000).fit(X_train_sc, y_train)

    enet_cv = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], alphas=alphas,
                           cv=5, max_iter=10000).fit(X_train_sc, y_train)
    enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_,
                      max_iter=10000).fit(X_train_sc, y_train)

    models = {
        'Linear Regression (OLS)': lr,
        'Ridge Regression':        ridge,
        'Lasso Regression':        lasso,
        'ElasticNet Regression':   enet,
    }

    # Metrics
    def get_metrics(model, X_tr, y_tr, X_te, y_te):
        p_tr = model.predict(X_tr)
        p_te = model.predict(X_te)
        n, p = X_tr.shape
        def m(yt, yp):
            r2 = r2_score(yt, yp)
            return {
                'MAE' : round(mean_absolute_error(yt, yp), 4),
                'RMSE': round(np.sqrt(mean_squared_error(yt, yp)), 4),
                'R²'  : round(r2, 4),
                'Adj R²': round(1 - (1-r2)*(n-1)/(n-p-1), 4)
            }
        return {'Train': m(y_tr, p_tr), 'Test': m(y_te, p_te)}

    metrics = {name: get_metrics(m, X_train_sc, y_train, X_test_sc, y_test)
               for name, m in models.items()}

    return (df, df_clean, X, y, X_train_sc, X_test_sc,
            y_train, y_test, scaler, models, metrics,
            num_cols, target_col, transformed_cols)


# ── Load Everything ───────────────────────────────────────────────────────────
with st.spinner("Training models..."):
    (df, df_clean, X, y, X_train_sc, X_test_sc,
     y_train, y_test, scaler, models, metrics,
     num_cols, target_col, transformed_cols) = load_and_train()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📱 Mobile Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict smartphone prices using Linear, Ridge, Lasso & ElasticNet regression</p>', unsafe_allow_html=True)
st.divider()


# ── Sidebar — User Inputs ─────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Phone Specifications")
st.sidebar.markdown("Adjust the sliders to match the phone specs.")
st.sidebar.divider()

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    list(models.keys())
)

st.sidebar.divider()
st.sidebar.markdown("**📋 Specifications**")

user_input = {}
for col in X.columns:
    col_min  = float(df_clean[col].min())
    col_max  = float(df_clean[col].max())
    col_mean = float(df_clean[col].mean())

    if df[col].nunique() <= 10 and df[col].dtype in [np.int64, np.int32]:
        user_input[col] = st.sidebar.selectbox(
            col, sorted(df[col].unique()), index=0
        )
    else:
        # Use original df range for display
        orig_min  = float(df[col].min())
        orig_max  = float(df[col].max())
        orig_mean = float(df[col].mean())
        user_input[col] = st.sidebar.slider(
            col,
            min_value=round(orig_min, 2),
            max_value=round(orig_max, 2),
            value=round(orig_mean, 2),
            step=round((orig_max - orig_min) / 100, 2)
        )


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_price(user_input, model_name):
    input_df = pd.DataFrame([user_input])

    # Apply same log1p transforms
    for col in transformed_cols:
        if col in input_df.columns and input_df[col].min() >= 0:
            input_df[col] = np.log1p(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction   = models[model_name].predict(input_scaled)[0]
    return max(0, prediction)

predicted_price = predict_price(user_input, selected_model)


# ── Layout — Main ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

# Left — Prediction + Model Metrics
with col_left:
    st.markdown(f"""
    <div class="price-box">
        <div class="price-label">Predicted Price</div>
        <div class="price-value">${predicted_price:,.2f}</div>
        <div style="color:#94a3b8; font-size:0.85rem; margin-top:8px;">{selected_model}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    m = metrics[selected_model]
    c1, c2, c3, c4 = st.columns(4)
    for col_ui, label, val in zip(
        [c1, c2, c3, c4],
        ['Test MAE', 'Test RMSE', 'Test R²', 'Test Adj R²'],
        [m['Test']['MAE'], m['Test']['RMSE'], m['Test']['R²'], m['Test']['Adj R²']]
    ):
        col_ui.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Overfitting check
    st.markdown('<div class="section-header">Overfitting Check</div>', unsafe_allow_html=True)
    gap = round(m['Train']['R²'] - m['Test']['R²'], 4)
    status = "⚠️ Possible Overfitting" if gap > 0.10 else "✅ Model looks good"
    st.markdown(f"""
    | | Train R² | Test R² | Gap |
    |--|--|--|--|
    | **{selected_model.split()[0]}** | {m['Train']['R²']} | {m['Test']['R²']} | {gap} |
    """)
    st.info(status)


# Right — Charts
with col_right:

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Actual vs Predicted", "🔍 Feature Importance"])

    # Tab 1 — Model Comparison
    with tab1:
        st.markdown('<div class="section-header">All Models — Train vs Test R²</div>', unsafe_allow_html=True)

        model_names_short = ['OLS', 'Ridge', 'Lasso', 'ElasticNet']
        train_r2 = [metrics[m]['Train']['R²'] for m in models]
        test_r2  = [metrics[m]['Test']['R²']  for m in models]
        test_mae = [metrics[m]['Test']['MAE']  for m in models]
        test_rmse= [metrics[m]['Test']['RMSE'] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # R² grouped bar
        x = np.arange(len(model_names_short))
        w = 0.35
        axes[0].bar(x - w/2, train_r2, w, label='Train R²', color='#0f172a', alpha=0.85)
        axes[0].bar(x + w/2, test_r2,  w, label='Test R²',  color='#38bdf8', alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names_short, fontsize=9)
        axes[0].set_ylabel('R²')
        axes[0].set_title('Train vs Test R²', fontsize=11, fontweight='bold')
        axes[0].set_ylim(0, 1.15)
        axes[0].legend(fontsize=8)
        for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
            axes[0].text(i - w/2, tr + 0.01, f'{tr:.3f}', ha='center', fontsize=7)
            axes[0].text(i + w/2, te + 0.01, f'{te:.3f}', ha='center', fontsize=7)

        # RMSE bar
        colors = ['#0f172a', '#0ea5e9', '#38bdf8', '#7dd3fc']
        bars = axes[1].bar(model_names_short, test_rmse, color=colors, edgecolor='white')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Test RMSE (Lower = Better)', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, test_rmse):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.002 * max(test_rmse),
                         f'{val:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Full metrics table
        st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
        rows = []
        for name in models:
            m = metrics[name]
            rows.append({
                'Model'      : name,
                'Train MAE'  : m['Train']['MAE'],
                'Test MAE'   : m['Test']['MAE'],
                'Train RMSE' : m['Train']['RMSE'],
                'Test RMSE'  : m['Test']['RMSE'],
                'Train R²'   : m['Train']['R²'],
                'Test R²'    : m['Test']['R²'],
            })
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    # Tab 2 — Actual vs Predicted
    with tab2:
        st.markdown('<div class="section-header">Actual vs Predicted — Selected Model</div>', unsafe_allow_html=True)

        model_obj   = models[selected_model]
        y_pred_test = model_obj.predict(X_test_sc)
        residuals   = y_test.values - y_pred_test

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Scatter
        axes[0].scatter(y_test, y_pred_test, alpha=0.5, color='#0ea5e9', edgecolor='none', s=40)
        mn = min(y_test.min(), y_pred_test.min())
        mx = max(y_test.max(), y_pred_test.max())
        axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title(f'Actual vs Predicted\n{selected_model}', fontsize=10, fontweight='bold')
        axes[0].legend(fontsize=8)

        # Residuals
        axes[1].hist(residuals, bins=30, color='#0f172a', edgecolor='white', alpha=0.85)
        axes[1].axvline(0, color='#38bdf8', linestyle='--', lw=2)
        axes[1].set_xlabel('Residual (Actual − Predicted)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tab 3 — Feature Importance
    with tab3:
        st.markdown('<div class="section-header">Feature Coefficients</div>', unsafe_allow_html=True)

        model_obj   = models[selected_model]
        coef_series = pd.Series(model_obj.coef_, index=X.columns).sort_values()
        colors_coef = ['#ef4444' if c < 0 else '#0ea5e9' for c in coef_series]

        fig, ax = plt.subplots(figsize=(9, max(4, len(coef_series) * 0.45)))
        coef_series.plot(kind='barh', color=colors_coef, edgecolor='white', ax=ax)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'Feature Coefficients — {selected_model}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Coefficient Value')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🔵 Blue = positive impact on price  |  🔴 Red = negative impact on price")

        # Lasso zeroed features
        if 'Lasso' in selected_model:
            zeroed = (coef_series == 0).sum()
            st.info(f"Lasso zeroed out **{zeroed}** out of **{len(coef_series)}** features (automatic feature selection).")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("📊 Dataset: Mobile Price Prediction — Kaggle | Models: OLS · Ridge · Lasso · ElasticNet | Built with Streamlit")
