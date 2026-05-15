import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #E3270B;
        text-align: center;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .price-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        margin: 20px 0;
    }
    .price-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .price-amount {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
    }
    .price-model {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 8px;
    }
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #E3270B;
        margin: 20px 0 10px 0;
        padding-left: 10px;
        border-left: 3px solid #E3270B;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 14px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        margin-top: 10px;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.9; }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.85rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
    }
    .tip {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #0369a1;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Train Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df = pd.read_csv('Cellphone.csv')
    target_col = 'Price'

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    # Log1p transform
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
    for col in num_cols + [target_col]:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if ((df_clean[col] < lower) | (df_clean[col] > upper)).sum() > 0:
            df_clean[col] = df_clean[col].clip(
                df_clean[col].quantile(0.01),
                df_clean[col].quantile(0.99)
            )

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    alphas = np.logspace(-3, 4, 50)

    ridge_cv  = RidgeCV(alphas=alphas, cv=5).fit(X_train_sc, y_train)
    lasso_cv  = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train_sc, y_train)
    enet_cv   = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                              alphas=alphas, cv=5, max_iter=10000).fit(X_train_sc, y_train)

    models = {
        'Linear Regression': LinearRegression().fit(X_train_sc, y_train),
        'Ridge Regression':  Ridge(alpha=ridge_cv.alpha_).fit(X_train_sc, y_train),
        'Lasso Regression':  Lasso(alpha=lasso_cv.alpha_, max_iter=10000).fit(X_train_sc, y_train),
        'ElasticNet':        ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_,
                                        max_iter=10000).fit(X_train_sc, y_train),
    }

    # Feature ranges from original df for input bounds
    feature_info = {}
    for col in X.columns:
        feature_info[col] = {
            'min':  float(df[col].min()),
            'max':  float(df[col].max()),
            'mean': float(df[col].mean()),
            'is_categorical': df[col].nunique() <= 10 and df[col].dtype in [np.int64, np.int32],
            'unique_vals': sorted(df[col].unique().tolist()) if df[col].nunique() <= 10 else []
        }

    return models, scaler, X.columns.tolist(), transformed_cols, feature_info


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    models, scaler, feature_cols, transformed_cols, feature_info = train_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">📱 Mobile Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter phone specifications below to predict its market price</p>', unsafe_allow_html=True)

# ── Model Selection ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Select Model</div>', unsafe_allow_html=True)
selected_model = st.selectbox("", list(models.keys()), label_visibility="collapsed")

# ── Phone Specs Input ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Phone Specifications</div>', unsafe_allow_html=True)

# Friendly labels
labels = {
    'Sale':         'Sale Price (original listing)',
    'weight':       'Weight (grams)',
    'resoloution':  'Screen Resolution (inches)',
    'ppi':          'PPI (Pixels Per Inch)',
    'cpu core':     'CPU Cores',
    'cpu freq':     'CPU Frequency (GHz)',
    'internal mem': 'Internal Memory (GB)',
    'ram':          'RAM (GB)',
    'RearCam':      'Rear Camera (MP)',
    'Front_Cam':    'Front Camera (MP)',
    'battery':      'Battery Capacity (mAh)',
    'thickness':    'Thickness (mm)',
    'Product_id':   'Product ID',
}

user_input = {}
cols_per_row = 2
col_list = feature_cols.copy()

# Render inputs in 2-column grid
for i in range(0, len(col_list), cols_per_row):
    row_cols = st.columns(cols_per_row)
    for j, col in enumerate(col_list[i:i+cols_per_row]):
        info  = feature_info[col]
        label = labels.get(col, col)
        with row_cols[j]:

            # 1. Product ID — text input (type only)
            if col == 'Product_id':
                val = st.text_input(label, value=str(int(info['mean'])))
                try:
                    user_input[col] = float(val)
                except:
                    user_input[col] = float(info['mean'])

            # 2. CPU Frequency — slider
            elif col == 'cpu freq':
                user_input[col] = st.slider(
                    label,
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=round(float(info['mean']), 2),
                    step=0.1
                )

            # 3. RAM — dropdown (integers only)
            elif col == 'ram':
                ram_options = sorted(set(
                    range(int(info['min']), int(info['max']) + 1, 1)
                ))
                default_ram = min(ram_options, key=lambda x: abs(x - info['mean']))
                user_input[col] = st.selectbox(
                    label,
                    options=ram_options,
                    index=ram_options.index(default_ram)
                )

            # 4. Battery — integer only (no float)
            elif col == 'battery':
                user_input[col] = st.number_input(
                    label,
                    min_value=int(info['min']),
                    max_value=int(info['max']),
                    value=int(info['mean']),
                    step=1,
                    format="%d"
                )

            # Default — categorical dropdown or number input
            elif info['is_categorical']:
                user_input[col] = st.selectbox(
                    label,
                    options=info['unique_vals'],
                    index=info['unique_vals'].index(
                        min(info['unique_vals'],
                            key=lambda x: abs(x - info['mean']))
                    )
                )
            else:
                user_input[col] = st.number_input(
                    label,
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=round(float(info['mean']), 2),
                    step=round((info['max'] - info['min']) / 100, 2),
                    format="%.2f"
                )

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("")
predict_clicked = st.button("🔍 Predict Price")

if predict_clicked:
    # Preprocess input
    input_df = pd.DataFrame([user_input])

    for col in transformed_cols:
        if col in input_df.columns and input_df[col].min() >= 0:
            input_df[col] = np.log1p(input_df[col])

    input_scaled = scaler.transform(input_df[feature_cols])
    predicted    = max(0, models[selected_model].predict(input_scaled)[0])

    # Show result
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Market Price</div>
        <div class="price-amount">${predicted:,.2f}</div>
        <div class="price-model">Predicted using {selected_model}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tip
    if predicted < 1000:
        segment = "Budget segment phone"
    elif predicted < 2000:
        segment = "Mid-range phone"
    elif predicted < 280000:
        segment = "Upper mid-range phone"
    else:
        segment = "Flagship phone"

    st.markdown(f'<div class="tip">{segment} based on predicted price</div>',
                unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:20px;">
        ☝️ Fill in the specs above and click <b>Predict Price</b>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()

