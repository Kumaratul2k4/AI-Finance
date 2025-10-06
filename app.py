import os
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

from modules.data import get_data, export_csv
from modules.features import clean_and_engineer

# LLM (Gemini)
try:
    import google.generativeai as genai 
except Exception:  
    genai = None  

# Provide your Gemini API key here (code-only as requested). Do NOT commit real keys to version control.
HARD_CODED_GEMINI_API_KEY: str = ""
st.set_page_config(page_title="FinanceAI", page_icon="⚡", layout="wide")

SIDEBAR_CSS = """
<style>
    section[data-testid="stSidebar"] > div {width: 260px;}
    .brand-title {font-weight: 700; font-size: 1.5rem; display:flex; gap:.5rem; align-items:center}
    .brand-sub {font-size:.85rem; color:#6b7280;}
    .metric-card {padding:16px;border:1px solid #e5e7eb;border-radius:10px;background: #fff;}
</style>
"""
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

st.sidebar.markdown('<div class="brand-title">⚡ FinanceAI</div>', unsafe_allow_html=True)
st.sidebar.caption("AI-Powered Stock Price Prediction Dashboard")

# --- Gemini helpers ---
@st.cache_resource(show_spinner=False)
def init_gemini_model(api_key: str, model_name: str = "gemini-2.5-pro"):
    """Initialize and return a Gemini GenerativeModel. Falls back if model not found."""
    if genai is None:
        raise RuntimeError("google-generativeai package not installed. Please install it first.")
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(model_name)
    except Exception:
        # fallback to commonly available model
        return genai.GenerativeModel("gemini-1.5-pro")


def build_insights_prompt(symbol: str, prices: pd.DataFrame, feats: pd.DataFrame, next_pred: float | None = None) -> str:
    tail = feats.tail(60).copy() if not feats.empty else prices.tail(60).copy()
    cols = [c for c in ["Open","High","Low","Close","Volume","SMA_5","SMA_20","RSI_14","Volatility_20"] if c in tail.columns]
    sample_csv = tail[cols].to_csv(index=True)

    latest = feats.iloc[-1] if not feats.empty else prices.iloc[-1]
    parts = [
        f"You are an equity research assistant. Analyze {symbol} based on the recent 60 trading days.",
        "Summarize trend, momentum, notable support/resistance (from SMAs), RSI interpretation, and volatility.",
        "Be concise (120-180 words) and actionable. Avoid disclaimers.",
        "\nRecent data (CSV):\n" + sample_csv,
        f"Latest close: {float(latest['Close']):.2f}" if 'Close' in latest else "",
    ]
    if next_pred is not None:
        parts.append(f"Model next-day predicted close: {next_pred:.2f}")
    return "\n".join([p for p in parts if p])

# ADD: central place to read API key from code (env or secrets)
def get_gemini_api_key() -> str | None:
    if HARD_CODED_GEMINI_API_KEY:
        return HARD_CODED_GEMINI_API_KEY
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


pages = {
    "Dashboard": "Overview & key metrics",
    "Data Acquisition": "Fetch & manage stock data",
    "Feature Engineering": "Technical indicators & features",
    "Prediction Center": "AI/ML price forecasting",
    "Data Visualization": "Charts & analysis",
    "Model Performance": "Compare model accuracy",
}

page = st.sidebar.radio("Navigate", options=list(pages.keys()), format_func=lambda x: x, index=0)
st.sidebar.write("---")

# Sidebar: Gemini controls (no secrets stored on disk)
# AI Insights controls only visible on Data Visualization page
if page == "Data Visualization":
    st.sidebar.subheader("AI Insights (Gemini)")
    ai_enabled = st.sidebar.toggle("Enable Gemini", value=False)
    # Show model selection only when AI is enabled
    if ai_enabled:
        default_model = st.session_state.get("gemini_model", "gemini-2.5-pro")
        model_name = st.sidebar.selectbox(
            "Model",
            ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"].index(default_model),
        )
        st.session_state["gemini_model"] = model_name
else:
    ai_enabled = False
 
@st.cache_data(ttl=60*60)
def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
     return get_data(symbol, start, end, use_cache=True)

@st.cache_data(ttl=60*60)
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    return clean_and_engineer(df)


def kpi_row(df: pd.DataFrame):
    if df.empty:
        return
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100 if prev['Close'] else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Last Close", f"{last['Close']:.2f}", f"{change:+.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Daily Change %", f"{pct:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Volume", f"{int(last.get('Volume', 0)):,}")
        st.markdown('</div>', unsafe_allow_html=True)


def plot_candles(df: pd.DataFrame):
    if df.empty:
        st.info("No data to plot.")
        return
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='royalblue')))
    if 'SMA_5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name='SMA 5', line=dict(color='orange')))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def split_train_test(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    n = len(X)
    k = max(1, int(n * (1 - test_ratio)))
    return X.iloc[:k, :], X.iloc[k:, :], y.iloc[:k], y.iloc[k:]


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }


def train_eval_lr(X: pd.DataFrame, y: pd.Series, features_df: pd.DataFrame):
    Xtr, Xte, ytr, yte = split_train_test(X, y)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    m = metrics_dict(yte, preds)
    next_pred = float(pipe.fit(X, y).predict(features_df[X.columns].tail(1))[0])
    return m, preds, yte, next_pred


def train_eval_rf(X: pd.DataFrame, y: pd.Series, features_df: pd.DataFrame):
    Xtr, Xte, ytr, yte = split_train_test(X, y)
    pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('model', RandomForestRegressor(random_state=42, n_estimators=300))])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    m = metrics_dict(yte, preds)
    next_pred = float(pipe.fit(X, y).predict(features_df[X.columns].tail(1))[0])
    return m, preds, yte, next_pred


def train_eval_arima(close_series: pd.Series):
    n = len(close_series)
    split = max(30, int(n * 0.8))
    train, test = close_series.iloc[:split], close_series.iloc[split:]
    try:
        model = ARIMA(train.values, order=(1,1,1))
        fit = model.fit()
        preds = fit.forecast(steps=len(test))
        m = metrics_dict(test.values, preds)
        fit_full = ARIMA(close_series.values, order=(1,1,1)).fit()
        next_pred = float(fit_full.forecast(steps=1)[0])
    except Exception:
        preds = np.repeat(train.values[-1], len(test))
        m = metrics_dict(test.values, preds)
        next_pred = float(train.values[-1])
    return m, preds, test.values, next_pred


if page == "Dashboard":
    st.title("Dashboard")
    with st.expander("Data Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value=st.session_state.get('symbol', 'AAPL')).upper()
        with col2:
            start = st.date_input("Start Date", value=st.session_state.get('start', date.today() - timedelta(days=365*2)))
        with col3:
            end = st.date_input("End Date", value=st.session_state.get('end', date.today()))
        fetch = st.button("Fetch Data", type="primary")

    if fetch or 'prices' not in st.session_state or st.session_state.get('symbol') != symbol or st.session_state.get('start') != start or st.session_state.get('end') != end:
        if fetch:
            load_prices.clear()
        st.session_state['symbol'] = symbol
        st.session_state['start'] = start
        st.session_state['end'] = end
        with st.spinner("Loading prices..."):
            prices = load_prices(symbol, str(start), str(end))
        st.session_state['prices'] = prices
        if not prices.empty:
            st.session_state['features'] = make_features(prices)

    prices = st.session_state.get('prices', pd.DataFrame())
    if prices is None:
        prices = pd.DataFrame()

    kpi_row(prices)
    st.subheader("Price Chart")
    plot_candles(prices)

elif page == "Data Acquisition":
    st.title("Data Acquisition")
    symbol = st.text_input("Stock Symbol", value=st.session_state.get('symbol', 'AAPL')).upper()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", value=st.session_state.get('start', date.today() - timedelta(days=365)))
    with col2:
        end = st.date_input("End", value=st.session_state.get('end', date.today()))

    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("Load Data", type="primary"):
        load_prices.clear()
        with st.spinner("Downloading..."):
            prices = load_prices(symbol, str(start), str(end))
        st.success(f"Loaded {len(prices):,} rows for {symbol}")
        st.session_state['symbol'] = symbol
        st.session_state['start'] = start
        st.session_state['end'] = end
        st.session_state['prices'] = prices
        if not prices.empty:
            st.session_state['features'] = make_features(prices)

    prices = st.session_state.get('prices', pd.DataFrame())
    st.dataframe(prices.tail(200))

    if not prices.empty and c2.button("Export CSV"):
        path = export_csv(prices)
        st.success(f"Exported to {path}")

elif page == "Feature Engineering":
    st.title("Feature Engineering")
    prices = st.session_state.get('prices', pd.DataFrame())
    if prices.empty:
        st.info("Load data first from the Dashboard or Data Acquisition page.")
    else:
        feats = make_features(prices)
        st.session_state['features'] = feats
        st.dataframe(feats.tail(100))
        st.write("Feature Columns:", [c for c in feats.columns if c not in ['Open','High','Low','Close','Adj Close','Volume']])
        st.subheader("Chart with Moving Averages")
        plot_candles(feats)

elif page == "Prediction Center":
    st.title("Prediction Center")
    prices = st.session_state.get('prices', pd.DataFrame())
    feats = st.session_state.get('features', pd.DataFrame())
    if prices.empty or feats.empty:
        st.info("Please load data and compute features first.")
    else:
        feature_cols = [c for c in feats.columns if c != 'Close']
        X = feats[feature_cols].iloc[:-1]
        y = feats['Close'].shift(-1).dropna()

        model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "ARIMA(1,1,1)"])
        next_pred = None
        if st.button("Train & Evaluate", type="primary"):
            if model_choice == "Linear Regression":
                m, preds, yte, next_pred = train_eval_lr(X, y, feats)
            elif model_choice == "Random Forest":
                m, preds, yte, next_pred = train_eval_rf(X, y, feats)
            else:
                m, preds, yte, next_pred = train_eval_arima(prices['Close'])

            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{m['RMSE']:.3f}")
            c2.metric("MAE", f"{m['MAE']:.3f}")
            c3.metric("R²", f"{m['R2']:.3f}")

            st.subheader("Test Predictions vs Actual")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=yte if isinstance(yte, (list, np.ndarray)) else yte.values, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted'))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Next-Day Predicted Close: {next_pred:.2f}")

elif page == "Data Visualization":
    st.title("Data Visualization")
    prices = st.session_state.get('prices', pd.DataFrame())
    if prices.empty:
        st.info("Load data first.")
    else:
        feats = st.session_state.get('features')
        if feats is None or (isinstance(feats, pd.DataFrame) and feats.empty):
            feats = make_features(prices)
        st.session_state['features'] = feats
        st.subheader("Candlestick + SMAs")
        plot_candles(feats)
        st.subheader("RSI")
        if 'RSI_14' in feats.columns:
            st.line_chart(feats['RSI_14'])
        else:
            st.info("RSI not available.")

        # AI insights on visualization
        if ai_enabled:
            st.write("---")
            st.subheader("AI Insights")
            col_ai1, col_ai2 = st.columns([1,2])
            with col_ai1:
                gen_btn = st.button("Generate Insights", type="secondary")
            with col_ai2:
                model_used = st.selectbox("LLM Model", [st.session_state.get("gemini_model", "gemini-2.5-pro")], disabled=True)
            if gen_btn:
                if genai is None:
                    st.error("google-generativeai is not installed. Please install it in your environment.")
                elif not get_gemini_api_key():
                    st.info("Add your Gemini key to HARD_CODED_GEMINI_API_KEY in app.py, then rerun.")
                else:
                    with st.spinner("Generating insights..."):
                        try:
                            model = init_gemini_model(get_gemini_api_key(), st.session_state.get("gemini_model","gemini-2.5-pro"))
                            prompt = build_insights_prompt(symbol=st.session_state.get('symbol','SYMBOL'), prices=prices, feats=feats)
                            resp = model.generate_content(prompt)
                            st.write(resp.text)
                        except Exception as e:
                            st.warning(f"Gemini error: {e}")

elif page == "Model Performance":
    st.title("Model Performance")
    prices = st.session_state.get('prices', pd.DataFrame())
    feats = st.session_state.get('features', pd.DataFrame())
    if prices.empty or feats.empty:
        st.info("Please load data and compute features first.")
    else:
        feature_cols = [c for c in feats.columns if c != 'Close']
        X = feats[feature_cols].iloc[:-1]
        y = feats['Close'].shift(-1).dropna()
        lr_m, lr_preds, lr_y, lr_next = train_eval_lr(X, y, feats)
        rf_m, rf_preds, rf_y, rf_next = train_eval_rf(X, y, feats)
        ar_m, ar_preds, ar_y, ar_next = train_eval_arima(prices['Close'])

        rows = [
            {"Model": "Linear Regression", **lr_m, "NextDay": lr_next},
            {"Model": "Random Forest", **rf_m, "NextDay": rf_next},
            {"Model": "ARIMA(1,1,1)", **ar_m, "NextDay": ar_next},
        ]
        perf = pd.DataFrame(rows).sort_values("RMSE")
        st.dataframe(perf, use_container_width=True)
        st.bar_chart(perf.set_index('Model')['RMSE'])
        best = perf.iloc[0]
        st.success(f"Best model: {best['Model']} (RMSE={best['RMSE']:.3f})")