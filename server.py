from __future__ import annotations
import os
from datetime import date
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form, Request
import json
import re

from modules.data import get_data, export_csv
from modules.features import clean_and_engineer

# Load Gemini key from .streamlit/secrets.toml if not in environment
try:
    import tomllib as _tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    _tomllib = None

SECRETS_PATH = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
if "GEMINI_API_KEY" not in os.environ and os.path.isfile(SECRETS_PATH):
    try:
        if _tomllib is not None:
            with open(SECRETS_PATH, "rb") as f:
                _cfg = _tomllib.load(f)
                _key = _cfg.get("GEMINI_API_KEY")
        else:
            _key = None
            with open(SECRETS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY"):
                        _key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if _key:
            os.environ["GEMINI_API_KEY"] = _key
    except Exception:
        pass

# Optional Gemini
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

GEMINI_MODEL_DEFAULT = "gemini-2.5-pro"

app = FastAPI(title="FinanceAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")

def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.index.name is None:
        df = df.copy()
        df.index.name = "Date"
    df = df.reset_index()
    # Ensure JSON serializable types
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(float)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(int)
    return df.to_dict(orient="records")


@app.get("/api/prices")
def api_prices(symbol: str = Query(..., min_length=1), start: str = Query(...), end: str = Query(...)):
    try:
        df = get_data(symbol.upper(), start, end, use_cache=True)
        return {"symbol": symbol.upper(), "rows": to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features")
def api_features(symbol: str, start: str, end: str):
    df = get_data(symbol.upper(), start, end, use_cache=True)
    if df.empty:
        return {"symbol": symbol.upper(), "rows": []}
    feats = clean_and_engineer(df)
    return {"symbol": symbol.upper(), "rows": to_records(feats)}


@app.get("/api/export")
def api_export(symbol: str, start: str, end: str):
    df = get_data(symbol.upper(), start, end, use_cache=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data to export")
    path = export_csv(df)
    return {"path": path}


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


def split_train_test(X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2):
    n = len(X)
    k = max(1, int(n * (1 - test_ratio)))
    return X.iloc[:k, :], X.iloc[k:, :], y.iloc[:k], y.iloc[k:]


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


@app.post("/api/predict")
def api_predict(payload: Dict[str, Any]):
    # payload: {symbol, start, end, model}
    symbol = payload.get("symbol", "AAPL").upper()
    start = payload.get("start")
    end = payload.get("end")
    model_choice = payload.get("model", "Linear Regression")

    df = get_data(symbol, start, end, use_cache=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for given range")
    feats = clean_and_engineer(df)
    feature_cols = [c for c in feats.columns if c != "Close"]
    X = feats[feature_cols].iloc[:-1]
    y = feats["Close"].shift(-1).dropna()

    if len(X) < 5:
        raise HTTPException(status_code=400, detail="Not enough data for training")

    y_true = []
    y_pred = []

    if model_choice == "Linear Regression":
        Xtr, Xte, ytr, yte = split_train_test(X, y)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        metrics = metrics_dict(yte, preds)
        next_pred = float(pipe.fit(X, y).predict(feats[feature_cols].tail(1))[0])
        y_true = yte.tolist() if hasattr(yte, 'tolist') else list(yte)
        y_pred = preds.tolist()
    elif model_choice == "Random Forest":
        Xtr, Xte, ytr, yte = split_train_test(X, y)
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", RandomForestRegressor(random_state=42, n_estimators=300))])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        metrics = metrics_dict(yte, preds)
        next_pred = float(pipe.fit(X, y).predict(feats[feature_cols].tail(1))[0])
        y_true = yte.tolist() if hasattr(yte, 'tolist') else list(yte)
        y_pred = preds.tolist()
    else:  # ARIMA
        close_series = df["Close"]
        n = len(close_series)
        split = max(30, int(n * 0.8))
        train, test = close_series.iloc[:split], close_series.iloc[split:]
        try:
            model = ARIMA(train.values, order=(1, 1, 1))
            fit = model.fit()
            preds = fit.forecast(steps=len(test))
            metrics = metrics_dict(test.values, preds)
            fit_full = ARIMA(close_series.values, order=(1, 1, 1)).fit()
            next_pred = float(fit_full.forecast(steps=1)[0])
        except Exception:
            preds = np.repeat(train.values[-1], len(test))
            metrics = metrics_dict(test.values, preds)
            next_pred = float(train.values[-1])
        y_true = test.values.tolist()
        y_pred = preds.tolist()

    return {
        "symbol": symbol,
        "metrics": metrics,
        "next_pred": next_pred,
        "y_true": y_true,
        "y_pred": y_pred,
    }


@app.post("/api/performance")
def api_performance(payload: Dict[str, Any]):
    symbol = payload.get("symbol", "AAPL").upper()
    start = payload.get("start")
    end = payload.get("end")
    df = get_data(symbol, start, end, use_cache=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for given range")
    feats = clean_and_engineer(df)
    feature_cols = [c for c in feats.columns if c != "Close"]
    X = feats[feature_cols].iloc[:-1]
    y = feats["Close"].shift(-1).dropna()

    # Linear
    Xtr, Xte, ytr, yte = split_train_test(X, y)
    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr.fit(Xtr, ytr)
    lr_preds = lr.predict(Xte)
    lr_m = metrics_dict(yte, lr_preds)
    lr_next = float(lr.fit(X, y).predict(feats[feature_cols].tail(1))[0])

    # RF
    rf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", RandomForestRegressor(random_state=42, n_estimators=300))])
    rf.fit(Xtr, ytr)
    rf_preds = rf.predict(Xte)
    rf_m = metrics_dict(yte, rf_preds)
    rf_next = float(rf.fit(X, y).predict(feats[feature_cols].tail(1))[0])

    # ARIMA
    close_series = df["Close"]
    n = len(close_series)
    split = max(30, int(n * 0.8))
    train, test = close_series.iloc[:split], close_series.iloc[split:]
    try:
        model = ARIMA(train.values, order=(1, 1, 1))
        fit = model.fit()
        ar_preds = fit.forecast(steps=len(test))
        ar_m = metrics_dict(test.values, ar_preds)
        fit_full = ARIMA(close_series.values, order=(1, 1, 1)).fit()
        ar_next = float(fit_full.forecast(steps=1)[0])
    except Exception:
        ar_preds = np.repeat(train.values[-1], len(test))
        ar_m = metrics_dict(test.values, ar_preds)
        ar_next = float(train.values[-1])

    rows = [
        {"Model": "Linear Regression", **lr_m, "NextDay": lr_next},
        {"Model": "Random Forest", **rf_m, "NextDay": rf_next},
        {"Model": "ARIMA(1,1,1)", **ar_m, "NextDay": ar_next},
    ]
    return {"rows": rows}


@app.post("/api/insights")
def api_insights(payload: Dict[str, Any], request: Request):
    if genai is None:
        raise HTTPException(status_code=400, detail="google-generativeai not installed")
    symbol = payload.get("symbol", "AAPL").upper()
    start = payload.get("start")
    end = payload.get("end")
    model_name = payload.get("model", GEMINI_MODEL_DEFAULT)
    # Read API key only from environment/Streamlit secrets
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    df = get_data(symbol, start, end, use_cache=True)
    feats = clean_and_engineer(df) if not df.empty else df

    # Build prompt similar to Streamlit app
    tail = (feats if not feats.empty else df).tail(60).copy()
    cols = [c for c in ["Open","High","Low","Close","Volume","SMA_5","SMA_20","RSI_14","VOL_20"] if c in tail.columns]
    sample_csv = tail[cols].to_csv(index=True)
    latest_close = float(tail['Close'].iloc[-1]) if 'Close' in tail.columns else None

    prompt_parts = [
        f"You are an equity research assistant. Analyze {symbol} based on the recent 60 trading days.",
        "Summarize trend, momentum, notable support/resistance (from SMAs), RSI interpretation, and volatility.",
        "Be concise (120-180 words) and actionable. Avoid disclaimers.",
        "\nRecent data (CSV):\n" + sample_csv,
        f"Latest close: {latest_close:.2f}" if latest_close is not None else "",
    ]
    prompt = "\n".join([p for p in prompt_parts if p])

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        # Normalize inline asterisk separators like "... days. * **Trend..." into Markdown list items
        cleaned = text.replace("\r\n", "\n").strip()
        cleaned = re.sub(r"\s\*\s(?=\*\*)", "\n- ", cleaned)
        return {"text": cleaned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New: analyze uploaded chart screenshot for S/R, stance, entry/SL/targets
@app.post("/api/insights/image")
async def api_insights_image(
    file: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    request: Request = None,
) -> Dict[str, Any]:
    if genai is None:
        raise HTTPException(status_code=400, detail="google-generativeai not installed")
    # Read API key only from environment/Streamlit secrets
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing GEMINI_API_KEY")

    # Validate image
    mime = (file.content_type or "").lower()
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    prompt = (
        "You are a trading assistant. Analyze the provided stock chart image to infer: "
        "1) nearest support and resistance levels, "
        "2) stance as one of: Buy, Sell, or Neutral, "
        "3) suggested entry price, stop-loss, and target price. "
        "Assume the chart is recent and reflects current context. If prices are not clearly readable, infer approximate levels based on structure. "
        "Return ONLY minified JSON with keys: support, resistance, stance, entry, stoploss, target, confidence (0-1), reasoning (<=80 words). "
        "No markdown, no code fences, no extra text."
    )

    try:
        genai.configure(api_key=api_key)
        model_name = model or GEMINI_MODEL_DEFAULT
        gmodel = genai.GenerativeModel(model_name)
        resp = gmodel.generate_content([
            prompt,
            {"mime_type": mime, "data": data},
            (f"Symbol: {symbol}" if symbol else ""),
        ])
        text = getattr(resp, "text", "") or ""

        def _extract_json(s: str):
            if not s:
                return None
            s2 = s.strip()
            # strip fenced code blocks like ```json ... ```
            if s2.startswith("```"):
                s2 = re.sub(r"^```[a-zA-Z]*\s*", "", s2, flags=re.S)
                s2 = re.sub(r"\s*```$", "", s2, flags=re.S)
            # First attempt: direct JSON parse
            try:
                return json.loads(s2)
            except Exception:
                pass
            # Second attempt: grab the first {...} block
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

        parsed = _extract_json(text)
        if not parsed:
            # Fallback empty structure
            parsed = {
                "support": None,
                "resistance": None,
                "stance": "Neutral",
                "entry": None,
                "stoploss": None,
                "target": None,
                "confidence": 0.0,
                "reasoning": text[:180],
            }
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static frontend after all API routes are defined so /api takes precedence
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")