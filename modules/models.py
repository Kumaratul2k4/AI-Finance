from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]


def make_supervised(df: pd.DataFrame, target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col].shift(-1)  # next-day target
    data = pd.concat([X, y.rename('Target')], axis=1).dropna()
    return data[feature_cols], data['Target']


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred))
    }


def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> ModelResult:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    return ModelResult('LinearRegression', y.values, preds, evaluate(y, preds))


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> ModelResult:
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # RF doesn't need scaling; keep sparse safety
        ('model', RandomForestRegressor(random_state=42))
    ])
    param_grid = {'model__n_estimators': [100, 300], 'model__max_depth': [None, 5, 10]}
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X, y)
    best = grid.best_estimator_
    preds = best.predict(X)
    return ModelResult('RandomForest', y.values, preds, evaluate(y, preds))


def train_arima(close_series: pd.Series) -> ModelResult:
    # ARIMA on Close, forecast next day iteratively over train window
    values = close_series.dropna().values
    preds = []
    true_vals = []
    # walk-forward from a minimum window
    start = max(30, int(len(values) * 0.2))
    for t in range(start, len(values) - 1):
        train = values[:t]
        test_val = values[t+1]
        try:
            model = ARIMA(train, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
        except Exception:
            forecast = train[-1]
        preds.append(forecast)
        true_vals.append(test_val)
    metrics = evaluate(np.array(true_vals), np.array(preds))
    return ModelResult('ARIMA(1,1,1)', np.array(true_vals), np.array(preds), metrics)


def compare_models(X: pd.DataFrame, y: pd.Series, close_series: pd.Series) -> List[ModelResult]:
    results = [
        train_linear_regression(X, y),
        train_random_forest(X, y),
        train_arima(close_series)
    ]
    return results


def predict_next_day(model_name: str, X_last_row: pd.DataFrame, close_series: pd.Series) -> float:
    if model_name == 'LinearRegression':
        pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
        return float(pipe.fit(X_last_row, close_series.shift(-1).dropna()).predict(X_last_row.tail(1))[0])
    elif model_name == 'RandomForest':
        pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('model', RandomForestRegressor(random_state=42))])
        pipe.fit(X_last_row, close_series.shift(-1).dropna())
        return float(pipe.predict(X_last_row.tail(1))[0])
    elif model_name.startswith('ARIMA'):
        try:
            model = ARIMA(close_series.dropna().values, order=(1,1,1))
            fit = model.fit()
            return float(fit.forecast(steps=1)[0])
        except Exception:
            return float(close_series.dropna().iloc[-1])
    else:
        raise ValueError('Unknown model name')