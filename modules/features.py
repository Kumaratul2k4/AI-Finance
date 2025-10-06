import pandas as pd
import numpy as np


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Simple moving averages
    out['SMA_5'] = out['Close'].rolling(5).mean()
    out['SMA_20'] = out['Close'].rolling(20).mean()
    # Exponential moving average
    out['EMA_12'] = out['Close'].ewm(span=12, adjust=False).mean()
    out['EMA_26'] = out['Close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = out['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out['RSI_14'] = 100 - (100 / (1 + rs))

    # Volatility
    out['RET'] = out['Close'].pct_change()
    out['VOL_20'] = out['RET'].rolling(20).std() * np.sqrt(252)

    # Lags
    out['Close_Lag1'] = out['Close'].shift(1)
    out['Close_Lag5'] = out['Close'].shift(5)
    out['Close_Lag10'] = out['Close'].shift(10)

    out = out.drop(columns=['RET'])
    return out


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_missing(df)
    df = add_technical_features(df)
    df = df.dropna()
    return df