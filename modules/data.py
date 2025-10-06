import os
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(symbol: str, start: str, end: str) -> str:
    raw = f"{symbol}|{start}|{end}".encode()
    return hashlib.md5(raw).hexdigest()  # short stable key


def _fetch_from_stooq(symbol: str) -> pd.DataFrame:
    # Stooq uses symbols like msft.us
    sym = symbol.lower()
    if "." not in sym:
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
        if df.empty:
            return df
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # Align columns to expected names
        df = df.rename(columns={c: c.title() for c in df.columns})
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_data(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Download OHLCV data with local CSV cache and fallback provider.

    Primary: Yahoo Finance via yfinance
    Fallback: Stooq daily data
    """
    key = _cache_key(symbol, start, end)
    cache_path = CACHE_DIR / f"{symbol}_{key}.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return df

    df = pd.DataFrame()
    # Try yfinance first
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns={c: c.title() for c in df.columns})
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
        else:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    # Fallback to Stooq if yfinance failed/empty
    if df.empty:
        full_df = _fetch_from_stooq(symbol)
        if not full_df.empty:
            mask = (full_df.index >= pd.to_datetime(start)) & (full_df.index <= pd.to_datetime(end))
            df = full_df.loc[mask].copy()

    # Cache if we have data
    if use_cache and isinstance(df, pd.DataFrame) and not df.empty:
        tmp = df.copy()
        tmp.index.name = "Date"
        tmp.to_csv(cache_path)

    return df


def export_csv(df: pd.DataFrame, file_path: Optional[str] = None) -> str:
    """Export DataFrame to CSV and return the path. If file_path is None, write into cache dir."""
    if file_path is None:
        file_path = str(CACHE_DIR / "export.csv")
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    out = df.copy()
    if out.index.name is None:
        out.index.name = "Date"
    out.to_csv(file_path)
    return file_path