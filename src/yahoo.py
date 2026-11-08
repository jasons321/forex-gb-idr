import os
import sys

import pandas as pd
import yaml
import yfinance as yf

# Anchored to the repo root so files land in one place regardless of
# whether you run from the project root or from notebooks/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "yahoo")


def load_tickers():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["yahoo"], cfg.get("start_date", "2015-01-01")


def download(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def get_data_incremental(name, ticker, start="2015-01-01"):
    """Fetch `ticker` into data/raw/yahoo/<name>.csv, only downloading
    dates after what is already saved."""
    cache_path = os.path.join(RAW_DIR, f"{name}.csv")

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        last_date = cached.index[-1].strftime("%Y-%m-%d")
        new_data = download(ticker, start=last_date)
        combined = pd.concat([cached, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        print(f"{name:10} +{len(combined) - len(cached)} rows (total {len(combined)})")
    else:
        combined = download(ticker, start=start)
        print(f"{name:10} {len(combined)} rows (full history)")

    os.makedirs(RAW_DIR, exist_ok=True)
    combined.to_csv(cache_path)
    return combined


def fetch_all():
    tickers, start = load_tickers()
    for name, ticker in tickers.items():
        try:
            get_data_incremental(name, ticker, start)
        except Exception as e:
            print(f"{name:10} FAILED: {e}")


def load(name):
    return pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"), index_col=0, parse_dates=True)


if __name__ == "__main__":
    fetch_all()
