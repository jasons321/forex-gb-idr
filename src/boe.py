import io
import os

import pandas as pd
import requests
import yaml

# Anchored to the repo root so files land in one place regardless of
# whether you run from the project root or from notebooks/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "boe")

# Bank of England Statistical Interactive Database, CSV export.
# It rejects requests without a browser-like User-Agent.
URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_series_codes():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["boe"], cfg.get("start_date", "2015-01-01")


def download(code, start):
    params = {
        "csv.x": "yes",
        "Datefrom": pd.Timestamp(start).strftime("%d/%b/%Y"),
        "Dateto": "now",
        "SeriesCodes": code,
        "CSVF": "TN",  # tabular, no titles
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }
    r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if "csv" not in r.headers.get("content-type", ""):
        raise ValueError(f"{code}: expected CSV, got {r.headers.get('content-type')}")

    df = pd.read_csv(io.StringIO(r.text))
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    return df.set_index("DATE").rename_axis("Date")


def fetch(name, code, start="2015-01-01"):
    """Download the full history of `code` into data/raw/boe/<name>.csv.
    These series are small, so it re-downloads everything each run."""
    df = download(code, start)
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(os.path.join(RAW_DIR, f"{name}.csv"))
    print(f"{name:10} {len(df)} rows, last {df.index[-1].date()}")
    return df


def fetch_all():
    codes, start = load_series_codes()
    for name, code in codes.items():
        try:
            fetch(name, code, start)
        except Exception as e:
            print(f"{name:10} FAILED: {e}")


def load(name):
    return pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"), index_col=0, parse_dates=True)


if __name__ == "__main__":
    fetch_all()
