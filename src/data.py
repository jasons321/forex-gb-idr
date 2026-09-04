import os
import pandas as pd
import yfinance as yf

CACHE_PATH = "data/raw/gbpidr.csv"

def get_data_incremental(ticker="GBPIDR=X", cache_path=CACHE_PATH):
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        cached.index = pd.to_datetime(cached.index)
        
        last_date = cached.index[-1].strftime('%Y-%m-%d')
        print(f"Cache found, last date: {last_date}. Checking for new data...")

        new_data = yf.download(ticker, start=last_date)

        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.droplevel(1)

            cached.index = cached.index.tz_localize(None)
            new_data.index = new_data.index.tz_localize(None)

            new_data = new_data.drop(columns=["Volume"], errors="ignore")

            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

            combined.to_csv(cache_path)
            print(f"Added {len(new_data)} rows (updated cache total: {len(combined)} rows).")
            return combined
        else:
            print("No new data available, using cache as-is.")
            return cached
    else:
        print("No cache found, fetching full history...")
        df = yf.download(ticker, start="2015-01-01")
        df = df.drop(columns=["Volume"], errors="ignore")
        
        # Handle MultiIndex for initial download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.index = df.index.tz_localize(None)
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
        return df