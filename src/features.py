import pandas as pd

def lagged_returns(df, lag_days):
    """
    Calculate lagged returns for a given DataFrame and lag period.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' prices.
    lag_days (int): Number of days to lag for return calculation.

    Returns:
    pd.Series: Series containing the lagged returns.
    """
    return df["Close"].pct_change(periods=lag_days)

def rolling_volatility(df, window_days):
    """
    Calculate rolling volatility for a given DataFrame and window period.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' prices.
    window_days (int): Window size in days for volatility calculation.

    Returns:
    pd.Series: Series containing the rolling volatility.
    """
    return df["Close"].pct_change().rolling(window=window_days).std()

def rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' prices.
    period (int): Period for RSI calculation.

    Returns:
    pd.Series: Series containing the RSI values.
    """
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def macd(df, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' prices.
    short_period (int): Short period for EMA calculation.
    long_period (int): Long period for EMA calculation.
    signal_period (int): Period for signal line calculation.

    Returns:
    pd.DataFrame: DataFrame containing MACD line and signal line.
    """
    short_ema = df["Close"].ewm(span=short_period, adjust=False).mean()
    long_ema = df["Close"].ewm(span=long_period, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line})