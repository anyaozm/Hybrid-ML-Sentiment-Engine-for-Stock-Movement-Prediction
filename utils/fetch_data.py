import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start: str = "2022-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data
