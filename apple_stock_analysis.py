import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_apple_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=182)  # ~6 months

    ticker = yf.Ticker("AAPL")
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        raise ValueError("No data returned for AAPL.")

    return df


def calculate_moving_average(df, window=50):
    df = df.copy()
    df[f"MA{window}"] = df["Close"].rolling(window=window).mean()
    return df


def print_summary(df):
    close = df["Close"]
    ma50 = df["MA50"]

    latest_date = df.index[-1].strftime("%Y-%m-%d")
    latest_close = close.iloc[-1]
    latest_ma50 = ma50.dropna().iloc[-1] if not ma50.dropna().empty else None

    print("=" * 50)
    print("  Apple (AAPL) Stock Summary — Last 6 Months")
    print("=" * 50)
    print(f"Period:              {df.index[0].strftime('%Y-%m-%d')} to {latest_date}")
    print(f"Trading days:        {len(df)}")
    print(f"Latest close:        ${latest_close:.2f}")
    print(f"Period high:         ${close.max():.2f}")
    print(f"Period low:          ${close.min():.2f}")
    print(f"Period return:       {((latest_close / close.iloc[0]) - 1) * 100:.2f}%")
    print(f"Avg daily volume:    {int(df['Volume'].mean()):,}")

    if latest_ma50 is not None:
        signal = "above" if latest_close > latest_ma50 else "below"
        print(f"50-day MA:           ${latest_ma50:.2f}  (price is {signal} MA)")
    else:
        print("50-day MA:           Not enough data")

    print("=" * 50)


def main():
    print("Fetching AAPL data...")
    df = fetch_apple_data()
    df = calculate_moving_average(df, window=50)
    print_summary(df)


if __name__ == "__main__":
    main()
