import argparse
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, days: int = 400) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    stock = yf.Ticker(ticker.upper())
    df = stock.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker.upper()}.")

    return df


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Golden Cross: MA50 crosses above MA200 → Buy signal
    # Death Cross:  MA50 crosses below MA200 → Sell signal
    valid = df["MA50"].notna() & df["MA200"].notna()
    df["Signal"] = 0
    df.loc[valid, "Signal"] = (df.loc[valid, "MA50"] > df.loc[valid, "MA200"]).astype(int)
    df["Crossover"] = df["Signal"].diff()
    return df


def print_summary(df: pd.DataFrame, ticker: str) -> None:
    close = df["Close"]
    ma50 = df["MA50"].dropna()
    ma200 = df["MA200"].dropna()
    rsi = df["RSI"].dropna()

    latest_date = df.index[-1].strftime("%Y-%m-%d")
    latest_close = close.iloc[-1]

    crossovers = df[df["Crossover"] != 0].dropna(subset=["Crossover"])

    print("=" * 55)
    print(f"  {ticker.upper()} Stock Analysis")
    print("=" * 55)
    print(f"Period:              {df.index[0].strftime('%Y-%m-%d')} to {latest_date}")
    print(f"Trading days:        {len(df)}")
    print(f"Latest close:        ${latest_close:.2f}")
    print(f"Period high:         ${close.max():.2f}")
    print(f"Period low:          ${close.min():.2f}")
    print(f"Period return:       {((latest_close / close.iloc[0]) - 1) * 100:.2f}%")
    print(f"Avg daily volume:    {int(df['Volume'].mean()):,}")

    if not ma50.empty:
        rel = "above" if latest_close > ma50.iloc[-1] else "below"
        print(f"50-day MA:           ${ma50.iloc[-1]:.2f}  (price is {rel} MA)")
    else:
        print("50-day MA:           Not enough data")

    if not ma200.empty:
        rel = "above" if latest_close > ma200.iloc[-1] else "below"
        print(f"200-day MA:          ${ma200.iloc[-1]:.2f}  (price is {rel} MA)")
    else:
        print("200-day MA:          Not enough data")

    if not rsi.empty:
        latest_rsi = rsi.iloc[-1]
        condition = (
            "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        )
        print(f"RSI (14):            {latest_rsi:.1f}  ({condition})")
    else:
        print("RSI (14):            Not enough data")

    if not crossovers.empty:
        last = crossovers.iloc[-1]
        signal_type = "BUY (Golden Cross)" if last["Crossover"] == 1 else "SELL (Death Cross)"
        signal_date = crossovers.index[-1].strftime("%Y-%m-%d")
        print(f"Last MA signal:      {signal_type} on {signal_date}")
    else:
        print("Last MA signal:      No crossover in period")

    print("=" * 55)


def save_chart(df: pd.DataFrame, ticker: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{ticker.upper()}_analysis.png")

    buy_signals = df[df["Crossover"] == 1]
    sell_signals = df[df["Crossover"] == -1]

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    fig.suptitle(f"{ticker.upper()} Stock Analysis", fontsize=16, fontweight="bold")

    # --- Panel 1: Price, MAs, crossover signals ---
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], label="Close", color="#2196F3", linewidth=1.5)
    ax1.plot(df.index, df["MA50"], label="MA 50", color="#FF9800", linewidth=1.2, linestyle="--")
    ax1.plot(df.index, df["MA200"], label="MA 200", color="#9C27B0", linewidth=1.2, linestyle="--")
    if not buy_signals.empty:
        ax1.scatter(
            buy_signals.index, buy_signals["Close"],
            marker="^", color="#4CAF50", s=120, zorder=5, label="Buy (Golden Cross)"
        )
    if not sell_signals.empty:
        ax1.scatter(
            sell_signals.index, sell_signals["Close"],
            marker="v", color="#F44336", s=120, zorder=5, label="Sell (Death Cross)"
        )
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Volume ---
    ax2 = axes[1]
    bar_colors = [
        "#4CAF50" if c >= o else "#F44336"
        for c, o in zip(df["Close"], df["Open"])
    ]
    ax2.bar(df.index, df["Volume"], color=bar_colors, alpha=0.7, width=1)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))

    # --- Panel 3: RSI ---
    ax3 = axes[2]
    ax3.plot(df.index, df["RSI"], color="#FF5722", linewidth=1.2)
    ax3.axhline(70, color="#F44336", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.axhline(30, color="#4CAF50", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), color="#F44336", alpha=0.2, label="Overbought")
    ax3.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), color="#4CAF50", alpha=0.2, label="Oversold")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI (14)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    return filename


def main():
    parser = argparse.ArgumentParser(description="Stock technical analysis tool")
    parser.add_argument(
        "ticker", nargs="?", default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Fetching {ticker} data...")

    df = fetch_stock_data(ticker)
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_signals(df)

    print_summary(df, ticker)

    chart_path = save_chart(df, ticker)
    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    main()
