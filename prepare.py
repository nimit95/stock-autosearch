"""
prepare.py - Data fetching and evaluation for Indian stock market ML.
DO NOT MODIFY - this file is read-only per program.md.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-03-23"
TIME_BUDGET = 300  # seconds (wall-clock training time)
TOP_K = 10  # stocks held in portfolio each rebalance
HOLD_DAYS = 1
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 index
CACHE_DIR = Path.home() / ".cache" / "autoresearch"

NIFTY_100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS",
    "TECHM.NS", "BAJAJFINSV.NS", "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BPCL.NS", "GRASIM.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BRITANNIA.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "DMART.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS",
    "HINDALCO.NS", "VEDL.NS", "GODREJCP.NS", "PIDILITIND.NS", "DABUR.NS",
    "BERGEPAINT.NS", "HAVELLS.NS", "SIEMENS.NS", "BOSCHLTD.NS",
    "AMBUJACEM.NS", "ACC.NS", "DLF.NS", "BANKBARODA.NS", "PNB.NS",
    "INDIGO.NS", "NAUKRI.NS", "ICICIPRULI.NS", "MARICO.NS", "COLPAL.NS",
    "TORNTPHARM.NS", "LUPIN.NS", "AUROPHARMA.NS", "BIOCON.NS",
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
    "TRENT.NS", "PERSISTENT.NS", "LTIM.NS", "PIIND.NS", "SBICARD.NS",
    "HDFCAMC.NS", "ICICIGI.NS", "CANBK.NS", "IOC.NS", "GAIL.NS",
    "PETRONET.NS", "TATAPOWER.NS", "ABB.NS", "MOTHERSON.NS", "ZOMATO.NS",
    "LICI.NS", "IRCTC.NS", "PAGEIND.NS", "MPHASIS.NS", "MAXHEALTH.NS",
    "JUBLFOOD.NS", "SHREECEM.NS", "INDUSTOWER.NS", "JSWENERGY.NS",
]


# ──────────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────────


def _flatten_columns(df):
    """Handle yfinance multi-level column format."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_data():
    """Fetch OHLCV data for all Nifty 100 stocks + benchmark. Caches to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "market_data.pkl"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Fetching data from Yahoo Finance...")
    all_data = {}
    failed = []

    for ticker in NIFTY_100:
        try:
            df = yf.download(ticker, start=TRAIN_START, end=TEST_END, progress=False)
            df = _flatten_columns(df)
            if len(df) >= 252:  # at least ~1 year of data
                all_data[ticker] = df
                print(f"  OK: {ticker} ({len(df)} rows)")
            else:
                failed.append(ticker)
                print(f"  SKIP: {ticker} (only {len(df)} rows)")
        except Exception as e:
            failed.append(ticker)
            print(f"  FAIL: {ticker} ({e})")

    # Benchmark
    print(f"Fetching benchmark {BENCHMARK_TICKER}...")
    benchmark = yf.download(
        BENCHMARK_TICKER, start=TRAIN_START, end=TEST_END, progress=False
    )
    benchmark = _flatten_columns(benchmark)

    data = {"stocks": all_data, "benchmark": benchmark}

    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    print(f"\nFetched {len(all_data)} stocks ({len(failed)} failed), cached to {cache_file}")
    return data


def load_data():
    """Return raw per-stock OHLCV DataFrames and benchmark daily returns."""
    raw_data = fetch_data()
    bench = raw_data["benchmark"]
    benchmark_returns = bench["Close"].pct_change().dropna()

    print(f"Loaded {len(raw_data['stocks'])} stocks")

    return {
        "stocks": raw_data["stocks"],
        "benchmark_daily_returns": benchmark_returns,
    }


# ──────────────────────────────────────────────────────────────
# Evaluation (backtesting)
# ──────────────────────────────────────────────────────────────


def evaluate_strategy(predictions, data):
    """
    Backtest a long-only portfolio with configurable holding period.

    Every HOLD_DAYS days, pick top TOP_K stocks by predicted return (equal weight).
    Hold for HOLD_DAYS days, then rebalance. Daily returns are tracked for sharpe.
    Compare against Nifty 50 benchmark.

    Required keys in data:
        test_dates, test_tickers, y_test (actual 1-day fwd returns),
        benchmark_daily_returns
    """
    df = pd.DataFrame(
        {
            "date": data["test_dates"],
            "ticker": data["test_tickers"],
            "predicted": predictions,
            "actual": data["y_test"],
        }
    )

    risk_free_daily = 0.07 / 252
    all_dates = sorted(df["date"].unique())
    portfolio_returns = []
    ret_dates = []
    cash_days = 0
    rebalance_spearman = []
    rebalance_topk_precision = []
    num_rebalances = 0

    held_tickers = None
    days_since_rebalance = HOLD_DAYS  # force rebalance on first day

    for date in all_dates:
        group = df[df["date"] == date]
        if len(group) < TOP_K:
            continue

        # Rebalance if holding period is up
        if days_since_rebalance >= HOLD_DAYS:
            top_k = group.nlargest(TOP_K, "predicted")
            held_tickers = set(top_k["ticker"].values)
            days_since_rebalance = 0
            num_rebalances += 1

            # Cash signal: sit out if predictions are bearish
            if top_k["predicted"].mean() < 0:
                held_tickers = None
                cash_days += HOLD_DAYS  # approximate

            # Ranking quality (only on rebalance days)
            rho, _ = spearmanr(group["predicted"], group["actual"])
            rebalance_spearman.append(rho)
            actual_top = set(group.nlargest(TOP_K, "actual")["ticker"].values)
            pred_top = set(top_k["ticker"].values)
            rebalance_topk_precision.append(len(pred_top & actual_top) / TOP_K)

        # Daily return from held stocks
        if held_tickers is None:
            daily_return = risk_free_daily
        else:
            held = group[group["ticker"].isin(held_tickers)]
            if len(held) > 0:
                daily_return = held["actual"].mean()
            else:
                daily_return = risk_free_daily

        portfolio_returns.append(daily_return)
        ret_dates.append(date)
        days_since_rebalance += 1

    portfolio_returns = pd.Series(
        portfolio_returns, index=pd.DatetimeIndex(ret_dates)
    )

    # Align benchmark
    bench = data["benchmark_daily_returns"]
    bench = bench.reindex(portfolio_returns.index).fillna(0)

    # --- Sharpe ratio (annualized, 7% risk-free for India) ---
    excess = portfolio_returns - risk_free_daily
    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0
    )

    # --- Total return ---
    total_return_pct = ((1 + portfolio_returns).prod() - 1) * 100

    # --- Benchmark return ---
    benchmark_return_pct = ((1 + bench).prod() - 1) * 100

    # --- Alpha ---
    alpha_pct = total_return_pct - benchmark_return_pct

    # --- Max drawdown ---
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown_pct = abs(drawdown.min()) * 100

    # --- Win rate ---
    win_rate_pct = (portfolio_returns > 0).mean() * 100

    # --- Trade count ---
    trading_days = len(portfolio_returns)
    num_trades = num_rebalances * TOP_K

    # --- Ranking quality ---
    mean_spearman = round(float(np.mean(rebalance_spearman)), 4) if rebalance_spearman else 0.0
    mean_topk_precision = round(float(np.mean(rebalance_topk_precision)), 4) if rebalance_topk_precision else 0.0

    return {
        "sharpe_ratio": round(float(sharpe), 6),
        "alpha_pct": round(float(alpha_pct), 1),
        "total_return_pct": round(float(total_return_pct), 1),
        "benchmark_return_pct": round(float(benchmark_return_pct), 1),
        "max_drawdown_pct": round(float(max_drawdown_pct), 1),
        "win_rate_pct": round(float(win_rate_pct), 1),
        "num_trades": int(num_trades),
        "cash_days": int(cash_days),
        "trading_days": int(trading_days),
        "hold_days": HOLD_DAYS,
        "num_rebalances": int(num_rebalances),
        "mean_spearman": mean_spearman,
        "mean_topk_precision": mean_topk_precision,
    }


def print_results(metrics, training_seconds, total_seconds):
    """Print results in the format expected by program.md."""
    print("---")
    print(f"sharpe_ratio:        {metrics['sharpe_ratio']:.6f}")
    print(f"alpha_pct:           {metrics['alpha_pct']:.1f}")
    print(f"total_return_pct:    {metrics['total_return_pct']:.1f}")
    print(f"benchmark_return_pct: {metrics['benchmark_return_pct']:.1f}")
    print(f"max_drawdown_pct:    {metrics['max_drawdown_pct']:.1f}")
    print(f"win_rate_pct:        {metrics['win_rate_pct']:.1f}")
    print(f"num_trades:          {metrics['num_trades']}")
    print(f"cash_days:           {metrics['cash_days']}")
    print(f"trading_days:        {metrics['trading_days']}")
    print(f"hold_days:           {metrics['hold_days']}")
    print(f"num_rebalances:      {metrics['num_rebalances']}")
    print(f"mean_spearman:       {metrics['mean_spearman']:.4f}")
    print(f"mean_topk_precision: {metrics['mean_topk_precision']:.4f}")
    print(f"training_seconds:    {training_seconds:.1f}")
    print(f"total_seconds:       {total_seconds:.1f}")


# ──────────────────────────────────────────────────────────────
# CLI: fetch and cache data
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching and caching Nifty 100 + benchmark data...")
    fetch_data()
    print("\nDone. Ready to run train.py")
