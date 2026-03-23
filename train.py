"""
train.py - Model training for Indian stock market prediction.
This is the ONLY file you modify during experimentation.
"""

import time

import numpy as np
import pandas as pd
import talib
import xgboost as xgb

from prepare import (
    TRAIN_END,
    TEST_START,
    evaluate_strategy,
    load_data,
    print_results,
)


# ── Feature engineering ───────────────────────────────────────
# Modify this function to experiment with different features.


def compute_features(df):
    """Compute technical indicators for a single stock's OHLCV data."""
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    features = pd.DataFrame(index=df.index)

    # --- Trend ---
    features["sma_5"] = talib.SMA(close, timeperiod=5)
    features["sma_10"] = talib.SMA(close, timeperiod=10)
    features["sma_20"] = talib.SMA(close, timeperiod=20)
    features["sma_50"] = talib.SMA(close, timeperiod=50)
    features["ema_12"] = talib.EMA(close, timeperiod=12)
    features["ema_26"] = talib.EMA(close, timeperiod=26)

    features["price_sma5_ratio"] = close / features["sma_5"] - 1
    features["price_sma20_ratio"] = close / features["sma_20"] - 1
    features["price_sma50_ratio"] = close / features["sma_50"] - 1

    # --- MACD ---
    features["macd"], features["macd_signal"], features["macd_hist"] = talib.MACD(close)

    # --- RSI ---
    features["rsi_14"] = talib.RSI(close, timeperiod=14)

    # --- Bollinger Bands ---
    features["bb_upper"], features["bb_middle"], features["bb_lower"] = talib.BBANDS(
        close, timeperiod=20
    )
    bb_width = features["bb_upper"] - features["bb_lower"]
    features["bb_position"] = np.where(
        bb_width > 0, (close - features["bb_lower"]) / bb_width, 0.5
    )

    # --- ATR ---
    features["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    features["atr_ratio"] = features["atr_14"] / close

    # --- ADX ---
    features["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

    # --- CCI ---
    features["cci_14"] = talib.CCI(high, low, close, timeperiod=14)

    # --- Williams %R ---
    features["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)

    # --- Rate of Change ---
    features["roc_10"] = talib.ROC(close, timeperiod=10)

    # --- Stochastic ---
    features["stoch_k"], features["stoch_d"] = talib.STOCH(high, low, close)

    # --- OBV ---
    features["obv"] = talib.OBV(close, volume)
    features["obv_sma"] = talib.SMA(features["obv"].values.astype(float), timeperiod=20)

    # --- Volume ---
    vol_sma = talib.SMA(volume, timeperiod=20)
    features["volume_ratio"] = np.where(vol_sma > 0, volume / vol_sma, 1.0)

    # --- Returns ---
    close_series = pd.Series(close, index=df.index)
    ret = close_series.pct_change()
    features["return_1d"] = ret.values
    features["return_5d"] = close_series.pct_change(5).values
    features["return_10d"] = close_series.pct_change(10).values
    features["return_20d"] = close_series.pct_change(20).values

    # --- Volatility ---
    features["volatility_20d"] = ret.rolling(20).std().values

    return features


# ── Dataset construction ──────────────────────────────────────
# Modify compute_target() to experiment with different labels.
# y_test is ALWAYS 1-day forward return (used by evaluate_strategy for P&L).
# y_train can be anything — it's what the model learns to predict.


def compute_target(df):
    """Return the training target series. Modify this for target experiments.

    Examples:
        df["Close"].pct_change().shift(-1)          # 1-day forward return (default)
        df["Close"].pct_change(5).shift(-5)          # 5-day forward return
        (df["Close"].shift(-1) > df["Close"]).astype(float)  # binary up/down
    """
    return df["Close"].pct_change().shift(-1)


def build_dataset(stocks, benchmark_returns):
    """Build train/test datasets from raw stock data."""
    chunks = []

    for ticker, df in stocks.items():
        features = compute_features(df)

        fwd_1d = df["Close"].pct_change().shift(-1)  # fixed: for evaluation P&L
        target = compute_target(df)  # experimental: for training

        combined = features.copy()
        combined["_fwd_1d"] = fwd_1d
        combined["_target"] = target
        combined["_ticker"] = ticker

        # Drop rows where features have NaN (lookback warmup)
        feature_cols = list(features.columns)
        combined = combined.dropna(subset=feature_cols)

        chunks.append(combined)

    all_data = pd.concat(chunks)
    feature_names = [c for c in all_data.columns if not c.startswith("_")]

    train = all_data[all_data.index <= pd.Timestamp(TRAIN_END)]
    test = all_data[all_data.index >= pd.Timestamp(TEST_START)]

    # Drop NaN targets independently so different horizons don't shrink test set
    train = train.dropna(subset=["_target"])
    test = test.dropna(subset=["_fwd_1d"])

    print(
        f"Dataset: {len(train)} train rows, {len(test)} test rows, "
        f"{len(feature_names)} features"
    )

    return {
        "X_train": train[feature_names].values,
        "y_train": train["_target"].values,
        "X_test": test[feature_names].values,
        "y_test": test["_fwd_1d"].values,
        "test_dates": test.index.values,
        "test_tickers": test["_ticker"].values,
        "feature_names": feature_names,
        "benchmark_daily_returns": benchmark_returns,
    }


# ── Training ──────────────────────────────────────────────────


def main():
    total_start = time.time()

    raw = load_data()
    data = build_dataset(raw["stocks"], raw["benchmark_daily_returns"])

    train_start = time.time()

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(data["X_train"], data["y_train"])

    training_seconds = time.time() - train_start

    predictions = model.predict(data["X_test"])
    metrics = evaluate_strategy(predictions, data)

    total_seconds = time.time() - total_start
    print_results(metrics, training_seconds, total_seconds)


if __name__ == "__main__":
    main()
