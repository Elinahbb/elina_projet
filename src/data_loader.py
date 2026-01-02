"""
Data loading and preprocessing for Bitcoin price & news sentiment.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    btc_path,
    news_path,
    test_size=0.2,
    random_state=42
):
    """
    Load Bitcoin price data and news sentiment data,
    merge them, create target variable, and split into train/test.
    """

    # ---------------------------
    # 1. Load datasets
    # ---------------------------
    btc_df = pd.read_csv(btc_path)
    news_df = pd.read_csv(news_path)

    # ---------------------------
    # 2. Parse dates
    # ---------------------------
    btc_df["date"] = pd.to_datetime(btc_df["date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # ---------------------------
    # 3. Keep relevant period
    # ---------------------------
    btc_df = btc_df[btc_df["date"] >= "2021-11-05"]

    # ---------------------------
    # 4. Create daily return
    # ---------------------------
    btc_df["return"] = btc_df["close"].pct_change()

    # Target: 1 if price goes up, 0 otherwise
    btc_df["target"] = (btc_df["return"] > 0).astype(int)

    btc_df = btc_df.dropna()

    # ---------------------------
    # 5. Aggregate news sentiment by day
    # ---------------------------
    daily_news = (
        news_df
        .groupby(news_df["Date"].dt.date)["Accurate Sentiments"]
        .mean()
        .reset_index()
    )

    daily_news.columns = ["date", "mean_sentiment"]
    daily_news["date"] = pd.to_datetime(daily_news["date"])

    # ---------------------------
    # 6. Merge datasets
    # ---------------------------
    df = pd.merge(btc_df, daily_news, on="date", how="left")

    # If no news that day → neutral sentiment
    df["mean_sentiment"] = df["mean_sentiment"].fillna(0)

    # ---------------------------
    # 7. Features and target
    # ---------------------------
    # Use YESTERDAY's return and sentiment to predict TODAY's direction
    df["return_lag1"] = df["return"].shift(1)
    df["mean_sentiment_lag1"] = df["mean_sentiment"].shift(1)
    
    df = df.dropna()  # Remove first row (no lag available)

    X = df[["return_lag1", "mean_sentiment_lag1"]]  # ← Use lagged return
    y = df["target"]

    # ---------------------------
    # 8. Train / test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )

    # ---------------------------
    # 9. Feature scaling
    # ---------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
