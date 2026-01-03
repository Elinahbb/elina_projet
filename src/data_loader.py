"""
Data loading and preprocessing for Bitcoin price & news sentiment.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    btc_path,               # path to the Bitcoin price CSV file
    news_path,              # path to the news sentiment CSV file
    test_size=0.2,          # proportion of data used for testing
    random_state=42         # random seed for reproducibility
):
    """
    Load Bitcoin price data and news sentiment data,
    merge them at a daily frequency, construct features and target variable,
    and split the data into training and testing sets.
    
    This function prepares the dataset for supervised machine learning
    while avoiding common time-series pitfalls such as look-ahead bias.
    """

    # ---------------------------
    # 1. Load datasets
    # ---------------------------
    
    # Load raw Bitcoin price data and news sentiment data from CSV files
    btc_df = pd.read_csv(btc_path)
    news_df = pd.read_csv(news_path)

    # ---------------------------
    # 2. Parse dates
    # ---------------------------
    
    # Convert date columns to datetime
    btc_df["date"] = pd.to_datetime(btc_df["date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # ---------------------------
    # 3. Keep relevant period
    # ---------------------------
    
    # Keep only dates where sentiment data is available
    btc_df = btc_df[btc_df["date"] >= "2021-11-05"]

    # ---------------------------
    # 4. Create daily return
    # ---------------------------
    
    # Compute daily percentage returns of Bitcoin prices
    btc_df["return"] = btc_df["close"].pct_change()

    # Binary classification target:
    # 1 if the Bitcoin price increases, 0 otherwise
    btc_df["target"] = (btc_df["return"] > 0).astype(int)

    # Remove rows with missing values created by return calculation
    btc_df = btc_df.dropna()

    # ---------------------------
    # 5. Aggregate news sentiment by day
    # ---------------------------
    
    # Aggregate multiple news articles per day by computing the average daily sentiment score
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
    
    # Merge Bitcoin price data with daily news sentiment using a left join to preserve all trading days
    df = pd.merge(btc_df, daily_news, on="date", how="left")

    # If no news is available on a given day, assume neutral sentiment (value = 0)
    df["mean_sentiment"] = df["mean_sentiment"].fillna(0)

    # ---------------------------
    # 7. Features and target
    # ---------------------------
    
    # Use yesterday's return and sentiment to predict today's price movement
    df["return_lag1"] = df["return"].shift(1)
    df["mean_sentiment_lag1"] = df["mean_sentiment"].shift(1)
    
    # Remove the first observation where lagged values are unavailable
    df = df.dropna()

    # Feature matrix (lagged return and sentiment)
    X = df[["return_lag1", "mean_sentiment_lag1"]]
    
    # Binary target variable
    y = df["target"]

    # ---------------------------
    # 8. Train / test split
    # ---------------------------
    
    # Split the data while preserving chronological order (important for time-series data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )

    # ---------------------------
    # 9. Feature scaling
    # ---------------------------
    
    # Scale features before training the models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
