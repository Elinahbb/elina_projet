"""
Main script for predicting Bitcoin price direction using news sentiment.

# Runs the full ML workflow: load data, train models, evaluate, and visualize.
"""

from src.data_loader import load_and_prepare_data
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_knn
)
from src.evaluation import evaluate_model


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a clean and consistent visualization style
sns.set(style="whitegrid")


def main():
    
    """
    Main function that runs all steps of the project sequentially.
    """
    
    print("=" * 60)
    print("Bitcoin Price Direction Prediction")
    print("=" * 60)

    
    # Load and preprocess data
    
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        btc_path="data/raw/BTC.csv",
        news_path="data/raw/bitcoin_sentiments_21_24.csv"
    )

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    
    # Train models
    
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    
    # Evaluate models
    
    results = {}
    results["Logistic Regression"] = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results["Random Forest"] = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results["KNN"] = evaluate_model(knn_model, X_test, y_test, "KNN")

    
    # Display best model
    
    best_model = max(results, key=lambda m: results[m]["accuracy"])

    print("=" * 60)
    print(
        f"Best model: {best_model} "
        f"(Accuracy = {results[best_model]['accuracy']:.3f})"
    )
    print("=" * 60)

    
    # Baseline model (naive strategy) : always predict that Bitcoin price goes up.
    # This naive benchmark helps assess whether machine learning models provide real predictive value.
    
    # Always predict class = 1 (price goes up)
    baseline_pred = pd.Series(1, index=y_test.index)

    # Accuracy: proportion of actual upward movements
    baseline_acc = (y_test == 1).mean()

    # Precision: among predicted "up", proportion of correct predictions
    baseline_precision = (y_test == 1).sum() / len(y_test)

    # Recall: all actual upward movements are predicted as up
    baseline_recall = 1.0
    
    print("\nBaseline Model")
    print("-" * 40)
    print(f"Accuracy : {baseline_acc:.3f}")
    print(f"Precision: {baseline_precision:.3f}")
    print(f"Recall   : {baseline_recall:.3f}")

    results["Baseline (Always Up)"] = {
        "accuracy": baseline_acc,
        "precision": baseline_precision,
        "recall": baseline_recall
    }
    
    
    # Save model performance
    
    performance_df = pd.DataFrame([
        {
            "Model": model,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"]
        }
        for model, metrics in results.items()
    ])
    
    performance_df.to_csv("results/model_performance.csv", index=False)
    print("\nModel performance saved to results/model_performance.csv")
    

    # Prepare data for visualizations
    # Reload raw BTC prices and news sentiment
    # Compute returns and daily average sentiment, then lag sentiment by one day
    
    btc_df = pd.read_csv("data/raw/BTC.csv")
    news_df = pd.read_csv("data/raw/bitcoin_sentiments_21_24.csv")

    btc_df["date"] = pd.to_datetime(btc_df["date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    btc_df = btc_df[btc_df["date"] >= "2021-11-05"]

    btc_df["return"] = btc_df["close"].pct_change()
    btc_df = btc_df.dropna()

    daily_news = (news_df.groupby(news_df["Date"].dt.date)["Accurate Sentiments"].mean().reset_index())
    daily_news.columns = ["date", "mean_sentiment"]
    daily_news["date"] = pd.to_datetime(daily_news["date"])

    df = pd.merge(btc_df, daily_news, on="date", how="left")
    df["mean_sentiment"] = df["mean_sentiment"].fillna(0)

    df["mean_sentiment_lag1"] = df["mean_sentiment"].shift(1)
    df = df.dropna()

    
    # Graph 1 : Sentiment distribution
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df["mean_sentiment_lag1"], bins=20, kde=True)
    plt.title("Distribution of News Sentiment (Lagged by One Day)")
    plt.xlabel("Sentiment (t-1)")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.savefig("results/sentiment_distribution.png")
    plt.close()


    # Graph 2 : Sentiment vs Bitcoin return

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="mean_sentiment_lag1", y="return", data=df)
    plt.title("Yesterday's Sentiment vs Today's Bitcoin Return")
    plt.xlabel("Sentiment (t-1)")
    plt.ylabel("Bitcoin return (t)")
    plt.tight_layout()
    plt.savefig("results/sentiment_vs_return.png")
    plt.close()


    # Graph 3 : Rolling averages of sentiment and returns

    plt.figure(figsize=(10, 6))
    df_rolling = (df.set_index("date")[["mean_sentiment_lag1", "return"]].rolling(7).mean())
    plt.plot(df_rolling.index, df_rolling["mean_sentiment_lag1"], label="Sentiment (7-day MA)")
    plt.plot(df_rolling.index, df_rolling["return"], label="Bitcoin Return (7-day MA)")
    plt.legend()
    plt.title("7-Day Rolling Averages of Sentiment and Bitcoin Returns")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("results/rolling_sentiment_return.png")
    plt.close()

    
    # Correlation analysis
    # Check if yesterday's sentiment is related to today's Bitcoin return
    
    corr = df["mean_sentiment_lag1"].corr(df["return"])
    print(f"\nCorrelation between yesterday's sentiment and today's BTC return: {corr:.3f}")

    with open("results/sentiment_return_correlation.txt", "w") as f:
        f.write(f"Correlation between yesterday's sentiment and today's BTC return: {corr:.3f}\n")

    
    # Additional Figure: Class distribution
    # This figure shows the proportion of upward vs downward price movements

    plt.figure(figsize=(6, 4))
    y_test.value_counts(normalize=True).sort_index().plot(kind="bar")
    plt.title("Distribution of Bitcoin Price Movements")
    plt.xlabel("Price Direction (0 = Down, 1 = Up)")
    plt.ylabel("Proportion of Days")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
