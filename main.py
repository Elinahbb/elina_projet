"""
Main entry point of the Bitcoin price direction prediction project.

# This script runs the full machine learning pipeline:
# - data loading and preprocessing
# - model training
# - model evaluation
# - baseline comparison
# - visualization and correlation analysis
"""

from src.data_loader import load_and_prepare_data
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_knn
)
from src.evaluation import evaluate_model


# External libraries used for data manipulation and visualization
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

    
    # ---------------------------
    # Load data
    # ---------------------------
    
    # Load and preprocess Bitcoin price data and news sentiment data.
    # This step includes feature engineering, lag creation, train/test split, and feature scaling.
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        btc_path="data/raw/BTC.csv",
        news_path="data/raw/bitcoin_sentiments_21_24.csv"
    )

    print("Data loaded successfully")
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    # ---------------------------
    # Train models
    # ---------------------------
    
    # Train three different classification models in order to 
    # compare linear, non-linear, and distance-based approaches.
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # ---------------------------
    # Evaluate models
    # ---------------------------
    
    # Evaluate each model on unseen test data using accuracy and a detailed classification report
    results = {}
    results["Logistic Regression"] = evaluate_model(
        lr_model, X_test, y_test, "Logistic Regression"
    )
    results["Random Forest"] = evaluate_model(
        rf_model, X_test, y_test, "Random Forest"
    )
    results["KNN"] = evaluate_model(
        knn_model, X_test, y_test, "KNN"
    )

    
    # ---------------------------
    # Baseline model (naive strategy)
    # ---------------------------
    
    # Baseline strategy: always predict that Bitcoin price goes up.
    # This naive benchmark helps assess whether machine learning models provide real predictive value.
    baseline_acc = (y_test == 1).mean()

    print("\nBaseline Model")
    print("-" * 40)
    print(f"Accuracy (always predict up): {baseline_acc:.3f}")
    results["Baseline (Always Up)"] = baseline_acc


    # ---------------------------
    # Visualisations
    # ---------------------------
    
    # Reload raw datasets to perform exploratory data analysis.
    # This avoids any data leakage from the machine learning pipeline.
    btc_df = pd.read_csv("data/raw/BTC.csv")
    news_df = pd.read_csv("data/raw/bitcoin_sentiments_21_24.csv")

    btc_df["date"] = pd.to_datetime(btc_df["date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # Same period as the machine learning experiment
    btc_df = btc_df[btc_df["date"] >= "2021-11-05"]

    # Compute daily Bitcoin returns
    btc_df["return"] = btc_df["close"].pct_change()
    btc_df = btc_df.dropna()

    # Aggregate news sentiment at the daily level
    daily_news = (
        news_df
        .groupby(news_df["Date"].dt.date)["Accurate Sentiments"]
        .mean()
        .reset_index()
    )

    daily_news.columns = ["date", "mean_sentiment"]
    daily_news["date"] = pd.to_datetime(daily_news["date"])

    # Merge Bitcoin prices and sentiment data
    df = pd.merge(btc_df, daily_news, on="date", how="left")
    df["mean_sentiment"] = df["mean_sentiment"].fillna(0)

    # Use lagged sentiment to explain next-day returns
    df["mean_sentiment_lag1"] = df["mean_sentiment"].shift(1)
    df = df.dropna()

    # ---------------------------
    # Graph 1 : Sentiment distribution
    # ---------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["mean_sentiment_lag1"], bins=20, kde=True)
    plt.title("Distribution of News Sentiment (Lagged by One Day)")
    plt.xlabel("Sentiment (t-1)")
    plt.ylabel("Number of Days")
    plt.tight_layout()
    plt.savefig("results/sentiment_distribution.png")
    plt.close()

    # ---------------------------
    # Graph 2 : Sentiment vs return
    # ---------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="mean_sentiment_lag1",
        y="return",
        data=df
    )
    plt.title("Yesterday's Sentiment vs Today's Bitcoin Return")
    plt.xlabel("Sentiment (t-1)")
    plt.ylabel("Bitcoin return (t)")
    plt.tight_layout()
    plt.savefig("results/sentiment_vs_return.png")
    plt.close()

    # ---------------------------
    # Graph 3 : Rolling averages
    # ---------------------------
    plt.figure(figsize=(10, 6))
    df_rolling = (
        df
        .set_index("date")[["mean_sentiment_lag1", "return"]]
        .rolling(7)
        .mean()
    )

    plt.plot(df_rolling.index, df_rolling["mean_sentiment_lag1"], label="Sentiment (7-day MA)")
    plt.plot(df_rolling.index, df_rolling["return"], label="Bitcoin Return (7-day MA)")
    plt.legend()
    plt.title("7-Day Rolling Averages of Sentiment and Bitcoin Returns")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("results/rolling_sentiment_return.png")
    plt.close()

    
    # ---------------------------
    # Correlation analysis
    # ---------------------------
    
    # Compute Pearson correlation to measure the linear relationship
    # between yesterday's sentiment and today's Bitcoin return.
    corr = df["mean_sentiment_lag1"].corr(df["return"])
    print(f"\nCorrelation between yesterday's sentiment and today's BTC return: {corr:.3f}")

    with open("results/sentiment_return_correlation.txt", "w") as f:
        f.write(f"Correlation between yesterday's sentiment and today's BTC return: {corr:.3f}\n")

    
    # ---------------------------
    # Save model performance
    # ---------------------------
    
    # Save model accuracies to a CSV file for reporting purposes.
    performance_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values())
    })

    performance_df.to_csv(
        "results/model_performance.csv",
        index=False
    )

    print("\nModel performance saved to results/model_performance.csv")

    
    # ---------------------------
    # Winner
    # ---------------------------
    
    # Select and display the best-performing model based on accuracy.
    winner = max(results, key=results.get)

    print("=" * 60)
    print(f"Best model: {winner} ({results[winner]:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
