"""
Main entry point for the Bitcoin news sentiment project.
"""

from src.data_loader import load_and_prepare_data
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_knn
)
from src.evaluation import evaluate_model


"""ChatGPT"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid")





def main():
    print("=" * 60)
    print("Bitcoin Price Direction Prediction")
    print("=" * 60)

    # ---------------------------
    # Load data
    # ---------------------------
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
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # ---------------------------
    # Evaluate models
    # ---------------------------
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

    

    
    #CHATGPT
    # ---------------------------
    # Visualisations
    # ---------------------------

    # Recharger les données complètes pour visualisation
    btc_df = pd.read_csv("data/raw/BTC.csv")
    news_df = pd.read_csv("data/raw/bitcoin_sentiments_21_24.csv")

    btc_df["date"] = pd.to_datetime(btc_df["date"])
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    # Même période que le ML
    btc_df = btc_df[btc_df["date"] >= "2021-11-05"]

    # Retour journalier
    btc_df["return"] = btc_df["close"].pct_change()
    btc_df = btc_df.dropna()

    # Sentiment moyen par jour
    daily_news = (
        news_df
        .groupby(news_df["Date"].dt.date)["Accurate Sentiments"]
        .mean()
        .reset_index()
    )

    daily_news.columns = ["date", "mean_sentiment"]
    daily_news["date"] = pd.to_datetime(daily_news["date"])

    # Merge prix + news
    df = pd.merge(btc_df, daily_news, on="date", how="left")
    df["mean_sentiment"] = df["mean_sentiment"].fillna(0)

    # Sentiment à T-1 pour expliquer retour à T
    df["mean_sentiment_lag1"] = df["mean_sentiment"].shift(1)
    df = df.dropna()

    # ---------------------------
    # Graph 1 : Distribution du sentiment
    # ---------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["mean_sentiment_lag1"], bins=20, kde=True)
    plt.title("Distribution du sentiment (décalé d'un jour)")
    plt.xlabel("Sentiment (T-1)")
    plt.ylabel("Nombre de jours")
    plt.tight_layout()
    plt.savefig("results/sentiment_distribution.png")
    plt.close()

    # ---------------------------
    # Graph 2 : Sentiment vs retour
    # ---------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="mean_sentiment_lag1",
        y="return",
        data=df
    )
    plt.title("Sentiment d'hier vs retour Bitcoin aujourd'hui")
    plt.xlabel("Sentiment (T-1)")
    plt.ylabel("Retour BTC (T)")
    plt.tight_layout()
    plt.savefig("results/sentiment_vs_return.png")
    plt.close()

    # ---------------------------
    # Graph 3 : Moyennes mobiles
    # ---------------------------
    plt.figure(figsize=(10, 6))
    df_rolling = (
        df
        .set_index("date")[["mean_sentiment_lag1", "return"]]
        .rolling(7)
        .mean()
    )

    plt.plot(df_rolling.index, df_rolling["mean_sentiment_lag1"], label="Sentiment 7j")
    plt.plot(df_rolling.index, df_rolling["return"], label="Retour BTC 7j")
    plt.legend()
    plt.title("Sentiment et retour Bitcoin (moyenne mobile 7 jours)")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.tight_layout()
    plt.savefig("results/rolling_sentiment_return.png")
    plt.close()


    
    
    
    # ---------------------------
    # Winner
    # ---------------------------
    winner = max(results, key=results.get)

    print("=" * 60)
    print(f"Best model: {winner} ({results[winner]:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
