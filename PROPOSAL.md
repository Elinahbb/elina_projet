# Project Proposal: Bitcoin Price Direction Prediction Using News Sentiment

## Project Category:
Financial / Machine Learning / NLP

## Problem Statement / Motivation:
The goal of this project is to investigate whether news sentiment about Bitcoin can help predict the next-day price movement.  
Financial markets are influenced by public perception and news events, but the relationship between news sentiment and actual price changes is complex and noisy. By studying this relationship, we aim to determine if news can provide actionable signals for predicting whether Bitcoin will go up or down the following day.  
This project is motivated by the growing importance of sentiment analysis in financial decision-making and the challenge of modeling highly volatile assets like cryptocurrencies.

## Planned Approach and Technologies:
1. Load and preprocess two datasets:  
   - Historical Bitcoin price data (open, high, low, close).  
   - Bitcoin news with pre-calculated sentiment scores.  
2. Aggregate news sentiment by day and create lagged features:  
   - Previous day’s return (`return_lag1`).  
   - Previous day’s sentiment (`mean_sentiment_lag1`).  
3. Train machine learning models to predict next-day price direction:  
   - Logistic Regression  
   - Random Forest  
   - K-Nearest Neighbors (KNN)  
4. Evaluate models using accuracy and classification metrics.  
5. Generate visualizations: sentiment distribution, sentiment vs return scatter plot, and 7-day rolling averages.  

## Technologies: Python 3.11+, pandas, scikit-learn, matplotlib, seaborn, JupyterLab.

## Expected Challenges:
- Low signal-to-noise ratio: financial markets are highly volatile, and news sentiment may have only a weak effect.  
- Limited features: using only previous day’s return and sentiment might not capture complex market dynamics.  
- Temporal data handling: train-test splitting must preserve chronological order to avoid leakage.

## Success Criteria:
- Models trained successfully and evaluated on test data.  
- Visualizations generated to illustrate the relationship between sentiment and Bitcoin returns.  
- Clear reporting of correlation between sentiment and next-day returns.  
- Code is reproducible and structured following the project guidelines.

## Stretch Goals:
- Include additional features: trading volume, social media sentiment, or macroeconomic indicators.  
- Improve model performance with feature engineering or hyperparameter tuning.  
- Add interactive visualizations or dashboards to explore the sentiment-price relationship over time.
