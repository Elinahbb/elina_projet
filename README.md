# Bitcoin News Sentiment: Next-Day Price Prediction

## Research Question
Can Bitcoin news sentiment help predict the next-day direction of Bitcoin prices?

## Setup

# Create environment

cd elina_projet

conda env create -f environment.yml

conda activate elina-projet

## Usage

python main.py

Expected output:
- Accuracy and classification report for Logistic Regression, Random Forest, and KNN
- Visualizations saved in results/: 
  - sentiment_distribution.png
  - sentiment_vs_return.png
  - rolling_sentiment_return.png

## Project Structure

```text
elina_projet/
├── README.md              # Setup, usage, project overview
├── PROPOSAL.md            # Project proposal (300–500 words)
├── project_report.pdf     # Final technical report (PDF)
├── environment.yml        # Conda dependencies
├── main.py                # Entry point (python main.py)
├── src/
│   ├── data_loader.py     # Data loading & preprocessing
│   ├── models.py          # Model definitions
│   └── evaluation.py     # Evaluation & metrics
├── data/
│   └── raw/
│       ├── BTC.csv
│       └── bitcoin_sentiments_21_24.csv
├── results/
│   ├── sentiment_distribution.png
│   ├── sentiment_vs_return.png
│   └── rolling_sentiment_return.png
└── notebooks/
    └── test.ipynb
```


## Results
- Logistic Regression: ~0.564 accuracy
- Random Forest: ~0.546 accuracy
- KNN: ~0.533 accuracy
- Winner: Logistic Regression
- Correlation sentiment (T-1) vs return (T): 0.025
- Visualizations show:
  - Distribution of previous day sentiment
  - Scatter plot: sentiment vs next-day return
  - 7-day rolling averages of sentiment and returns

## Requirements
- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn
