# Bitcoin News Sentiment: Next-Day Price Prediction

## Research Question
Can Bitcoin news sentiment help predict the next-day direction of Bitcoin prices?

# Setup

## Create environment

```bash
conda env create -f environment.yml

conda activate elina-projet
```

## Usage

Run the main script:
```bash
python main.py
```

Expected output:
The script trains and evaluates multiple classification models (Logistic Regression, Random Forest, KNN) and prints:
- Accuracy, precision, and recall for each model
- A comparison against a naive baseline strategy
- Saved evaluation results in the `results/` directory

## Project Structure

```text
elina_projet/
├── README.md                # Setup and usage instructions
├── PROPOSAL.md              # Project proposal
├── project_report.pdf       # Final project report (PDF)
├── environment.yml          # Conda dependencies
├── main.py                  # Main entry point
├── src/                     # Source code
│   ├── data_loader.py       # Data loading & preprocessing
│   ├── models.py            # Model training
│   └── evaluation.py        # Evaluation metrics
├── data/
│   └── raw/
│       ├── BTC.csv
│       └── bitcoin_sentiments_21_24.csv
├── results/                 # Output figures & metrics
│   ├── sentiment_distribution.png
│   ├── sentiment_vs_return.png
│   ├── rolling_sentiment_return.png
│   ├── class_distribution.png
│   ├── sentiment_return_correlation.txt
│   └── model_performance.csv
└── notebooks/               # Optional exploration notebooks
    └── test.ipynb
```

Note: Jupyter checkpoint files are excluded from version control using a .gitignore file to keep the repository clean.

## Results
- Logistic Regression: ~0.564 accuracy
- Random Forest: ~0.546 accuracy
- KNN: ~0.533 accuracy
- Baseline (Always Up): ~0.512 accuracy
- Winner: Logistic Regression
- Correlation sentiment (T-1) vs return (T): 0.025
- Visualizations show:
  - Distribution of previous day sentiment
  - Scatter plot: sentiment vs next-day return
  - 7-day rolling averages of sentiment and returns
  - Distribution of Bitcoin price movements

## Requirements
- Python 3.11
- scikit-learn, pandas, numpy, matplotlib, seaborn
