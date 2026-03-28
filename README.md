# DS4400-Project
Final project for DS4400, this will be a Logistic Regression + Random Forest + Boosted Trees model to predict fraudulent credit card transactions


This project builds a machine learning system to detect fraudulent credit card transactions. The model takes transaction features as input and predicts whether a transaction is legitimate or fraudulent (binary classification).

To ensure strong performance on highly imbalanced data, we combine multiple models—including Logistic Regression, Random Forest, Gradient Boosting, and a Neural Network—and apply techniques such as resampling (e.g., SMOTE) to better capture rare fraud cases.

The system is evaluated using metrics like precision, recall, F1-score, and PR-AUC, with a focus on maximizing fraud detection while minimizing false positives.


HOW TO RUN:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`
2. `pip install -r requirements.txt`
3. `jupyter notebook`
4. Run notebooks in order: 01 → 02 → 03 → 04

The last notebook produces `results/model_comparison.csv` with PR-AUC, precision, recall, and F1 for all four models.