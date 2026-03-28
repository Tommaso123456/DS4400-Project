# DS4400-Project
Final project for DS4400, this will be a Logistic Regression + Random Forest + Boosted Trees model to predict fraudulent credit card transactions


This project builds a machine learning system to detect fraudulent credit card transactions. The model takes transaction features as input and predicts whether a transaction is legitimate or fraudulent (binary classification).

To ensure strong performance on highly imbalanced data, we combine multiple models—including Logistic Regression, Random Forest, Gradient Boosting, and a Neural Network—and apply techniques such as resampling (e.g., SMOTE) to better capture rare fraud cases.

The system is evaluated using metrics like precision, recall, F1-score, and PR-AUC, with a focus on maximizing fraud detection while minimizing false positives.