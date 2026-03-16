# Fraud Detection Microservice

A real-time fraud scoring microservice built using FastAPI and XGBoost.
The system evaluates financial transactions and returns a fraud probability score and decision (ALLOW, CHALLENGE, BLOCK).

The model was trained on a dataset of 10,000 credit card transactions. The machine learning pipeline began with exploratory data analysis (EDA) to assess class imbalance, detect skewed distributions, and identify missing values.Categorical merchant categories were transformed using one-hot encoding (pandas.get_dummies()), and all features were converted to integer format for compatibility with XGBoost.Because fraud datasets are typically highly imbalanced, the scale_pos_weight parameter was calculated using the negative-to-positive class ratio and applied during training.

The model was trained using:
300 estimators
Maximum depth: 6
Learning rate: 0.05
Row and column subsampling: 80%
Model performance was evaluated using:
ROC-AUC
Precision-Recall AUC
5-fold cross-validation

To improve fraud detection recall, threshold tuning was performed across a range of 0.1–0.9, with the final threshold set to 0.25, prioritising fraud detection while controlling false positives.Feature importance analysis was used to identify the most influential transaction signals driving fraud predictions.The trained model was serialised using Joblib and deployed as a REST API microservice using FastAPI, enabling real-time transaction scoring.

Features
Real-time fraud probability prediction
Decision engine returning:
ALLOW – transaction is likely legitimate
CHALLENGE – additional verification required
BLOCK – high likelihood of fraud
Input validation using Pydantic
Feature engineering and categorical encoding
Simulation endpoint for testing transactions

Tech Stack

Backend: FastAPI
Machine Learning: XGBoost
Data Processing: pandas, NumPy
Model Serialization: Joblib
Python Version: ≥3.10

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api
