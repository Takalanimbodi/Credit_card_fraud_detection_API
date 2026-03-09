# Fraud Detection Microservice

A **real-time transaction fraud scoring API** built with **FastAPI** and **XGBoost**.  
This service evaluates financial transactions and provides a fraud probability along with a recommended action (ALLOW, CHALLENGE, BLOCK).




## Features

- Predict fraud probability of financial transactions in real-time.
- Outputs actionable decisions:
  - **ALLOW**: Transaction seems safe.
  - **CHALLENGE**: Transaction is suspicious, requires verification.
  - **BLOCK**: High likelihood of fraud, block transaction.
- Supports internal and simulation endpoints for testing.
- Input validation using **Pydantic**.
- Feature engineering and one-hot encoding for merchant categories.

---

## Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: XGBoost (joblib model)
- **Data Processing**: pandas, numpy
- **Python Version**: >=3.10

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api