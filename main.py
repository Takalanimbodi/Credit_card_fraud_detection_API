from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import joblib
import random
import pandas as pd
import traceback

app = FastAPI(
    title="Fraud Detection Microservice",
    description="Real-time transaction fraud scoring API",
    version="1.0.0",
    docs_url="/docs",  # Publicity for /simulate
    redoc_url=None
)

model = joblib.load("model/xgboost_fraud_model.pkl")


MERCHANT_CATEGORIES = [
    "Electronics",
    "Food",
    "Travel",
    "Grocery",
    "Clothing"
]

FEATURE_ORDER = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
    "merchant_category_Clothing",
    "merchant_category_Electronics",
    "merchant_category_Food",
    "merchant_category_Grocery",
    "merchant_category_Travel"
]


# Input schema (API layer)

class TransactionEvent(BaseModel):
    amount: float
    merchant_category: str
    timestamp: str
    location_mismatch: int
    foreign_transaction: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int


# Feature engineering

def encode_merchant_category(category: str) -> dict:
    encoding = {f"merchant_category_{cat}": 0 for cat in MERCHANT_CATEGORIES}
    key = f"merchant_category_{category}"

    if key not in encoding:
        raise ValueError("Invalid merchant category")

    encoding[key] = 1
    return encoding


def build_feature_vector(tx: TransactionEvent) -> pd.DataFrame:
    try:
        transaction_hour = datetime.fromisoformat(
            tx.timestamp.replace("Z", "")
        ).hour
    except ValueError:
        raise ValueError("Invalid timestamp format")

    merchant_ohe = encode_merchant_category(tx.merchant_category)

    features = {
        "amount": tx.amount,
        "transaction_hour": transaction_hour,
        "foreign_transaction": tx.foreign_transaction,
        "location_mismatch": tx.location_mismatch,
        "device_trust_score": tx.device_trust_score,
        "velocity_last_24h": tx.velocity_last_24h,
        "cardholder_age": tx.cardholder_age,
        **merchant_ohe
    }

    # Enforce training feature order
    return pd.DataFrame([[features[col] for col in FEATURE_ORDER]],
                        columns=FEATURE_ORDER)


def fraud_decision(probability: float) -> str:
    if probability >= 0.8:
        return "BLOCK"
    elif probability >= 0.5:
        return "CHALLENGE"
    else:
        return "ALLOW"


def run_prediction(tx: TransactionEvent):
    X = build_feature_vector(tx)
    fraud_prob = float(model.predict_proba(X)[0][1])
    decision = fraud_decision(fraud_prob)
    return fraud_prob, decision


# Fake transaction generator

def generate_fake_transaction() -> TransactionEvent:
    is_fraud_scenario = random.random() < 0.1  # only 10% suspicious

    return TransactionEvent(
        amount=round(random.uniform(10, 500), 2) if not is_fraud_scenario 
               else round(random.uniform(1000, 5000), 2),
        merchant_category=random.choice(MERCHANT_CATEGORIES),
        timestamp=datetime.now().isoformat(),
        location_mismatch=random.choices([0, 1], weights=[90, 10])[0] if not is_fraud_scenario
                          else random.choices([0, 1], weights=[20, 80])[0],
        foreign_transaction=random.choices([0, 1], weights=[85, 15])[0] if not is_fraud_scenario
                            else random.choices([0, 1], weights=[30, 70])[0],
        device_trust_score=round(random.uniform(0.6, 1.0), 2) if not is_fraud_scenario
                           else round(random.uniform(0.0, 0.4), 2),
        velocity_last_24h=random.randint(0, 3) if not is_fraud_scenario
                          else random.randint(8, 15),
        cardholder_age=random.randint(18, 75)
    )
# Endpoints


@app.get("/simulate")
def simulate():
    try:
        tx = generate_fake_transaction()
        fraud_prob, decision = run_prediction(tx)

        return {
            "transaction": tx,
            "fraud_probability": round(fraud_prob, 4),
            "decision": decision
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Internal-only endpoint

@app.post("/predict", include_in_schema=False)
def predict_fraud(tx: TransactionEvent):
    """
    INTERNAL-ONLY: Call this endpoint from backend systems/apps.
    Returns fraud probability and decision.
    """
    try:
        fraud_prob, decision = run_prediction(tx)
        return {
            "transaction": tx,
            "fraud_probability": round(fraud_prob,4),
            "decision": decision
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed")
