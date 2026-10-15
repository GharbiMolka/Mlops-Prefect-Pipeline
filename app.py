from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd

from model_pipeline import prepare_data

MODEL_PATH = "classifier.joblib"
DATA_PATH = "Churn_Modelling.csv"

# Ordre des colonnes après les drop dans prepare_data()
FEATURE_ORDER = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

app = FastAPI(title="Churn Prediction API")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH} ({model.n_features_in_} features)")
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle '{MODEL_PATH}'.") from e

# Recrée exactement le même scaler qu'à l'entraînement (random_state=1 par défaut)
_, _, _, _, scaler = prepare_data(DATA_PATH)
print("Scaler recréé")


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        }
    )

    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]  # accepté, mais non utilisé par le modèle
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        data = features.model_dump()
        data["Gender"] = 1 if data["Gender"] == "Male" else 0  # LabelEncoder: Female=0, Male=1
        df = pd.DataFrame([data])[FEATURE_ORDER]
        X = scaler.transform(df.values)
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")