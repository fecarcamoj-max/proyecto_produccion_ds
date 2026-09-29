"""API HTTP para consultar el modelo guardado sin volver a entrenarlo."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path(__file__).parent / "artifacts" / "model_churn.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró {MODEL_PATH}. Ejecuta primero: py train_model.py")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="API modelo de fuga de clientes")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir al dominio del sitio.
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Cliente(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Contract: str
    InternetService: str
    TechSupport: str
    PaperlessBilling: str
    PaymentMethod: str
    Partner: str
    Dependents: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(cliente: Cliente):
    try:
        row = pd.DataFrame([cliente.model_dump()])
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        yes_probability = float(probabilities[classes.index("Yes")])
        prediction = str(model.predict(row)[0])
        return {
            "prediction": prediction,
            "churn_probability": yes_probability,
            "risk": "alto" if yes_probability >= 0.5 else "bajo",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
