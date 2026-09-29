"""Entrena y guarda el mejor modelo de churn del notebook educativo.

Uso: py train_model.py --csv clientes_churn.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "TechSupport", "PaperlessBilling",
    "PaymentMethod", "Partner", "Dependents",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


def build_pipeline(classifier) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("clientes_churn.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"No existe el CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise SystemExit(f"Faltan columnas requeridas: {', '.join(missing)}")
    df = df.dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET].astype(str)
    if set(y.unique()) != {"No", "Yes"}:
        raise SystemExit("La columna Churn debe contener las etiquetas 'No' y 'Yes'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y
    )
    candidates = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=0)
        ),
        "decision_tree": build_pipeline(
            DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=20, random_state=0)
        ),
    }
    results = {}
    best_name, best_auc, best_model = None, -1.0, None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        yes_index = list(model.classes_).index("Yes")
        probabilities = model.predict_proba(X_test)[:, yes_index]
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, pos_label="Yes", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, pos_label="Yes", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, pos_label="Yes", zero_division=0)),
            "roc_auc": float(roc_auc_score((y_test == "Yes").astype(int), probabilities)),
        }
        results[name] = metrics
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        if metrics["roc_auc"] > best_auc:
            best_name, best_auc, best_model = name, metrics["roc_auc"], model

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model_churn.joblib"
    joblib.dump(best_model, model_path)
    metadata = {
        "selected_model": best_name,
        "positive_class": "Yes",
        "features": FEATURES,
        "metrics": results,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelo seleccionado por ROC-AUC: {best_name}")
    print(f"Modelo guardado en: {model_path.resolve()}")


if __name__ == "__main__":
    main()
