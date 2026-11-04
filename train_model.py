"""Train a reproducible Random Forest model on clearly labelled synthetic demo data."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "demo_training_dataset.csv"
MODEL_PATH = ROOT / "postharvest_risk_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
FEATURES = ["temperature_c", "relative_humidity_pct", "soil_moisture_pct"]
CLASSES = ["NORMAL", "WARNING", "HIGH RISK"]
RANDOM_STATE = 42

def make_demo_dataset(n_samples=3000, seed=42):
    """Create synthetic environmental examples; labels are rule-generated, not field observations."""
    rng = np.random.default_rng(seed)
    temp = rng.uniform(10, 42, n_samples)
    humidity = rng.uniform(20, 100, n_samples)
    soil = rng.uniform(0, 100, n_samples)

    # Demonstration label logic only. Soil moisture is a contextual field/environment
    # indicator, not a direct measurement of moisture in harvested produce.
    heat_humidity = (temp >= 32) & (humidity >= 75)
    very_humid = humidity >= 90
    hot = temp >= 36
    dry_hot = (temp >= 34) & (humidity < 35)
    saturated_soil = soil >= 88
    moderate = ((temp >= 28) & (humidity >= 65)) | ((humidity >= 82) & (temp >= 25)) | ((soil >= 75) & (humidity >= 70))
    high = heat_humidity | very_humid | hot | dry_hot | (saturated_soil & (humidity >= 75))
    labels = np.where(high, "HIGH RISK", np.where(moderate, "WARNING", "NORMAL"))

    # Small controlled label noise makes the synthetic task less perfectly separable.
    flip = rng.random(n_samples) < 0.025
    for i in np.flatnonzero(flip):
        choices = [c for c in CLASSES if c != labels[i]]
        labels[i] = rng.choice(choices)
    return pd.DataFrame({
        "temperature_c": np.round(temp, 2),
        "relative_humidity_pct": np.round(humidity, 2),
        "soil_moisture_pct": np.round(soil, 2),
        "risk_category": labels,
        "data_source": "SYNTHETIC_DEMONSTRATION_DATA"
    })

def main():
    df = make_demo_dataset()
    df.to_csv(DATA_PATH, index=False)
    X = df[FEATURES]
    y = df["risk_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    labels_order = [c for c in CLASSES if c in set(y)]
    metrics = {
        "data_notice": "SYNTHETIC DEMONSTRATION DATA. Metrics do not represent real-world performance.",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "class_labels": labels_order,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels_order).tolist(),
        "classification_report": classification_report(y_test, predictions, labels=labels_order, output_dict=True, zero_division=0),
        "test_class_counts": y_test.value_counts().to_dict()
    }
    joblib.dump({"model": model, "features": FEATURES, "classes": labels_order,
                 "model_version": "1.0-demo", "data_notice": metrics["data_notice"]}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH.name}, {DATA_PATH.name}, {METRICS_PATH.name}")
    print(f"Synthetic holdout accuracy: {metrics['accuracy']:.3f} (not a real-world claim)")

if __name__ == "__main__":
    main()
