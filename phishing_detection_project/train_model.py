"""Train and save the URL phishing detection demo pipeline."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_extraction import features_for_urls

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "phishing_urls.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "phishing_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def main():
    data = pd.read_csv(DATA_PATH)
    required = {"url", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain columns named 'url' and 'label'.")
    data = data.dropna(subset=["url", "label"]).drop_duplicates(subset=["url"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    allowed = {"phishing", "legitimate"}
    if not set(data["label"].unique()).issubset(allowed):
        raise ValueError("Labels must be 'phishing' or 'legitimate'.")
    if data["label"].nunique() != 2 or len(data) < 10:
        raise ValueError("Provide at least 10 rows and both label classes.")

    X = features_for_urls(data["url"].astype(str).tolist())
    y = data["label"].map({"legitimate": 0, "phishing": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    k = min(10, X_train.shape[1])
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "rows_used": int(len(data)),
        "features_available": int(X.shape[1]),
        "features_selected": int(k),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "confusion_matrix_labels": ["legitimate", "phishing"],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0, 1], target_names=["legitimate", "phishing"], zero_division=0, output_dict=True),
        "note": "Metrics are for the included illustrative dataset only; they do not represent real-world performance."
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": list(X.columns)}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))

if __name__ == "__main__":
    main()
