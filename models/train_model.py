
from pathlib import Path
import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"]
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    params = {
        "classifier__C": [0.5, 1.0, 2.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)]
    }

    search = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    print("Entraînement et optimisation...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Meilleurs paramètres :", search.best_params_)

    # Évaluation
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")

    print("\nAccuracy :", accuracy)
    print("F1 macro :", f1_macro)
    print("\nRapport de classification :")
    report = classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=["Négatif", "Neutre", "Positif"],
        output_dict=True,
        zero_division=0
    )
    print(classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=["Négatif", "Neutre", "Positif"],
        zero_division=0
    ))

    # Matrice de confusion
    cm = confusion_matrix(y_test, predictions, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Négatif", "Neutre", "Positif"]
    )
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion")
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.close()

    # Sauvegarde
    joblib.dump(best_model, MODEL_DIR / "sentiment_pipeline.joblib")

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "best_params": search.best_params_,
        "classification_report": report
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # Mesure inférence sur 50 commentaires
    batch = X_test.iloc[:50].tolist()
    start = time.perf_counter()
    best_model.predict(batch)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nInférence pour 50 commentaires : {elapsed_ms:.2f} ms")
    print("Modèle sauvegardé dans models/sentiment_pipeline.joblib")

if __name__ == "__main__":
    main()
