import time
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

MODEL_PATH = "models/sentiment_pipeline.joblib"
TEST_PATH = "data/processed/test.csv"  # à adapter au nom de ton fichier de test


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])


def test_accuracy(model, test_df):
    preds = model.predict(test_df["text"])
    acc = accuracy_score(test_df["label"], preds)
    print("\n", classification_report(test_df["label"], preds))
    print("Matrice de confusion :\n", confusion_matrix(test_df["label"], preds))
    assert acc >= 0.80, f"Accuracy {acc:.3f} < 0.80"


def test_f1_par_classe(model, test_df):
    preds = model.predict(test_df["text"])
    scores = f1_score(test_df["label"], preds, average=None)
    for s in scores:
        assert s > 0.75, f"F1 trop bas : {scores}"


@pytest.mark.parametrize("texte", [
    "",                                   # vide
    "ok",                                 # très court
    "great " * 2000,                      # très long
    "😍😍😍🔥🔥",                          # emojis seuls
    "Super vidéo ! مباراة رائعة, très bien", # langues mélangées
    "!!!??? ###",                         # caractères spéciaux
])
def test_cas_limites(model, texte):
    pred = model.predict([texte])[0]
    assert pred in (-1, 0, 1)


def test_temps_inference_50(model, test_df):
    batch = test_df["text"].head(50).tolist()
    debut = time.perf_counter()
    model.predict(batch)
    duree_ms = (time.perf_counter() - debut) * 1000
    print(f"\nInférence 50 commentaires : {duree_ms:.1f} ms")
    assert duree_ms < 100