import statistics, time
from pathlib import Path
import joblib, pandas as pd, pytest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "models/sentiment_pipeline.joblib"
TEST_PATH = "data/processed/test.csv"
LABELS = {-1: "Négatif", 0: "Neutre", 1: "Positif"}

CAS = {
    "vide": "", "espaces": "     ", "très court": "ok", "un mot": "love",
    "très long": "great " * 2000, "emojis seuls": "😍😍😍🔥🔥",
    "emojis + texte": "I love this 😍😍", "arabe": "مباراة رائعة جدا",
    "darija": "wakha had lvideo zwina bzaf", "français": "Cette vidéo est vraiment nulle",
    "mélange": "Super vidéo ! مباراة رائعة, très bien", "chiffres": "12345 67890",
    "caractères spéciaux": "!!!??? ###", "URL seule": "https://youtu.be/xyz",
}


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])


def test_cas_limites_detail(model):
    lignes = []
    for nom, texte in CAS.items():
        proba = model.predict_proba([texte])[0]
        i = proba.argmax()
        lignes.append(f"{nom:<22} -> {LABELS[int(model.classes_[i])]:<8} conf={proba[i]:.2f}")
    print("\n" + "\n".join(lignes))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cas_limites.txt").write_text("\n".join(lignes), encoding="utf-8")


def test_rapport_complet(model, test_df):
    preds = model.predict(test_df["text"])
    rapport = (f"Accuracy : {accuracy_score(test_df['label'], preds):.4f}\n\n"
               f"{classification_report(test_df['label'], preds, digits=3)}\n"
               f"Matrice de confusion (lignes = réel, colonnes = prédit, ordre -1/0/1) :\n"
               f"{confusion_matrix(test_df['label'], preds, labels=[-1, 0, 1])}")
    print("\n" + rapport)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_report.txt").write_text(rapport, encoding="utf-8")


def test_temps_inference_mediane(model, test_df):
    batch = test_df["text"].head(50).tolist()
    model.predict(batch)  # préchauffage
    temps = []
    for _ in range(100):
        t = time.perf_counter()
        model.predict(batch)
        temps.append((time.perf_counter() - t) * 1000)
    med = statistics.median(temps)
    print(f"\nBatch 50 : médiane={med:.1f} ms, moyenne={statistics.mean(temps):.1f} ms, "
          f"p95={sorted(temps)[94]:.1f} ms")
    assert med < 100