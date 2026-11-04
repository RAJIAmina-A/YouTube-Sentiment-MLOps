
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "reddit.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

LABEL_MAP = {
    -1: -1,
    0: 0,
    1: 1,
    "-1": -1,
    "0": 0,
    "1": 1
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    print("Colonnes du dataset :", df.columns.tolist())

    # Dataset Reddit courant : commentaire + catégorie
    text_candidates = [
        "clean_comment", "comment", "text", "clean_text",
        "content", "sentiment_text"
    ]
    label_candidates = [
        "category", "label", "sentiment", "target"
    ]

    text_col = next(
        (c for c in text_candidates if c in df.columns), None
    )
    label_col = next(
        (c for c in label_candidates if c in df.columns), None
    )

    # Si les noms de colonnes sont inconnus, utilise
    # la première colonne comme texte et la dernière comme label.
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    # Garder uniquement les labels prévus dans le TP
    df = df[df["label"].isin([-1, 0, 1])]

    print("\nDataset préparé :", df.shape)
    print("\nDistribution des classes :")
    print(df["label"].value_counts().sort_index())

    # EDA : distribution des classes
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label")
    plt.title("Distribution des sentiments")
    plt.savefig(PROCESSED_DIR / "class_distribution.png")
    plt.close()

    # EDA : longueur des commentaires
    df["text_length"] = df["text"].str.split().str.len()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["text_length"], bins=50)
    plt.title("Longueur des commentaires")
    plt.xlabel("Nombre de mots")
    plt.savefig(PROCESSED_DIR / "text_length.png")
    plt.close()

    # Vérification des classes
    counts = df["label"].value_counts()
    print("\nNombre d'exemples par classe :")
    print(counts)

    if len(counts) < 3:
        raise ValueError("Le dataset ne contient pas les 3 classes.")

    if counts.min() < 300:
        print("Attention : une classe contient moins de 300 exemples.")

    df = df[["text", "label"]]

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    df.to_csv(PROCESSED_DIR / "cleaned_data.csv", index=False)

    print("\nTrain :", train_df.shape)
    print("Test :", test_df.shape)
    print("Fichiers sauvegardés dans data/processed")

if __name__ == "__main__":
    main()
