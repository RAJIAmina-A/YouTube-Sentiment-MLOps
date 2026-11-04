
from pathlib import Path
import requests
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/"
    "Himanshu-1703/reddit-sentiment-analysis/"
    "refs/heads/main/data/reddit.csv"
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "reddit.csv"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Téléchargement du dataset...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    RAW_FILE.write_bytes(response.content)
    print(f"Dataset téléchargé : {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    print("\nDimensions :", df.shape)
    print("\nColonnes originales :", df.columns.tolist())
    print("\nAperçu :")
    print(df.head())

    print("\nValeurs manquantes :")
    print(df.isnull().sum())

    print("\nDistribution des labels :")
    print(df.iloc[:, -1].value_counts(dropna=False))

if __name__ == "__main__":
    main()
