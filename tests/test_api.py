import os
import time
import pytest
import requests

BASE_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")


def post(comments):
    return requests.post(f"{BASE_URL}/predict_batch", json={"comments": comments}, timeout=30)


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    print("\n/health ->", r.json())


@pytest.mark.parametrize("taille", [1, 10, 50, 100])
def test_batch_tailles(taille):
    comments = ["I love this video", "This is terrible", "It is okay"] * (taille // 3 + 1)
    comments = comments[:taille]
    debut = time.perf_counter()
    r = post(comments)
    duree_ms = (time.perf_counter() - debut) * 1000
    print(f"\nBatch {taille} : {duree_ms:.0f} ms")
    assert r.status_code == 200
    assert duree_ms < 3000


def test_batch_vide():
    r = post([])
    assert r.status_code in (400, 422)


@pytest.mark.parametrize("payload", [
    {"comments": "pas une liste"},
    {"mauvaise_cle": ["abc"]},
    {"comments": [1, 2, 3]},
    {},
])
def test_donnees_invalides(payload):
    r = requests.post(f"{BASE_URL}/predict_batch", json=payload, timeout=10)
    assert r.status_code in (400, 422)