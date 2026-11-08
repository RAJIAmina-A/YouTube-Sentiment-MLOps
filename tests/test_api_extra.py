import os, statistics, time
import pytest, requests

BASE_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")
LABELS = {-1: "Négatif", 0: "Neutre", 1: "Positif"}


def post(comments):
    return requests.post(f"{BASE_URL}/predict_batch", json={"comments": comments}, timeout=30)


def test_health_modele_charge():
    j = requests.get(f"{BASE_URL}/health", timeout=10).json()
    assert j["status"] == "ok" and j["model_loaded"] is True


def test_structure_reponse():
    comments = ["I love this video", "This is terrible", "It is okay", "amazing work", "worst ever"]
    r = post(comments)
    assert r.status_code == 200
    j = r.json()
    assert j["total_comments"] == len(comments)
    assert len(j["results"]) == len(comments)
    for item in j["results"]:
        assert item["label"] in (-1, 0, 1)
        assert item["sentiment"] == LABELS[item["label"]]
        assert 0 < item["confidence"] <= 1
    assert sum(j["distribution"].values()) == len(comments)
    assert abs(sum(j["percentages"].values()) - 100) < 0.1


def test_ordre_conserve():
    comments = [f"comment number {i} great" for i in range(20)]
    j = post(comments).json()
    assert [x["comment"] for x in j["results"]] == comments


def test_limite_101_commentaires():
    assert post(["nice"] * 101).status_code == 422


@pytest.mark.parametrize("comments", [[""], ["   "], ["ok", ""], ["ok", "   "]])
def test_commentaire_vide(comments):
    assert post(comments).status_code in (400, 422)


def test_non_json():
    r = requests.post(f"{BASE_URL}/predict_batch", data="hello",
                      headers={"Content-Type": "text/plain"}, timeout=10)
    assert r.status_code in (400, 415, 422)


def test_cors_preflight():
    r = requests.options(f"{BASE_URL}/predict_batch", headers={
        "Origin": "chrome-extension://test",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"}, timeout=10)
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


CAS = {
    "très court": "ok",
    "très long": "great " * 2000,
    "emojis seuls": "😍😍😍🔥🔥",
    "emojis + texte": "I love this 😍😍",
    "arabe": "مباراة رائعة جدا",
    "darija": "wakha had lvideo zwina bzaf",
    "mélange FR/AR/EN": "Super vidéo ! مباراة رائعة, very good",
    "caractères spéciaux": "!!!??? ###",
    "URL seule": "https://youtu.be/xyz",
    "mention": "@user thanks!!!",
}


def test_cas_limites_via_api():
    lignes = []
    for nom, texte in CAS.items():
        r = post([texte])
        assert r.status_code in (200, 400, 422), f"{nom}: {r.status_code}"
        if r.status_code == 200:
            it = r.json()["results"][0]
            lignes.append(f"{nom:<22} -> 200 {it['sentiment']:<8} conf={it['confidence']}")
        else:
            lignes.append(f"{nom:<22} -> {r.status_code}")
    print("\n" + "\n".join(lignes))


def test_temps_batch_50():
    comments = ["this video is really great and I learned a lot"] * 50
    post(comments)  # préchauffage
    serveur, total = [], []
    for _ in range(20):
        t = time.perf_counter()
        j = post(comments).json()
        total.append((time.perf_counter() - t) * 1000)
        serveur.append(j["processing_time_ms"])
    print(f"\nBatch 50 : serveur médiane={statistics.median(serveur):.1f} ms | "
          f"total médiane={statistics.median(total):.0f} ms, max={max(total):.0f} ms")
    assert statistics.median(serveur) < 100