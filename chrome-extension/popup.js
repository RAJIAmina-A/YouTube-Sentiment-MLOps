const API_URL = "https://refactored-trout-xrw47q6r5pvj2qgx-7860.app.github.dev"
    .replace(/\/+$/, "");

let analysisResults = [];

const status = document.getElementById("status");
const resultsDiv = document.getElementById("results");
const analyzeButton = document.getElementById("analyzeButton");

// Mode sombre mémorisé
try {
    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark");
    }
} catch (e) {}

analyzeButton.addEventListener("click", async () => {
    analyzeButton.disabled = true;

    try {
        status.textContent = "Extraction des commentaires...";

        const [tab] = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        if (!tab.url || !tab.url.includes("youtube.com")) {
            throw new Error("Ouvre une vidéo YouTube.");
        }

        const response = await chrome.tabs.sendMessage(
            tab.id,
            { action: "extractComments" }
        );

        const comments = response.comments;

        if (!comments.length) {
            throw new Error("Aucun commentaire visible. Descends dans la page pour les charger.");
        }

        status.textContent = `Analyse de ${comments.length} commentaires en cours...`;

        const apiResponse = await fetch(
            `${API_URL}/predict_batch`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ comments })
            }
        );

        if (!apiResponse.ok) {
            throw new Error(`Erreur de l'API (code ${apiResponse.status}).`);
        }

        const data = await apiResponse.json();

        analysisResults = data.results;

        document.getElementById("positive").textContent =
            `${data.percentages.positive}% (${data.distribution.positive})`;

        document.getElementById("neutral").textContent =
            `${data.percentages.neutral}% (${data.distribution.neutral})`;

        document.getElementById("negative").textContent =
            `${data.percentages.negative}% (${data.distribution.negative})`;

        displayResults();

        status.textContent =
            `${data.total_comments} commentaires analysés en ${data.processing_time_ms} ms.`;

    } catch (error) {
        const msg = error.message || "";

        if (msg.includes("Failed to fetch")) {
            status.textContent =
                "API injoignable. Vérifie que le Codespace est allumé et le port 7860 en Public.";
        } else if (
            msg.includes("Receiving end does not exist") ||
            msg.includes("Could not establish connection")
        ) {
            status.textContent =
                "Recharge la page YouTube (Ctrl+R), puis réessaie.";
        } else {
            status.textContent = msg;
        }
    } finally {
        analyzeButton.disabled = false;
    }
});

function displayResults() {
    const filter = document.getElementById("filter").value;

    resultsDiv.innerHTML = "";

    const filtered = analysisResults.filter(item =>
        filter === "all" || String(item.label) === filter
    );

    filtered.forEach(item => {
        const div = document.createElement("div");
        div.className = "comment";

        const p = document.createElement("p");
        p.textContent = item.comment;

        const sentiment = document.createElement("strong");
        sentiment.textContent =
            `${item.sentiment} (${Math.round(item.confidence * 100)} %)`;

        div.appendChild(p);
        div.appendChild(sentiment);

        resultsDiv.appendChild(div);
    });
}

document.getElementById("filter")
.addEventListener("change", displayResults);

document.getElementById("themeButton")
.addEventListener("click", () => {
    document.body.classList.toggle("dark");

    try {
        localStorage.setItem(
            "theme",
            document.body.classList.contains("dark") ? "dark" : "light"
        );
    } catch (e) {}
});

document.getElementById("copyButton")
.addEventListener("click", async () => {
    const text = analysisResults.map(
        item => `${item.sentiment} (${Math.round(item.confidence * 100)} %): ${item.comment}`
    ).join("\n");

    await navigator.clipboard.writeText(text);

    status.textContent = "Résultats copiés.";
});