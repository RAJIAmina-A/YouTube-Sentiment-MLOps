const API_URL = "http://127.0.0.1:8000";

let analysisResults = [];

const status = document.getElementById("status");
const resultsDiv = document.getElementById("results");

document.getElementById("analyzeButton")
.addEventListener("click", async () => {
    try {
        status.textContent = "Extraction des commentaires...";

        const [tab] = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        if (!tab.url.includes("youtube.com")) {
            throw new Error("Ouvre une vidéo YouTube.");
        }

        const response = await chrome.tabs.sendMessage(
            tab.id,
            { action: "extractComments" }
        );

        const comments = response.comments;

        if (!comments.length) {
            throw new Error("Aucun commentaire visible.");
        }

        status.textContent = "Analyse en cours...";

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
            throw new Error("Erreur de l'API.");
        }

        const data = await apiResponse.json();

        analysisResults = data.results;

        document.getElementById("positive").textContent =
            data.percentages.positive + "%";

        document.getElementById("neutral").textContent =
            data.percentages.neutral + "%";

        document.getElementById("negative").textContent =
            data.percentages.negative + "%";

        displayResults();

        status.textContent =
            `${data.total_comments} commentaires analysés.`;

    } catch (error) {
        status.textContent = error.message;
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
        sentiment.textContent = item.sentiment;

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
});

document.getElementById("copyButton")
.addEventListener("click", async () => {
    const text = analysisResults.map(
        item => `${item.sentiment}: ${item.comment}`
    ).join("\n");

    await navigator.clipboard.writeText(text);

    status.textContent = "Résultats copiés.";
});