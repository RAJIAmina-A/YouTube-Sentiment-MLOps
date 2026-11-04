function extractComments() {
    const elements = document.querySelectorAll(
        "ytd-comment-thread-renderer #content-text"
    );

    return Array.from(elements)
        .map(element => element.innerText.trim())
        .filter(text => text.length > 0)
        .slice(0, 50);
}

chrome.runtime.onMessage.addListener(
    (request, sender, sendResponse) => {
        if (request.action === "extractComments") {
            sendResponse({
                comments: extractComments()
            });
        }
        return true;
    }
);