document.addEventListener("DOMContentLoaded", () => {
    const inputText = document.getElementById("inputText");
    const outputText = document.getElementById("outputText");
    const attackMode = document.getElementById("attackMode");
    const enableThinking = document.getElementById("enableThinking");
    const runBtn = document.getElementById("runBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const sampleBtn = document.getElementById("sampleBtn");
    const copyBtn = document.getElementById("copyBtn");

    const preGVal = document.getElementById("preGVal");
    const preBadge = document.getElementById("preBadge");
    const postGVal = document.getElementById("postGVal");
    const postBadge = document.getElementById("postBadge");
    const progressBar = document.getElementById("progressBar");
    const reductionPct = document.getElementById("reductionPct");
    const timingBadge = document.getElementById("timingBadge");
    const sanitizedBadge = document.getElementById("sanitizedBadge");

    const SAMPLE_TEXT = `Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing. The signal resides within n-gram hash patterns across token sequences and survives simple editing techniques. This sample text illustrates how the statistical excess of green-list tokens enables watermark detectors to identify synthetic text with high confidence.`;

    sampleBtn.addEventListener("click", () => {
        inputText.value = SAMPLE_TEXT;
    });

    copyBtn.addEventListener("click", () => {
        if (!outputText.value) return;
        navigator.clipboard.writeText(outputText.value).then(() => {
            const orig = copyBtn.innerText;
            copyBtn.innerText = "Copied!";
            setTimeout(() => copyBtn.innerText = orig, 2000);
        });
    });

    runBtn.addEventListener("click", async () => {
        const text = inputText.value.trim();
        if (!text) {
            alert("Please paste some text first!");
            return;
        }

        // Loading UI state
        runBtn.disabled = true;
        btnText.innerText = "Processing Pipeline...";
        btnSpinner.classList.remove("hidden");

        try {
            const response = await fetch("/remove-watermark", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    text: text,
                    attack_mode: attackMode.value,
                    enable_thinking: enableThinking.checked,
                    substitution_rate: 0.15
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server error during attack execution");
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            runBtn.disabled = false;
            btnText.innerText = "Remove Watermark";
            btnSpinner.classList.add("hidden");
        }
    });

    function renderResults(data) {
        // Output clean text
        outputText.value = data.clean_text;
        copyBtn.disabled = !data.clean_text;

        // Pre-Attack Score
        const preG = data.pre_attack.g_value;
        preGVal.innerText = preG.toFixed(2);
        if (data.pre_attack.is_watermarked) {
            preBadge.className = "status-badge red";
            preBadge.innerText = "Watermarked";
        } else {
            preBadge.className = "status-badge green";
            preBadge.innerText = "Unwatermarked";
        }

        // Post-Attack Score
        const postG = data.post_attack.g_value;
        postGVal.innerText = postG.toFixed(2);
        if (data.post_attack.is_watermarked) {
            postBadge.className = "status-badge red";
            postBadge.innerText = "Watermarked";
        } else {
            postBadge.className = "status-badge green";
            postBadge.innerText = "Clean";
        }

        // Reduction & Badges
        const redPct = data.watermark_reduction_pct;
        reductionPct.innerText = `${redPct.toFixed(1)}%`;
        progressBar.style.width = `${Math.min(Math.max(redPct, 5), 100)}%`;

        timingBadge.innerText = `Processing Time: ${data.processing_time_ms} ms`;
        sanitizedBadge.innerText = `Unicode Removed: ${data.sanitized_char_count} chars`;
    }
});
