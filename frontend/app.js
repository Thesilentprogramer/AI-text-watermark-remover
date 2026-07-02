document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Workspace
    const inputText = document.getElementById("inputText");
    const outputText = document.getElementById("outputText");
    const diffViewContainer = document.getElementById("diffViewContainer");
    const diffHtmlOutput = document.getElementById("diffHtmlOutput");
    const attackMode = document.getElementById("attackMode");
    const enableThinking = document.getElementById("enableThinking");

    const runBtn = document.getElementById("runBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const sampleWmBtn = document.getElementById("sampleWmBtn");
    const sampleCleanBtn = document.getElementById("sampleCleanBtn");
    const copyBtn = document.getElementById("copyBtn");
    const viewCleanBtn = document.getElementById("viewCleanBtn");
    const viewDiffBtn = document.getElementById("viewDiffBtn");

    const verdictBanner = document.getElementById("verdictBanner");
    const verdictText = document.getElementById("verdictText");
    const preGVal = document.getElementById("preGVal");
    const preBadge = document.getElementById("preBadge");
    const postGVal = document.getElementById("postGVal");
    const postBadge = document.getElementById("postBadge");
    
    const timingBadge = document.getElementById("timingBadge");
    const reductionBadge = document.getElementById("reductionBadge");

    const transformationSection = document.getElementById("transformationSection");
    const timelineTabs = document.getElementById("timelineTabs");
    const stepTitle = document.getElementById("stepTitle");
    const stepDesc = document.getElementById("stepDesc");
    const stepTextOutput = document.getElementById("stepTextOutput");
    const stepLogsList = document.getElementById("stepLogsList");

    if (!inputText) return; // Not on the app page

    let currentIntermediateSteps = [];
    let currentDiffHtml = "";

    const WATERMARKED_SAMPLE = "Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing. The signal resides within n-gram hash patterns across token sequences and survives simple editing techniques.";
    const CLEAN_SAMPLE = "Natural human writing features highly variable sentence lengths, creative metaphors, and unconstrained vocabulary distributions. Because no algorithmic logit biasing or n-gram hashing was applied during its creation, standard statistical watermark detectors will register baseline noise levels.";

    sampleWmBtn.addEventListener("click", () => inputText.value = WATERMARKED_SAMPLE);
    sampleCleanBtn.addEventListener("click", () => inputText.value = CLEAN_SAMPLE);

    viewCleanBtn.addEventListener("click", () => {
        viewCleanBtn.classList.add("active");
        viewDiffBtn.classList.remove("active");
        outputText.classList.remove("hidden");
        diffViewContainer.classList.add("hidden");
    });

    viewDiffBtn.addEventListener("click", () => {
        viewDiffBtn.classList.add("active");
        viewCleanBtn.classList.remove("active");
        outputText.classList.add("hidden");
        diffViewContainer.classList.remove("hidden");
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

        runBtn.disabled = true;
        btnText.innerText = "Executing Pipeline";
        btnSpinner.classList.remove("hidden");
        verdictBanner.className = "neo-banner";
        verdictText.innerText = "Processing...";

        try {
            const response = await fetch("/remove-watermark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    attack_mode: attackMode.value,
                    enable_thinking: enableThinking.checked,
                    substitution_rate: 0.15
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server error");
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            alert(`Error: ${err.message}`);
            verdictBanner.className = "neo-banner red";
            verdictText.innerText = `Error: ${err.message}`;
        } finally {
            runBtn.disabled = false;
            btnText.innerText = "Remove Watermark";
            btnSpinner.classList.add("hidden");
        }
    });

    function renderResults(data) {
        outputText.value = data.clean_text;
        currentDiffHtml = data.diff_html || data.clean_text;
        diffHtmlOutput.innerHTML = currentDiffHtml;
        copyBtn.disabled = !data.clean_text;

        verdictText.innerText = data.verdict_title;
        if (data.is_clean) {
            verdictBanner.className = "neo-banner green";
        } else if (data.watermark_reduction_pct > 15.0) {
            verdictBanner.className = "neo-banner yellow";
        } else {
            verdictBanner.className = "neo-banner red";
        }

        preGVal.innerText = data.pre_attack.g_value.toFixed(2);
        preBadge.className = data.pre_attack.is_watermarked ? "neo-badge red" : "neo-badge green";
        preBadge.innerText = data.pre_attack.is_watermarked ? "Watermarked" : "Clean";

        postGVal.innerText = data.post_attack.g_value.toFixed(2);
        postBadge.className = data.post_attack.is_watermarked ? "neo-badge red" : "neo-badge green";
        postBadge.innerText = data.post_attack.is_watermarked ? "Watermarked" : "Clean";

        timingBadge.innerText = `Runtime: ${data.processing_time_ms}ms`;
        reductionBadge.innerText = `Reduction: ${data.watermark_reduction_pct.toFixed(1)}%`;

        stepLogsList.innerHTML = "";
        if (data.step_logs && data.step_logs.length > 0) {
            data.step_logs.forEach(log => {
                const li = document.createElement("li");
                li.innerText = `> ${log}`;
                stepLogsList.appendChild(li);
            });
        }

        currentIntermediateSteps = data.intermediate_steps || [];
        if (currentIntermediateSteps.length > 0) {
            transformationSection.classList.remove("hidden");
            selectTimelineStep(1);
        }
    }

    if (timelineTabs) {
        timelineTabs.addEventListener("click", (e) => {
            if (e.target.tagName !== "BUTTON") return;
            const stepNum = parseInt(e.target.dataset.step, 10);
            selectTimelineStep(stepNum);
        });
    }

    function selectTimelineStep(stepNum) {
        const buttons = timelineTabs.querySelectorAll(".neo-tab");
        buttons.forEach(b => {
            if (parseInt(b.dataset.step, 10) === stepNum) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });

        const stepData = currentIntermediateSteps.find(s => s.step_number === stepNum);
        if (stepData) {
            stepTitle.innerText = `Step ${stepData.step_number}: ${stepData.step_name}`;
            stepDesc.innerText = stepData.description;
            stepTextOutput.value = stepData.text_after_step;
        }
    }
});
