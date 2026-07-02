document.addEventListener("DOMContentLoaded", () => {
    const inputText = document.getElementById("inputText");
    const outputText = document.getElementById("outputText");
    const attackMode = document.getElementById("attackMode");
    const enableThinking = document.getElementById("enableThinking");
    const runBtn = document.getElementById("runBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const sampleWmBtn = document.getElementById("sampleWmBtn");
    const sampleCleanBtn = document.getElementById("sampleCleanBtn");
    const copyBtn = document.getElementById("copyBtn");

    const verdictBanner = document.getElementById("verdictBanner");
    const verdictText = document.getElementById("verdictText");
    const preGVal = document.getElementById("preGVal");
    const preBadge = document.getElementById("preBadge");
    const postGVal = document.getElementById("postGVal");
    const postBadge = document.getElementById("postBadge");
    const progressBar = document.getElementById("progressBar");
    const reductionPct = document.getElementById("reductionPct");
    const stepLogsList = document.getElementById("stepLogsList");
    const timingBadge = document.getElementById("timingBadge");
    const sanitizedBadge = document.getElementById("sanitizedBadge");

    const transformationSection = document.getElementById("transformationSection");
    const timelineTabs = document.getElementById("timelineTabs");
    const stepTitle = document.getElementById("stepTitle");
    const stepDesc = document.getElementById("stepDesc");
    const stepTextOutput = document.getElementById("stepTextOutput");

    let currentIntermediateSteps = [];

    // Step Chips Element References
    const stepChips = [
        document.getElementById("stepChip1"),
        document.getElementById("stepChip2"),
        document.getElementById("stepChip3"),
        document.getElementById("stepChip4"),
        document.getElementById("stepChip5")
    ];

    // Sample Texts
    const WATERMARKED_SAMPLE = `Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing. The signal resides within n-gram hash patterns across token sequences and survives simple editing techniques. This sample text illustrates how the statistical excess of green-list tokens enables watermark detectors to identify synthetic text with high confidence.`;

    const CLEAN_SAMPLE = `Natural human writing features highly variable sentence lengths, creative metaphors, and unconstrained vocabulary distributions. Because no algorithmic logit biasing or n-gram hashing was applied during its creation, standard statistical watermark detectors will register baseline noise levels.`;

    sampleWmBtn.addEventListener("click", () => {
        inputText.value = WATERMARKED_SAMPLE;
    });

    sampleCleanBtn.addEventListener("click", () => {
        inputText.value = CLEAN_SAMPLE;
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

        // Loading UI state & animate step chips 1 -> 5
        runBtn.disabled = true;
        btnText.innerText = "Executing Attack Pipeline...";
        btnSpinner.classList.remove("hidden");

        verdictBanner.className = "verdict-banner gray-banner";
        verdictText.innerText = "Executing 5-step adversarial attack pipeline...";

        resetStepChips();
        animateStepChipsProgress();

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
            markAllStepsCompleted();
            renderResults(data);
        } catch (err) {
            alert(`Error: ${err.message}`);
            verdictBanner.className = "verdict-banner red-banner";
            verdictText.innerText = `Error executing attack: ${err.message}`;
            resetStepChips();
        } finally {
            runBtn.disabled = false;
            btnText.innerText = "Remove Watermark";
            btnSpinner.classList.add("hidden");
        }
    });

    function resetStepChips() {
        stepChips.forEach(chip => {
            if (chip) chip.className = "step-chip";
        });
    }

    function animateStepChipsProgress() {
        let current = 0;
        const interval = setInterval(() => {
            if (current < stepChips.length) {
                if (stepChips[current]) {
                    stepChips[current].className = "step-chip active";
                }
                current++;
            } else {
                clearInterval(interval);
            }
        }, 300);
    }

    function markAllStepsCompleted() {
        stepChips.forEach(chip => {
            if (chip) chip.className = "step-chip completed";
        });
    }

    function renderResults(data) {
        // Output clean text
        outputText.value = data.clean_text;
        copyBtn.disabled = !data.clean_text;

        // Render Verdict Banner
        verdictText.innerText = data.verdict_title;
        if (data.is_clean) {
            verdictBanner.className = "verdict-banner green-banner";
        } else if (data.watermark_reduction_pct > 15.0) {
            verdictBanner.className = "verdict-banner amber-banner";
        } else {
            verdictBanner.className = "verdict-banner red-banner";
        }

        // Pre & Post Scores
        preGVal.innerText = data.pre_attack.g_value.toFixed(2);
        preBadge.className = data.pre_attack.is_watermarked ? "status-badge red" : "status-badge green";
        preBadge.innerText = data.pre_attack.is_watermarked ? "Watermarked" : "Clean";

        postGVal.innerText = data.post_attack.g_value.toFixed(2);
        postBadge.className = data.post_attack.is_watermarked ? "status-badge red" : "status-badge green";
        postBadge.innerText = data.post_attack.is_watermarked ? "Watermarked" : "Clean";

        // Progress Bar
        const redPct = data.watermark_reduction_pct;
        reductionPct.innerText = `${redPct.toFixed(1)}%`;
        progressBar.style.width = `${Math.min(Math.max(redPct, 5), 100)}%`;

        // Step Logs
        stepLogsList.innerHTML = "";
        if (data.step_logs && data.step_logs.length > 0) {
            data.step_logs.forEach(log => {
                const li = document.createElement("li");
                li.innerText = `▸ ${log}`;
                stepLogsList.appendChild(li);
            });
        }

        timingBadge.innerText = `Runtime: ${data.processing_time_ms} ms`;
        sanitizedBadge.innerText = `Unicode Stripped: ${data.sanitized_char_count} chars`;

        // Render Paragraph Transformation Timeline
        currentIntermediateSteps = data.intermediate_steps || [];
        if (currentIntermediateSteps.length > 0) {
            transformationSection.classList.remove("hidden");
            selectTimelineStep(1);
        }
    }

    // Timeline Tab Click Listener
    timelineTabs.addEventListener("click", (e) => {
        const btn = e.target.closest(".tab-btn");
        if (!btn) return;
        const stepNum = parseInt(btn.dataset.step, 10);
        selectTimelineStep(stepNum);
    });

    function selectTimelineStep(stepNum) {
        // Highlight active tab
        const buttons = timelineTabs.querySelectorAll(".tab-btn");
        buttons.forEach(b => {
            if (parseInt(b.dataset.step, 10) === stepNum) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });

        // Find step payload
        const stepData = currentIntermediateSteps.find(s => s.step_number === stepNum);
        if (stepData) {
            stepTitle.innerText = `Step ${stepData.step_number}: ${stepData.step_name}`;
            stepDesc.innerText = stepData.description;
            stepTextOutput.value = stepData.text_after_step;
        }
    }
});
