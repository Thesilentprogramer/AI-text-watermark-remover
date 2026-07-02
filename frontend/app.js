document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
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
    const progressBar = document.getElementById("progressBar");
    const reductionPct = document.getElementById("reductionPct");
    const stepLogsList = document.getElementById("stepLogsList");
    const timingBadge = document.getElementById("timingBadge");
    const sanitizedBadge = document.getElementById("sanitizedBadge");

    // Carousel Elements
    const carouselTitle = document.getElementById("carouselTitle");
    const slideInputText = document.getElementById("slideInputText");
    const slideOutputText = document.getElementById("slideOutputText");
    const slideMathFormula = document.getElementById("slideMathFormula");
    const slideMathDesc = document.getElementById("slideMathDesc");
    const prevSlideBtn = document.getElementById("prevSlideBtn");
    const nextSlideBtn = document.getElementById("nextSlideBtn");
    const slideIndicator = document.getElementById("slideIndicator");

    // Provider Elements
    const providerTabs = document.querySelectorAll(".provider-tab");
    const providerBadge = document.getElementById("providerBadge");
    const providerTech = document.getElementById("providerTech");
    const providerGVal = document.getElementById("providerGVal");
    const providerDesc = document.getElementById("providerDesc");
    const loadProviderSampleBtn = document.getElementById("loadProviderSampleBtn");

    // Timeline Elements
    const transformationSection = document.getElementById("transformationSection");
    const timelineTabs = document.getElementById("timelineTabs");
    const stepTitle = document.getElementById("stepTitle");
    const stepDesc = document.getElementById("stepDesc");
    const stepTextOutput = document.getElementById("stepTextOutput");

    let currentIntermediateSteps = [];
    let currentDiffHtml = "";

    // Step Chips Element References
    const stepChips = [
        document.getElementById("stepChip1"),
        document.getElementById("stepChip2"),
        document.getElementById("stepChip3"),
        document.getElementById("stepChip4"),
        document.getElementById("stepChip5")
    ];

    // ==========================================
    // 1. CAROUSEL DATASET (6 ATTACK STRATEGIES)
    // ==========================================
    const CAROUSEL_SLIDES = [
        {
            title: "Mode 1: Combined Paraphrase + Perturb",
            input: "Google DeepMind SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing.",
            output: "DeepMind's SynthID framework incorporates a subtle statistical signal into AI text by altering token sampling distributions. Paraphrasing cleanly erases this signature.",
            formula: "P'(x_i | x_{<i}) = P(x_i | x_{<i}) \\times \\left(1 + g_i - \\mu\\right)",
            desc: "Full token sequence regeneration via Gemma 4 E2B combined with secondary synonym substitution. Completely resets n-gram context hashes, achieving ~31% G-value signal drop."
        },
        {
            title: "Mode 2: Gemma 4 E2B Paraphrase",
            input: "The watermark resides within n-gram hash patterns across token sequences and survives simple formatting changes.",
            output: "The statistical signal exists in sequence n-gram context patterns, rendering basic edits ineffective.",
            formula: "T' = \\text{AutoModelForImageTextToText}(T), \\quad \\text{Hash}(T') \\neq \\text{Hash}(T)",
            desc: "128K context window LLM regeneration. Passes native system role prompts and strips thinking tokens via parse_response()."
        },
        {
            title: "Mode 3: Homoglyph Attack",
            input: "SynthID logit biasing enables detection with high confidence.",
            output: "SуnthID lоgit biаsing еnаblеs dеtесtiоn with high соnfidеnсе.",
            formula: "c_{\\text{ASCII}} \\rightarrow c_{\\text{Cyrillic}}, \\quad \\text{Tokenizer}(c_{\\text{Cyrillic}}) \\neq \\text{Tokenizer}(c_{\\text{ASCII}})",
            desc: "Replaces ASCII characters with visually identical Cyrillic/Greek Unicode lookalikes, breaking tokenizer subword indexing while remaining human-readable."
        },
        {
            title: "Mode 4: Sentence Shuffling Attack",
            input: "Sentence 1: SynthID embeds watermarks. Sentence 2: Paraphrasing erases it. Sentence 3: Detection measures G-values.",
            output: "Sentence 2: Paraphrasing erases it. Sentence 1: SynthID embeds watermarks. Sentence 3: Detection measures G-values.",
            formula: "S_{\\pi(1)}, S_{\\pi(2)}, \\dots, S_{\\pi(n)}, \\quad \\text{Context}(S_i) \\text{ broken}",
            desc: "Permutes independent sentence order to disrupt 4-gram context boundary hashes between sentences."
        },
        {
            title: "Mode 5: Token Perturbation Only",
            input: "This sample text illustrates how the statistical excess of green-list tokens enables detection.",
            output: "This sample content demonstrates how the statistical surplus of green-list words facilitates identification.",
            formula: "w_i \\rightarrow \\text{Synonym}(w_i), \\quad r = 0.15",
            desc: "Replaces selected words with natural synonyms to introduce noise into residual n-gram green-list distributions."
        },
        {
            title: "Mode 6: Unicode Character Sanitizer",
            input: "Text containing hidden zero-width characters: Hello\\u200B World\\uFEFF!",
            output: "Cleaned character text: Hello World!",
            formula: "\\text{Text}' = \\text{Text} \\setminus \\{\\text{U+200B}, \\text{U+FEFF}, \\text{U+200D}\\}",
            desc: "Instantly strips zero-width non-joiners and hidden unicode character tricks injected at the character level."
        }
    ];

    let currentSlideIdx = 0;

    function renderSlide(idx) {
        const slide = CAROUSEL_SLIDES[idx];
        carouselTitle.innerText = slide.title;
        slideInputText.innerText = slide.input;
        slideOutputText.innerText = slide.output;
        slideMathFormula.innerText = slide.formula;
        slideMathDesc.innerText = slide.desc;
        slideIndicator.innerText = `${idx + 1} / ${CAROUSEL_SLIDES.length}`;
    }

    prevSlideBtn.addEventListener("click", () => {
        currentSlideIdx = (currentSlideIdx - 1 + CAROUSEL_SLIDES.length) % CAROUSEL_SLIDES.length;
        renderSlide(currentSlideIdx);
    });

    nextSlideBtn.addEventListener("click", () => {
        currentSlideIdx = (currentSlideIdx + 1) % CAROUSEL_SLIDES.length;
        renderSlide(currentSlideIdx);
    });

    renderSlide(0);

    // ==========================================
    // 2. AI PROVIDER SHOWCASE BENCHMARKS
    // ==========================================
    const PROVIDER_DATA = {
        gemini: {
            name: "GOOGLE GEMINI",
            tech: "SynthID N-gram Logit Biasing",
            gval: "G-Value: 0.72 (Watermarked)",
            badgeClass: "status-badge red",
            desc: "Google Gemini embeds SynthID statistical logit watermarks during generation by seeding n-gram context hashes. Paraphrasing with Gemma 4 drops G-values from 0.72 → 0.49.",
            sample: "Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing. The signal resides within n-gram hash patterns across token sequences and survives simple editing techniques."
        },
        chatgpt: {
            name: "OPENAI CHATGPT (GPT-4o)",
            tech: "Distribution Shift & Zero-Width Injection",
            gval: "G-Value: 0.64 (Watermarked)",
            badgeClass: "status-badge red",
            desc: "ChatGPT models often use semantic distribution shifts and character-level zero-width unicode tokens. Our Step 1 Sanitizer strips hidden characters instantly.",
            sample: "OpenAI ChatGPT text outputs can include structural distribution signatures alongside invisible unicode formatting markers. Running Sanitizer + Paraphrase normalizes text formatting."
        },
        anthropic: {
            name: "ANTHROPIC CLAUDE 3.5",
            tech: "Entropy Fingerprinting & Prompt Constraints",
            gval: "G-Value: 0.58 (Watermarked)",
            badgeClass: "status-badge red",
            desc: "Claude models feature distinctive low-entropy phrasing and prompt constraints. Token perturbation and paraphrasing break these stylistic fingerprints.",
            sample: "Anthropic Claude generates balanced, highly structured paragraphs with characteristic stylistic cadence. Paraphrasing transforms sentence rhythms."
        },
        llama: {
            name: "META LLAMA 3 / HUGGINGFACE",
            tech: "KGW / Unigram Green-List Watermarking",
            gval: "G-Value: 0.69 (Watermarked)",
            badgeClass: "status-badge red",
            desc: "Open-source Llama models deployed via HuggingFace often apply KGW (Kirchenbauer et al.) unigram watermarking. Our attack pipeline destroys green-list distributions cleanly.",
            sample: "Open-source models using KGW unigram logit bias force high green-list token ratios. Homoglyph and synonym substitution attacks disrupt unigram hashes effectively."
        }
    };

    let selectedProvider = "gemini";

    providerTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            providerTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            selectedProvider = tab.dataset.provider;
            renderProvider(selectedProvider);
        });
    });

    function renderProvider(key) {
        const p = PROVIDER_DATA[key];
        providerBadge.innerText = p.name;
        providerTech.innerText = p.tech;
        providerGVal.innerText = p.gval;
        providerGVal.className = p.badgeClass;
        providerDesc.innerText = p.desc;
    }

    loadProviderSampleBtn.addEventListener("click", () => {
        const p = PROVIDER_DATA[selectedProvider];
        inputText.value = p.sample;
    });

    renderProvider("gemini");

    // ==========================================
    // 3. SAMPLE BUTTONS & VIEW TOGGLE
    // ==========================================
    const WATERMARKED_SAMPLE = PROVIDER_DATA.gemini.sample;
    const CLEAN_SAMPLE = `Natural human writing features highly variable sentence lengths, creative metaphors, and unconstrained vocabulary distributions. Because no algorithmic logit biasing or n-gram hashing was applied during its creation, standard statistical watermark detectors will register baseline noise levels.`;

    sampleWmBtn.addEventListener("click", () => inputText.value = WATERMARKED_SAMPLE);
    sampleCleanBtn.addEventListener("click", () => inputText.value = CLEAN_SAMPLE);

    viewCleanBtn.addEventListener("click", () => {
        viewCleanBtn.classList.add("active-view");
        viewDiffBtn.classList.remove("active-view");
        outputText.classList.remove("hidden");
        diffViewContainer.classList.add("hidden");
    });

    viewDiffBtn.addEventListener("click", () => {
        viewDiffBtn.classList.add("active-view");
        viewCleanBtn.classList.remove("active-view");
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

    // ==========================================
    // 4. MAIN ATTACK RUNNER
    // ==========================================
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
        // Output clean text & Diff HTML
        outputText.value = data.clean_text;
        currentDiffHtml = data.diff_html || data.clean_text;
        diffHtmlOutput.innerHTML = currentDiffHtml;
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
        const buttons = timelineTabs.querySelectorAll(".tab-btn");
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
