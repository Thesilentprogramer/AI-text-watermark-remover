document.addEventListener("DOMContentLoaded", () => {
    function typesetMathElements(elements) {
        const targets = (elements || []).filter(Boolean);
        if (!targets.length || !window.MathJax?.startup?.promise) return;

        return window.MathJax.startup.promise
            .then(() => {
                if (window.MathJax.typesetClear) window.MathJax.typesetClear(targets);
                return window.MathJax.typesetPromise(targets);
            })
            .catch(err => console.warn("MathJax typeset error:", err));
    }

    // ==========================================
    // CAROUSEL DATASET (6 ATTACK STRATEGIES)
    // ==========================================
    const CAROUSEL_SLIDES = [
        {
            title: "Mode 1: Combined Paraphrase + Perturb",
            input: "Google DeepMind SynthID technology embeds an imperceptible statistical watermark into AI-generated text by biasing token selection probabilities during logit processing.",
            output: "DeepMind's SynthID framework incorporates a subtle statistical signal into AI text by altering token sampling distributions. Paraphrasing cleanly erases this signature.",
            formula: "\\[ P'(x_i | x_{<i}) = P(x_i | x_{<i}) \\cdot \\left(1 + g_i - \\sum_j g_j P(x_j)\\right) \\]",
            desc: "SynthID biases logit probabilities by hashing n-gram context tokens ($k=4$). Paraphrasing regenerates the token sequence from scratch, resetting context hashes and reducing mean G-values from 0.72 to 0.49."
        },
        {
            title: "Mode 2: Gemma 4 Paraphrase",
            input: "The watermark resides within n-gram hash patterns across token sequences and survives simple formatting changes.",
            output: "The statistical signal exists in sequence n-gram context patterns, rendering basic edits ineffective.",
            formula: "\\[ T' = \\text{Paraphrase}(T) \\implies \\text{Hash}(T') \\neq \\text{Hash}(T) \\]",
            desc: "Uses a 128K context window LLM to completely regenerate the text, passing native system role prompts and stripping thinking tokens."
        },
        {
            title: "Mode 3: Homoglyph Attack",
            input: "SynthID logit biasing enables detection with high confidence.",
            output: "SуnthID lоgit biаsing еnаblеs dеtесtiоn with high соnfidеnсе.",
            formula: "\\[ c_{\\text{ASCII}} \\to c_{\\text{Cyrillic}} \\implies \\text{Tokenizer}(c_{\\text{Cyrillic}}) \\neq \\text{Tokenizer}(c_{\\text{ASCII}}) \\]",
            desc: "Replaces ASCII characters with visually identical Cyrillic/Greek Unicode lookalikes, breaking tokenizer subword indexing while remaining human-readable."
        },
        {
            title: "Mode 4: Sentence Shuffling Attack",
            input: "Sentence 1: SynthID embeds watermarks. Sentence 2: Paraphrasing erases it. Sentence 3: Detection measures G-values.",
            output: "Sentence 2: Paraphrasing erases it. Sentence 1: SynthID embeds watermarks. Sentence 3: Detection measures G-values.",
            formula: "\\[ (S_1, S_2, \\dots, S_n) \\to (S_{\\pi(1)}, S_{\\pi(2)}, \\dots, S_{\\pi(n)}) \\]",
            desc: "Permutes independent sentence order to disrupt 4-gram context boundary hashes between sentences."
        },
        {
            title: "Mode 5: Token Perturbation Only",
            input: "This sample text illustrates how the statistical excess of green-list tokens enables detection.",
            output: "This sample content demonstrates how the statistical surplus of green-list words facilitates identification.",
            formula: "\\[ w_i \\to \\text{Synonym}(w_i), \\quad r = 0.15 \\]",
            desc: "Replaces selected words with natural synonyms to introduce noise into residual n-gram green-list distributions."
        },
        {
            title: "Mode 6: Unicode Character Sanitizer",
            input: "Text containing hidden zero-width characters: Hello\\u200B World\\uFEFF!",
            output: "Cleaned character text: Hello World!",
            formula: "\\[ T' = T \\setminus \\{ \\text{U+200B}, \\text{U+FEFF}, \\text{U+200D} \\} \\]",
            desc: "Instantly strips zero-width non-joiners and hidden unicode character tricks injected at the character level."
        }
    ];

    let currentSlideIdx = 0;
    const carouselTitle = document.getElementById("carouselTitle");
    const slideInputText = document.getElementById("slideInputText");
    const slideOutputText = document.getElementById("slideOutputText");
    const slideMathFormula = document.getElementById("slideMathFormula");
    const slideMathDesc = document.getElementById("slideMathDesc");
    const slideIndicator = document.getElementById("slideIndicator");

    function renderSlide(idx) {
        if (!carouselTitle) return;
        const slide = CAROUSEL_SLIDES[idx];
        carouselTitle.innerText = slide.title;
        slideInputText.innerText = slide.input;
        slideOutputText.innerText = slide.output;
        slideMathFormula.textContent = slide.formula;
        slideMathDesc.textContent = slide.desc;
        slideIndicator.innerText = `${idx + 1}/${CAROUSEL_SLIDES.length}`;

        typesetMathElements([slideMathFormula, slideMathDesc]);
    }

    const prevSlideBtn = document.getElementById("prevSlideBtn");
    const nextSlideBtn = document.getElementById("nextSlideBtn");

    if (prevSlideBtn && nextSlideBtn) {
        prevSlideBtn.addEventListener("click", () => {
            currentSlideIdx = (currentSlideIdx - 1 + CAROUSEL_SLIDES.length) % CAROUSEL_SLIDES.length;
            renderSlide(currentSlideIdx);
        });
        nextSlideBtn.addEventListener("click", () => {
            currentSlideIdx = (currentSlideIdx + 1) % CAROUSEL_SLIDES.length;
            renderSlide(currentSlideIdx);
        });
        renderSlide(0);
    }

    // ==========================================
    // AI PROVIDER BENCHMARKS
    // ==========================================
    const PROVIDER_DATA = {
        gemini: {
            name: "GOOGLE GEMINI",
            tech: "SynthID N-gram Logit Biasing",
            gval: "G-Value: 0.72 (Watermarked)",
            badgeClass: "neo-badge red",
            desc: "Google Gemini embeds SynthID statistical logit watermarks during generation by seeding n-gram context hashes. Paraphrasing with Gemma 4 drops G-values from 0.72 to 0.49."
        },
        chatgpt: {
            name: "OPENAI CHATGPT (GPT-4o)",
            tech: "Distribution Shift & Zero-Width",
            gval: "G-Value: 0.64 (Watermarked)",
            badgeClass: "neo-badge red",
            desc: "ChatGPT models often use semantic distribution shifts and character-level zero-width unicode tokens. Our Step 1 Sanitizer strips hidden characters instantly."
        },
        anthropic: {
            name: "ANTHROPIC CLAUDE 3.5",
            tech: "Entropy Fingerprinting",
            gval: "G-Value: 0.58 (Watermarked)",
            badgeClass: "neo-badge red",
            desc: "Claude models feature distinctive low-entropy phrasing and prompt constraints. Token perturbation and paraphrasing break these stylistic fingerprints."
        },
        llama: {
            name: "META LLAMA 3 / HF",
            tech: "KGW Unigram Watermarking",
            gval: "G-Value: 0.69 (Watermarked)",
            badgeClass: "neo-badge red",
            desc: "Open-source models deployed via HuggingFace often apply KGW unigram watermarking. Our attack pipeline destroys green-list distributions cleanly."
        }
    };

    const providerTabs = document.querySelectorAll(".neo-tab[data-provider]");
    const providerBadge = document.getElementById("providerBadge");
    const providerTech = document.getElementById("providerTech");
    const providerGVal = document.getElementById("providerGVal");
    const providerDesc = document.getElementById("providerDesc");

    providerTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            providerTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            
            const p = PROVIDER_DATA[tab.dataset.provider];
            if (providerBadge) {
                providerBadge.innerText = p.name;
                providerTech.innerText = `Tech: ${p.tech}`;
                providerGVal.innerText = p.gval;
                providerGVal.className = `neo-badge ${p.badgeClass.includes("red") ? "red" : ""}`;
                providerDesc.innerText = p.desc;
            }
        });
    });

    // ==========================================
    // METHOD ARTIFACTS (animated demos)
    // ==========================================
    const ARTIFACT_DEMOS = [
        {
            title: "Auto Selector",
            description: "Reads G-value, token count, perplexity, and unicode anomalies to pick the optimal attack automatically.",
            tag: "WATERMARK REMOVAL",
            agentName: "Confidence Selector Agent",
            statusLabel: "Analyzing",
            userMessage: "Analyze this Gemini article about Harry Kane (442 tokens).",
            steps: ["Scan G-value", "Count tokens", "Check perplexity", "Select mode"],
            outcomes: [
                "Scanning n-gram green-list alignment… G-value = 0.50.",
                "Tokenizing with GPT-2… 442 tokens detected.",
                "GPT-2 perplexity = 28.4 → likely AI.",
                "Decision: paraphrase mode — long text, no SynthID signal but AI flagged."
            ],
            metaChips: ["G: 0.50", "PPL: 28", "Mode: paraphrase"],
            infoBoxes: {
                left: { title: "METRICS", lines: ["G-Value: 0.50", "Tokens: 442", "Perplexity: 28"], footer: "AI flagged" },
                right: { title: "DECISION", lines: ["Mode: paraphrase", "Layers: 1", "Est: 10–20s"], footer: "Ready" }
            }
        },
        {
            title: "SynthID G-Value Detection",
            description: "Real SynthID green-list alignment via vendored synthid-text logits processor.",
            tag: "DETECTION",
            agentName: "G-Value Scanner",
            statusLabel: "Detecting",
            userMessage: "Measure green-list alignment for this text sample.",
            steps: ["Tokenize", "4-gram hash", "G-value", "Verdict"],
            outcomes: [
                "GPT-2 tokenizer: 128 tokens encoded.",
                "Computing 4-gram context green-list hits…",
                "Mean G-value = 0.71 (threshold ≥ 0.55).",
                "Pre: 0.71 WATERMARKED → Post-attack: 0.49 CLEAN."
            ],
            metaChips: ["Pre: 0.71", "Post: 0.49", "Verdict: CLEAN"],
            infoBoxes: {
                left: { title: "PRE-ATTACK", lines: ["G-Value: 0.71", "Status: WATERMARKED", "Confidence: high"], footer: "Attack needed" },
                right: { title: "POST-ATTACK", lines: ["G-Value: 0.49", "Status: CLEAN", "Drop: 31%"], footer: "Success" }
            }
        },
        {
            title: "Perplexity Detection",
            description: "GPT-2 perplexity scoring flags AI text even when SynthID G-value is baseline.",
            tag: "DETECTION",
            agentName: "Perplexity Scanner",
            statusLabel: "Scanning",
            userMessage: "Is this AI-generated? No SynthID signal detected.",
            steps: ["Tokenize", "GPT-2 forward", "Compute PPL", "Classify"],
            outcomes: [
                "Encoding text with GPT-2 tokenizer…",
                "Running forward pass, computing cross-entropy loss…",
                "Perplexity = exp(loss) = 31.2",
                "Classification: likely AI. Flagged for attack despite G=0.50."
            ],
            metaChips: ["PPL: 31.2", "Label: likely AI", "G: 0.50"]
        },
        {
            title: "Combined Pipeline",
            description: "Full layered attack — paraphrase, synonym perturbation, and entropy for maximum signal destruction.",
            tag: "WATERMARK REMOVAL",
            agentName: "Combined Attack Agent",
            statusLabel: "Processing",
            userMessage: "Remove SynthID from this watermarked passage.",
            steps: ["Sanitize", "Paraphrase", "Perturb", "Entropy"],
            outcomes: [
                "Stripped 0 hidden unicode characters.",
                "Gemma 4 E2B rewrote token sequence under fresh distributions.",
                "Synonym perturbation applied (rate=0.15).",
                "G dropped 0.72 → 0.49. Signal destroyed."
            ],
            metaChips: ["G: 0.72→0.49", "Drop: 32%", "Status: CLEAN"]
        }
    ];

    let currentArtifactIdx = 0;
    const artifactCarousel = document.getElementById("artifactCarousel");
    const artifactIndicator = document.getElementById("artifactIndicator");

    const artifactTimers = new Map();
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function buildStepProgress(steps, artifactId) {
        const labels = steps.map((label, i) =>
            `<span class="step-progress__step" data-step="${i}">${i + 1}. ${label}</span>`
        ).join("");
        return `
            <div class="step-progress" data-artifact="${artifactId}">
                <div class="step-progress__track">
                    <div class="step-progress__bar"></div>
                </div>
                <div class="step-progress__labels">${labels}</div>
            </div>`;
    }

    function buildInfoBoxes(boxes) {
        if (!boxes) return "";
        const box = (side, data) => `
            <div class="info-box">
                <div class="info-box__title">${data.title}</div>
                ${data.lines.map(l => `<div class="info-box__line">${l}</div>`).join("")}
                <div class="info-box__footer">${data.footer}</div>
            </div>`;
        return `<div class="info-boxes">${box("left", boxes.left)}${box("right", boxes.right)}</div>`;
    }

    function buildArtifactCard(demo, idx) {
        const id = `artifact-${idx}`;
        const stepBlock = demo.infoBoxes
            ? buildInfoBoxes(demo.infoBoxes)
            : buildStepProgress(demo.steps, id);

        const chips = demo.metaChips.map(c => `<span class="artifact-chip">${c}</span>`).join("");

        return `
            <article class="method-artifact neo-card" id="${id}" data-artifact-idx="${idx}">
                <h3 class="artifact-title">${demo.title}</h3>
                <p class="artifact-desc">${demo.description}</p>
                <div class="artifact-demo">
                    <div class="artifact-header">
                        <div class="artifact-header__left">
                            <span class="artifact-tag">${demo.tag}</span>
                            <span class="artifact-agent-name">${demo.agentName}</span>
                        </div>
                        <span class="status-pill"><span class="status-dot"></span>${demo.statusLabel}</span>
                    </div>
                    <div class="chat-bubble user artifact-user">
                        <div class="avatar">IN</div>
                        <div class="bubble-body">${demo.userMessage}</div>
                    </div>
                    ${stepBlock}
                    <div class="chat-bubble agent artifact-agent">
                        <div class="avatar out">AI</div>
                        <div class="bubble-body">
                            <div class="artifact-outcome artifact-outcome--animate">${demo.outcomes[0]}</div>
                        </div>
                    </div>
                    <div class="artifact-meta">${chips}</div>
                </div>
            </article>`;
    }

    function setArtifactStep(cardEl, stepIdx) {
        const idx = parseInt(cardEl.dataset.artifactIdx, 10);
        const demo = ARTIFACT_DEMOS[idx];
        if (!demo) return;

        const maxStep = demo.outcomes.length - 1;
        const active = Math.min(stepIdx, maxStep);

        const outcomeEl = cardEl.querySelector(".artifact-outcome");
        if (outcomeEl) {
            outcomeEl.classList.remove("artifact-outcome--animate");
            void outcomeEl.offsetWidth;
            outcomeEl.textContent = demo.outcomes[active];
            outcomeEl.classList.add("artifact-outcome--animate");
        }

        const progress = cardEl.querySelector(".step-progress");
        if (progress) {
            const steps = progress.querySelectorAll(".step-progress__step");
            const bar = progress.querySelector(".step-progress__bar");
            const pct = ((active + 1) / demo.steps.length) * 100;
            if (bar) bar.style.width = `${pct}%`;
            steps.forEach((s, i) => {
                s.classList.toggle("step-progress__step--active", i === active);
                s.classList.toggle("step-progress__step--done", i < active);
            });
        }

        cardEl.dataset.currentStep = String(active);
    }

    function startArtifactAnimation(cardEl) {
        if (prefersReducedMotion || artifactTimers.has(cardEl)) return;

        const idx = parseInt(cardEl.dataset.artifactIdx, 10);
        const demo = ARTIFACT_DEMOS[idx];
        if (!demo) return;

        let step = 0;
        setArtifactStep(cardEl, step);

        const timer = setInterval(() => {
            step = (step + 1) % demo.outcomes.length;
            setArtifactStep(cardEl, step);
        }, 1800);

        artifactTimers.set(cardEl, timer);
        cardEl.classList.add("artifact--playing");
    }

    function stopArtifactAnimation(cardEl) {
        if (!cardEl) return;
        const timer = artifactTimers.get(cardEl);
        if (timer) {
            clearInterval(timer);
            artifactTimers.delete(cardEl);
        }
        cardEl.classList.remove("artifact--playing");
    }

    function renderArtifactSlide(idx) {
        if (!artifactCarousel) return;

        const existing = artifactCarousel.querySelector(".method-artifact");
        stopArtifactAnimation(existing);

        artifactCarousel.innerHTML = buildArtifactCard(ARTIFACT_DEMOS[idx], idx);
        if (artifactIndicator) {
            artifactIndicator.textContent = `${idx + 1}/${ARTIFACT_DEMOS.length}`;
        }

        const card = artifactCarousel.querySelector(".method-artifact");
        if (!card) return;

        if (prefersReducedMotion) {
            setArtifactStep(card, ARTIFACT_DEMOS[idx].outcomes.length - 1);
        } else {
            startArtifactAnimation(card);
        }
    }

    function initArtifactCarousel() {
        if (!artifactCarousel) return;

        const prevArtifactBtn = document.getElementById("prevArtifactBtn");
        const nextArtifactBtn = document.getElementById("nextArtifactBtn");

        if (prevArtifactBtn && nextArtifactBtn) {
            prevArtifactBtn.addEventListener("click", () => {
                currentArtifactIdx = (currentArtifactIdx - 1 + ARTIFACT_DEMOS.length) % ARTIFACT_DEMOS.length;
                renderArtifactSlide(currentArtifactIdx);
            });
            nextArtifactBtn.addEventListener("click", () => {
                currentArtifactIdx = (currentArtifactIdx + 1) % ARTIFACT_DEMOS.length;
                renderArtifactSlide(currentArtifactIdx);
            });
        }

        renderArtifactSlide(0);
    }

    initArtifactCarousel();
});
