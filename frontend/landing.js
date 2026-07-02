document.addEventListener("DOMContentLoaded", () => {
    function typeSetMath() {
        if (window.MathJax) {
            if (window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise().catch(err => console.log('MathJax typeset error:', err));
            } else if (window.MathJax.typeset) {
                window.MathJax.typeset();
            }
        }
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
        slideMathFormula.innerHTML = slide.formula;
        slideMathDesc.innerHTML = slide.desc;
        slideIndicator.innerText = `${idx + 1}/${CAROUSEL_SLIDES.length}`;
        
        // Trigger MathJax re-render
        setTimeout(typeSetMath, 50);
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

    // Initial MathJax trigger
    setTimeout(typeSetMath, 300);

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
});
