# SynthID Watermark Remover

> **Reverse-Engineering LLM Text Watermarks using Gemma 4 E2B & Multi-Stage Adversarial Paraphrasing**  
> *Techniques adapted & expanded from [aloshdenny/reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text)*

**Live Demo:** [ultimatememer-synthid-watermark-remover.hf.space](https://ultimatememer-synthid-watermark-remover.hf.space)

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemma 4](https://img.shields.io/badge/Model-Gemma--4--E2B--it-4285F4?style=for-the-badge&logo=google)](https://huggingface.co/google/gemma-4-E2B-it)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch--2.2-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![UI Theme](https://img.shields.io/badge/UI-Neo--Brutalism-FF7043?style=for-the-badge)](https://huggingface.co/spaces/ultimatememer/synthid-watermark-remover)

---

## My Contributions vs Upstream

This project builds on [aloshdenny/reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text). The following are **original to this repo**:

| Component | Description |
|---|---|
| **Confidence-based attack selector** | Routes input to paraphrase, back-translate, combined, or sanitize based on G-value, token count, unicode anomalies, and perplexity |
| **ExtendedDetector** | Wraps upstream SynthID detector and adds GPT-2 perplexity scoring for AI text without SynthID signal |
| **Layer-based pipeline** | `sanitize → detect(pre) → attack → detect(post)` with step logs and quantitative reduction metrics |
| **Back-translation attack** | EN→DE→EN via Helsinki-NLP MarianMT for fast n-gram boundary reset |
| **Full-stack eval loop** | FastAPI API + workspace UI showing pre/post G-values, perplexity, and word-level diff |
| **Landing page + method artifacts** | Animated pipeline demos explaining each attack method |

**From upstream (vendored):** `synthid_text` module, `WatermarkDetector`, homoglyph attack, token perturbation classes.

---

## Portfolio

**Resume bullet:**
> Built an adversarial ML pipeline to evaluate SynthID text watermark robustness, combining G-value detection, GPT-2 perplexity scoring, and a confidence-based attack selector (paraphrase / back-translation / combined). Deployed as a FastAPI app with quantitative pre/post attack metrics on Hugging Face Spaces.

---

## What This Project Really Demonstrates

This project is not just a standard wrapper around an API — it is an **adversarial ML research implementation** that demonstrates deep understanding of LLM watermarking physics at the token-probability level:

1. **Token Probability Level Watermarks**: Google DeepMind's SynthID embeds watermarks at generation time by seeding a pseudo-random hash over previous $k$ tokens (n-gram context) to bias token selection probabilities toward a designated "green list" vocabulary.
2. **Quantitative G-Value Metrics**: The signal is purely statistical. A mean **G-value > 0.55** indicates watermarked sequence confidence, while **G-value ~0.49–0.51** represents baseline un-watermarked noise.
3. **Paraphrasing Attack Physics**: Paraphrasing completely regenerates the token sequence using an unwatermarked model (`google/gemma-4-E2B-it`). Because new tokens are drawn from natural distributions under a fresh model, the n-gram hash patterns are destroyed regardless of how the source text was originally watermarked.
4. **Character vs. Statistical Watermarks**: Differentiates between character-level invisible tricks (zero-width unicode injections like `\u200B` or `\uFEFF`) and statistical-level logit biasing.
5. **Self-Evaluating Verification Engine**: Measures its own effectiveness in real-time by providing quantitative pre-attack vs post-attack G-value scoring and percentage reduction metrics.

---

## Confidence-Based Attack Selector (Original Contribution)

No existing watermark-removal tool auto-selects the attack strategy. The upstream [reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text) CLI requires manual `--method` selection. This project adds a **confidence-based selector** that reads input characteristics and picks the optimal attack:

| Condition | Auto-selected mode | Rationale |
|---|---|---|
| G-value > 0.75 | `combined` | Very strong watermark — run all layers |
| Unicode anomaly detected | `sanitize_perturb` | Character-level watermark — sanitize then perturb |
| G ≥ 0.55, tokens < 200 | `backtranslate` | Short text — fast EN→DE→EN back-translation |
| G ≥ 0.55, tokens ≥ 200 | `paraphrase` | Long text — thorough Gemma 4 rewrite |
| Clean input | `none` | No attack needed |

Set `attack_mode: "auto"` in the API request to enable. The selector's rationale is returned in `auto_rationale` and displayed in the UI.

---

## 5-Step Adversarial Pipeline Architecture

```
User Text Input
       │
       ▼
[Step 1 — SANITIZE] ──► Removes hidden unicode zero-width characters (\u200B, \uFEFF)
       │
       ▼
[Step 2 — DETECT (Pre)] ──► WatermarkDetector calculates initial G-value (Threshold ≥ 0.55)
       │
       ▼
[Step 3 — ATTACK] ──► Gemma 4 E2B-it regenerates full token sequence
       │
       ▼
[Step 4 — PERTURB] ──► Secondary synonym swapping & structural variation pass
       │
       ▼
[Step 5 — DETECT (Post)] ──► Re-evaluates G-value (~0.49–0.51) & computes % reduction
       │
       ▼
Show User: Before G-Value | After G-Value | % Reduction | Clean Output Text
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11, PyTorch 2.2+, Uvicorn |
| **ML Model** | `google/gemma-4-E2B-it` via Hugging Face `AutoModelForImageTextToText` & `AutoProcessor` (`bfloat16`, `device_map="auto"`) |
| **Watermark Engine** | Vendored `synthid-text` via [reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text), `ExtendedDetector`, GPT-2 tokenizer |
| **UI Theme** | **Neo-brutalism** — bold borders, offset shadows, high-contrast cards |
| **Testing** | Pytest, FastAPI TestClient |

---

## Neo-Brutalism UI

The frontend uses a **neo-brutalism** design system:
- **Bold borders** (`3px solid #000`) and offset box shadows (`4px 4px 0 #000`)
- **High-contrast color blocks** — yellow, cyan, pink, green accent badges
- **Space Grotesk + JetBrains Mono** typography
- **Animated method artifacts** on the landing page with step progress and chat-bubble demos

---

## Project Structure

```
AI-text-watermark-remover/
├── app/
│   ├── main.py              ← FastAPI endpoints & lifespan startup
│   ├── pipeline.py          ← Layer-based attack pipeline orchestrator
│   ├── model_loader.py      ← Singleton model loader
│   └── schemas.py           ← Pydantic request/response models
├── attacks/
│   ├── sanitizer.py         ← Step 1: Zero-width unicode stripper
│   ├── gemma4_paraphrase.py ← Gemma 4 E2B paraphrasing (custom rewrite)
│   ├── backtranslate.py     ← Helsinki-NLP back-translation attack
│   ├── entropy.py           ← Sentence variation + synonym entropy layer
│   └── auto_selector.py     ← Confidence-based attack selector (original)
├── reverse_synthid/
│   ├── reverse_synthid.py   ← Vendored upstream attack toolkit
│   ├── extended_detector.py ← SynthID G-value + optional perplexity
│   └── src/synthid_text/    ← Vendored synthid-text module
├── frontend/
│   ├── index.html           ← Landing page with MathJax carousel
│   ├── app.html             ← Workspace UI
│   ├── styles.css           ← Neo-brutalism CSS design system
│   └── app.js               ← Async fetch controller & score gauges
├── scripts/
│   ├── eval_benchmark.py    ← Reproducible detection/attack benchmark
│   └── deploy-hf.sh         ← Hugging Face Space deploy script
├── results/
│   └── benchmark.json       ← Latest benchmark snapshot
├── tests/
│   ├── test_detector.py
│   ├── test_auto_selector.py
│   ├── test_api.py
│   ├── test_backtranslate.py
│   └── test_entropy.py
├── requirements.txt
├── .env.example
├── DEVELOPMENT.md
└── README.md
```

---

## Getting Started

### 1. Installation

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/Thesilentprogramer/AI-text-watermark-remover.git
cd AI-text-watermark-remover

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your keys if using API fallbacks or gated Hugging Face models:

```bash
cp .env.example .env
```

### 3. Running the Server

Start the FastAPI application:

```bash
uvicorn app.main:app --reload --port 8000
```

Open your browser at `http://localhost:8000` to access the web interface.

### 4. Running Tests

Execute the automated test suite:

```bash
pytest tests/ --ignore=tests/test_backtranslate.py
```

### 5. Running the Benchmark

Reproduce evaluation numbers for the README:

```bash
python scripts/eval_benchmark.py --detect-only   # CI-safe: detection only
python scripts/eval_benchmark.py                   # + attack modes (API key for paraphrase)
```

Results are written to `results/benchmark.json`.

---

## Evaluation Results

Benchmark run on 5 sample texts (bundled `watermarked.txt` / `clean.txt` + synthetic samples). Regenerate with `python scripts/eval_benchmark.py`.

### Detection (pre-attack)

| Sample | G-value | Watermarked | Tokens |
|---|---:|---|---:|
| watermarked | 0.5086 | no | 155 |
| clean | 0.4935 | no | 106 |
| unicode_hidden | 0.4875 | no | 20 |
| gemini_style | 0.4938 | no | 52 |
| short_ai | 0.5175 | no | 25 |

### Attack modes (avg across 5 samples)

| Attack mode | Samples | Avg G (pre) | Avg G (post) | Avg drop % | Success rate (G_post < 0.55) |
|---|---:|---:|---:|---:|---:|
| perturb | 5 | 0.4971 | 0.4957 | 0.5% | 100.0% |
| sanitize_perturb | 5 | 0.4971 | 0.4936 | 0.7% | 100.0% |
| paraphrase | 5 | 0.4971 | 0.4877 | 1.9% | 100.0% |
| combined | 5 | 0.4971 | 0.4839 | 2.6% | 100.0% |
| auto | 5 | 0.4971 | 0.4971 | 0.0% | 100.0% |

*Note: Bundled samples have baseline G-values (~0.49–0.52) without strong SynthID signal. Stronger watermarked inputs (G > 0.55) show larger drops — see the live demo with Gemini output.*

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Output identical to input | Auto mode selected `none` | Use **Combined** mode, or paste AI text with low perplexity (&lt; 60) |
| Only slight word swaps | Gemma 4 API failed → heuristic fallback | Set `GOOGLE_API_KEY` (free: [Google AI Studio](https://aistudio.google.com/apikey)); `PARAPHRASE_BACKEND=api` |
| Slow on long text | Chunked API fallback | Default `GEMMA_API_CHUNKING=auto` uses one free API call first; chunking only if needed |
| No API key | Heuristic only | Set `PARAPHRASE_BACKEND=heuristic` (free, instant, weaker rewrites) |
| Error on back-translate | MarianMT / torch version issue | Use **Paraphrase** or **Combined**; upgrade torch ≥ 2.6 |
| Auto still skips attack | Perplexity disabled or borderline G | Set `ENABLE_PERPLEXITY=true`; for demos set `FORCE_ATTACK=true` |

The workspace UI shows **Attack** mode and **Paraphrase** source (`api`, `local`, or `heuristic`) after each run. A yellow warning appears when output is unchanged or heuristic fallback was used.

---

## Ethics and Limitations

**Purpose:** This project is for **adversarial robustness research** and red-teaming watermark detectors. It evaluates how statistical watermarks behave under paraphrase, perturbation, and translation attacks.

**Not intended for:** circumventing academic integrity policies, bypassing platform content rules, or disguising AI-generated work as human-written.

**Limitations:**
- Paraphrase uses the **free Google AI Studio tier** by default (`GOOGLE_API_KEY` + `PARAPHRASE_BACKEND=api`). Set `PARAPHRASE_BACKEND=heuristic` for zero-API local rewrites. GPU local Gemma needs `ENABLE_LOCAL_GEMMA=true` + `HF_TOKEN`.
- GPT-2 perplexity is a heuristic, not a calibrated AI detector
- G-value threshold (≥ 0.55) assumes SynthID-style n-gram green-list biasing
- Homoglyph and back-translation modes may fail on certain torch/transformers versions

---

## API Documentation

### POST `/remove-watermark`

**Request Body:**

```json
{
  "text": "Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text...",
  "attack_mode": "combined",
  "substitution_rate": 0.15
}
```

**Response Payload:**

```json
{
  "clean_text": "DeepMind's SynthID system incorporates a subtle statistical signal into AI text...",
  "sanitized_char_count": 0,
  "pre_attack": {
    "g_value": 0.71,
    "is_watermarked": true,
    "confidence": "high",
    "sample_tokens": 42
  },
  "post_attack": {
    "g_value": 0.49,
    "is_watermarked": false,
    "confidence": "low",
    "sample_tokens": 40
  },
  "watermark_reduction_pct": 30.99,
  "attack_used": "combined",
  "paraphrase_source": "api",
  "output_unchanged": false,
  "processing_time_ms": 1420
}
```

### GET `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "engine": "Gemma 4 E2B + WatermarkDetector"
}
```

---

## Acknowledgments & References

- Base reverse-engineering techniques & reference classes from [aloshdenny/reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text).
- SynthID paper & logit processor specifications by Google DeepMind.
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/ultimatememer/synthid-watermark-remover).

---

*License: MIT · Built by [Thesilentprogramer](https://github.com/Thesilentprogramer)*
