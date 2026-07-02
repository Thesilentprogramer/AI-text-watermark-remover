# SynthID Watermark Remover

> **Reverse-Engineering LLM Text Watermarks using Gemma 4 E2B & Multi-Stage Adversarial Paraphrasing**  
> *Techniques adapted & expanded from [aloshdenny/reverse-SynthID-text](https://github.com/aloshdenny/reverse-SynthID-text)*

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https.fastapi.tiangolo.com)
[![Gemma 4](https://img.shields.io/badge/Model-Gemma--4--E2B--it-4285F4?style=for-the-badge&logo=google)](https://huggingface.co/google/gemma-4-E2B-it)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch--2.2-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![UI Theme](https://img.shields.io/badge/UI-Claymorphism-FF7043?style=for-the-badge)](https://hype4.academy/articles/design/claymorphism-in-user-interfaces)

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
[Step 3 — ATTACK] ──► Gemma 4 E2B-it regenerates full token sequence (enable_thinking=True)
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
| **UI Theme** | **Claymorphism** — Dual inner light/dark shadows, 3D inflated cards (`24px`), tactile pill buttons based on [Hype4 Academy](https://hype4.academy/articles/design/claymorphism-in-user-interfaces) principles |
| **Testing** | Pytest, FastAPI TestClient |

---

## Claymorphism UI Aesthetics

Designed according to Michał Malewicz's **Claymorphism** specification:
- **Dual Inner Shadows**: Upper-left white specular reflection (`inset 4px 4px 8px ...`) paired with a lower-right dark inner depth shadow (`inset -6px -6px 12px ...`) creating soft 3D inflated volume.
- **Floating Drop Shadows**: Soft elevated shadows (`10px 15px 30px ...`) positioning containers cleanly above the slate backdrop.
- **Interactive Controls**: Tactile pill buttons (`999px` radius) with active click depress feedback (`transform: translateY(1px)`).

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
│   ├── index.html           ← Landing page
│   ├── app.html             ← Claymorphism workspace UI
│   ├── styles.css           ← Claymorphism CSS design system
│   └── app.js               ← Async fetch controller & score gauges
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

Open your browser at `http://localhost:8000` to access the Claymorphism web interface.

### 4. Running Tests

Execute the automated test suite:

```bash
pytest tests/
```

---

## API Documentation

### POST `/remove-watermark`

**Request Body:**

```json
{
  "text": "Google DeepMind's SynthID technology embeds an imperceptible statistical watermark into AI-generated text...",
  "attack_mode": "combined",
  "enable_thinking": true,
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
- Claymorphism UI design guidelines from [Hype4 Academy](https://hype4.academy/articles/design/claymorphism-in-user-interfaces).

---

*License: MIT · Built by [Thesilentprogramer](https://github.com/Thesilentprogramer)*
