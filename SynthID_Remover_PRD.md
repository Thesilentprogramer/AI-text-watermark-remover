# SynthID Watermark Remover — PRD

> Reverse-engineering LLM text watermarks using Gemma 4 E2B  
> Portfolio ML Project · v1.0 · 2026

| Attribute | Value |
|---|---|
| Project Type | Portfolio / Open Source |
| Primary Model | google/gemma-4-E2B-it |
| Base Repo | aloshdenny/reverse-SynthID-text |
| Backend | FastAPI + Python 3.11 |
| Frontend | Vanilla HTML / CSS / JS |
| Target Infra | Single GPU (T4 / A10G) or RunPod |
| Est. Build Time | 3–4 weeks (solo) |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Technical Background](#2-technical-background)
3. [Scope](#3-scope)
4. [System Architecture](#4-system-architecture)
5. [API Contract](#5-api-contract)
6. [Frontend Spec](#6-frontend-spec)
7. [Phased Development Plan](#7-phased-development-plan)
8. [Project Structure](#8-project-structure)
9. [Requirements](#9-requirements)
10. [Portfolio Framing](#10-portfolio-framing)

---

## 1. Problem Statement

Google DeepMind's SynthID embeds an invisible watermark into AI-generated text by biasing token selection probabilities during generation. The signal is statistical — it lives in n-gram hash patterns across the token sequence — and it survives copy-paste, minor edits, and format changes.

The goal of this project is to build a web application that detects and removes SynthID watermarks from text using a paraphrasing attack powered by Gemma 4 E2B-it, with quantitative before/after verification via G-value scoring.

> **Portfolio Angle** — This project demonstrates: (1) understanding of LLM watermarking at the token-probability level, (2) correct integration of a Gemma 4 multimodal model including its API differences from older Gemma versions, (3) ML attack design with quantitative evaluation — all wrapped in a clean deployable app.

---

## 2. Technical Background

### 2.1 How SynthID Watermarking Works

At generation time, the model computes a hash of the last k tokens (the n-gram context). That hash seeds a pseudo-random bit that biases the logit distribution toward a subset of the vocabulary — the "green list". Across thousands of tokens, the statistical excess of green-list tokens becomes a detectable signal.

Detection computes a **mean G-value**. A score above ~0.55 indicates a watermarked sequence with high confidence.

### 2.2 Why Paraphrasing Destroys It

Paraphrasing completely regenerates the token sequence using a different model — one with no SynthID watermarking applied. The n-gram hash patterns are entirely new, drawn from a natural distribution. The G-value drops to noise level (~0.49–0.51) regardless of how the source text was originally watermarked.

### 2.3 Attack Methods in the Repo

The base repo (`aloshdenny/reverse-SynthID-text`) ships five attack classes. These are used as-is except for `ParaphrasingAttack`, which is rewritten for Gemma 4 E2B.

| Class | Method | Effectiveness | GPU Needed |
|---|---|---|---|
| `ParaphrasingAttack` | Full token sequence regeneration via LLM | 90–100% | ✅ Yes |
| `TokenPerturbationAttack` | Synonym substitution + filler insertion | 50–70% | ❌ No |
| `HomoglyphAttack` | Unicode lookalike character replacement | 95–100% | ❌ No |
| `SentenceShufflingAttack` | Reorders sentences to break n-gram order | 30–50% | ❌ No |
| `WhitespaceAttack` | Inserts zero-width whitespace characters | 20–40% | ❌ No |

> **Important:** `WatermarkDetector` in the repo uses a GPT-2 tokenizer by default (`tokenizer_name="gpt2"`) and relies on `SynthIDLogitsProcessor` from `synthid-text`. You need both `synthid-text` installed and a GPT-2 tokenizer download just for the detection step — separate from your Gemma 4 generation model.

### 2.4 What Changes for Gemma 4 E2B

The repo's `ParaphrasingAttack` was written for `google/gemma-2b-it`. Gemma 4 E2B has a different API that requires these specific changes:

| Property | Gemma 2B-it (repo default) | Gemma 4 E2B-it (this project) |
|---|---|---|
| Model class | `AutoModelForCausalLM` | `AutoModelForImageTextToText` |
| Processor | `AutoTokenizer` | `AutoProcessor` |
| Context window | ~8K tokens | 128K tokens |
| System role | Workaround needed | Native support |
| Output parsing | Manual decode | `processor.parse_response()` |
| Thinking mode | None | `enable_thinking=True` |
| Recommended sampling | Not specified | temp=1.0, top_p=0.95, top_k=64 |
| Repo default sampling | temp=0.7, top_p=0.9 | Use Gemma 4 values instead |

---

## 3. Scope

### In Scope

- Watermark detection — compute G-value on input text using `WatermarkDetector` from the repo
- Paraphrasing attack — rewritten `Gemma4ParaphrasingAttack` using `AutoModelForImageTextToText`
- Token perturbation — `TokenPerturbationAttack` from repo used as-is (secondary pass)
- Post-attack verification — G-value before/after comparison with confidence label
- Simple web UI — textarea in, clean text out, scores displayed
- REST API — single `/remove-watermark` endpoint + `/health`
- Model loading with `bfloat16` + `device_map="auto"`

### Out of Scope

- Support for other watermarking schemes (KGW, Unigram, etc.)
- User accounts, auth, or data persistence
- Batch processing or file upload
- Mobile-native app
- Fine-tuning the paraphrasing model

---

## 4. System Architecture

### 4.1 Component Map

```
┌──────────────────────────────────────────────┐
│                  Browser                     │
│   [Textarea] → [Run Attack] → [Results UI]  │
└─────────────────────┬────────────────────────┘
                      │ POST /remove-watermark
                      ▼
┌──────────────────────────────────────────────┐
│              FastAPI Backend                 │
│  ┌──────────────┐   ┌─────────────────────┐  │
│  │  Watermark   │   │ Attack Orchestrator │  │
│  │  Detector    │   │   (pipeline.py)     │  │
│  └──────────────┘   └──────────┬──────────┘  │
│                                │              │
│              ┌─────────────────▼───────────┐  │
│              │  Gemma4ParaphrasingAttack   │  │
│              │  (rewritten for Gemma 4)    │  │
│              └─────────────────────────────┘  │
│                                               │
│              ┌─────────────────────────────┐  │
│              │  TokenPerturbationAttack    │  │
│              │  (used as-is from repo)     │  │
│              └─────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

### 4.2 Data Flow

| Step | Component | Input | Output |
|---|---|---|---|
| 1 | `WatermarkDetector` | Raw text | G-value (float), `is_watermarked` (bool) |
| 2 | `Gemma4ParaphrasingAttack` | Raw text | Paraphrased text (new token sequence) |
| 3 | `TokenPerturbationAttack` | Paraphrased text | Perturbed text (synonym swap) |
| 4 | `WatermarkDetector` | Perturbed text | G-value post-attack |
| 5 | API response | All of above | JSON → rendered in UI |

### 4.3 What Comes From the Repo vs What You Write

| Component | Source |
|---|---|
| `WatermarkDetector` | ✅ Repo — use as-is |
| `TokenPerturbationAttack` | ✅ Repo — use as-is |
| `HomoglyphAttack` | ✅ Repo — use as-is (optional layer) |
| `SentenceShufflingAttack` | ✅ Repo — use as-is (optional) |
| `ParaphrasingAttack` | ❌ Rewrite for Gemma 4 E2B API |
| `pipeline.py` | ✍️ Write fresh — orchestrates all steps |
| `model_loader.py` | ✍️ Write fresh — singleton Gemma 4 loader |
| FastAPI layer | ✍️ Write fresh |
| Frontend | ✍️ Write fresh |

---

## 5. API Contract

### POST `/remove-watermark`

**Request**

```json
{
  "text": "string (required) — the watermarked input text",
  "attack_mode": "paraphrase | perturb | combined  (default: combined)",
  "enable_thinking": "bool (default: true)",
  "substitution_rate": "float 0.0–1.0 (default: 0.15)"
}
```

**Response**

```json
{
  "clean_text": "string — watermark-removed output",
  "pre_attack": {
    "g_value": 0.71,
    "is_watermarked": true,
    "confidence": "high"
  },
  "post_attack": {
    "g_value": 0.49,
    "is_watermarked": false,
    "confidence": "low"
  },
  "attack_used": "combined",
  "processing_time_ms": 4821
}
```

### GET `/health`

```json
{ "status": "ok", "model_loaded": true, "device": "cuda" }
```

---

## 6. Frontend Spec

The frontend is intentionally minimal. Its job is to surface the ML output clearly — not to be a design showcase.

### Layout

```
┌─────────────────────────────────────────────┐
│  SynthID Watermark Remover          [dark]  │  ← header
├─────────────────────────────────────────────┤
│                                             │
│  [Paste your AI-generated text here...]    │  ← textarea
│                                             │
│  Attack Mode: [Combined ▾]  [Run Attack]   │  ← controls
│                                             │
├─────────────────────────────────────────────┤
│  Before: G=0.71 🔴 Watermarked             │  ← score bar
│  After:  G=0.49 🟢 Clean                   │
│  ████████████████░░░░  68% reduction        │
├─────────────────────────────────────────────┤
│  [Clean text output here...]               │  ← output
│                            [Copy] [Download]│
└─────────────────────────────────────────────┘
```

### Tech Stack

- Vanilla HTML + CSS + JS — no framework
- Single `index.html`, one `styles.css`, one `app.js`
- Fetch API for the backend call
- No build step, no npm on the frontend

---

## 7. Phased Development Plan

> **Total estimate: 3–4 weeks solo. Each phase is independently demo-able.**

---

### Phase 1 — Core ML Pipeline (Days 1–5)

**Goal: Get the attack working in pure Python, no web layer yet**

**Tasks**

- Clone `reverse-SynthID-text` and read the full source — understand every class before touching anything
- Install dependencies: `transformers`, `torch`, `accelerate`, `synthid-text`
- Write `Gemma4ParaphrasingAttack` class:
  - Use `AutoProcessor` + `AutoModelForImageTextToText` (not `AutoModelForCausalLM`)
  - Native system role in chat template
  - `enable_thinking=True` for higher paraphrase diversity
  - `processor.parse_response()` to strip thinking tokens from output
  - Sampling: `temperature=1.0, top_p=0.95, top_k=64` (Gemma 4 recommended, not repo defaults)
- Test `paraphrase()` on 3–5 sample watermarked texts
- Verify G-value drops from >0.55 to <0.52 consistently using `WatermarkDetector`
- Write a test script: input text → print before/after G-values to terminal

**Deliverable:** A single Python script that takes text, runs the attack, prints before/after G-values.

**Key gotcha:** Use `google/gemma-4-E2B-it` (instruct), not the base model. The base model ignores paraphrasing instructions entirely.

**Gemma 4 E2B code skeleton:**

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

class Gemma4ParaphrasingAttack:
    def __init__(self):
        self.model_id = "google/gemma-4-E2B-it"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def paraphrase(self, text: str, enable_thinking: bool = True) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a text rewriting assistant. Rewrite the given text completely in your own words while preserving the exact meaning. Output only the rewritten text."
            },
            {
                "role": "user",
                "content": f"Rewrite the following text:\n\n{text}"
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )

        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        return self.processor.parse_response(response)  # strips thinking tokens
```

---

### Phase 2 — FastAPI Backend (Days 6–10)

**Goal: Wrap the ML pipeline in a REST API**

**Tasks**

- Create project structure (see Section 8)
- Write `model_loader.py` — singleton pattern so Gemma 4 loads once at startup, not per request
- Write `pipeline.py` — orchestrates `WatermarkDetector` → `Gemma4ParaphrasingAttack` → `TokenPerturbationAttack` → `WatermarkDetector`
- Write Pydantic request/response models in `schemas.py`
- Implement `/remove-watermark` endpoint
- Implement `/health` endpoint
- Add CORS middleware for local frontend dev
- Error handling: model not loaded, empty text, GPU OOM
- Test all endpoints with `curl` or `httpie`

**Deliverable:** Running FastAPI server on `localhost:8000` that accepts text and returns clean text + G-value scores.

**Key gotcha:** Load the model at app startup using FastAPI's `lifespan` event, not inside the route handler. Loading Gemma 4 takes ~30s — you cannot do this per request.

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()   # runs once at startup
    yield

app = FastAPI(lifespan=lifespan)
```

---

### Phase 3 — Frontend (Days 11–14)

**Goal: A clean, functional UI that surfaces the ML output**

**Tasks**

- Build `index.html` — header, textarea, attack mode dropdown, run button
- Build results section — before/after G-value display, reduction progress bar, clean text output area
- Wire `fetch()` call to `/remove-watermark` endpoint
- Loading state — disable button and show spinner while model runs (can take 5–15s)
- Error state — display message if API returns an error
- Copy to clipboard button on output
- Dark mode CSS (one media query, signals technical taste without much effort)

**Deliverable:** Working end-to-end demo in browser — paste text, click run, get clean text with G-value comparison.

**Key gotcha:** The request can take 5–15 seconds depending on GPU. Show a clear loading state or users will think it's broken.

---

### Phase 4 — Polish & Portfolio Packaging (Days 15–21)

**Goal: Make it presentable as a portfolio piece**

**Tasks**

- Record a 2–3 minute demo video (Loom or OBS) — show a watermarked text going in, G-value dropping, clean text coming out
- Write `README.md`:
  - What SynthID is and how it works at the token level
  - Why paraphrasing defeats it
  - Why the repo's original `ParaphrasingAttack` needed rewriting for Gemma 4 E2B
  - Setup and run instructions
  - Hardware requirements
- Add a **Technical Deep Dive** section to README explaining G-value math
- Deploy backend to RunPod or a free-tier GPU cloud instance
- Optionally deploy frontend to Vercel or GitHub Pages
- Push to GitHub with a clean, readable commit history
- Write the portfolio blurb (see Section 10)

---

## 8. Project Structure

```
synthid-remover/
├── app/
│   ├── main.py              ← FastAPI app + CORS + lifespan
│   ├── pipeline.py          ← attack orchestration
│   ├── model_loader.py      ← Gemma 4 singleton loader
│   ├── detector.py          ← WatermarkDetector wrapper
│   └── schemas.py           ← Pydantic request/response models
├── attacks/
│   ├── gemma4_paraphrase.py ← rewritten ParaphrasingAttack for Gemma 4
│   └── token_perturbation.py← TokenPerturbationAttack (from repo, as-is)
├── reverse_synthid/         ← cloned repo classes (WatermarkDetector etc.)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 9. Requirements

### Python Dependencies

```
# requirements.txt
transformers>=4.51.0
torch>=2.2.0
accelerate>=0.27.0
synthid-text>=0.0.1
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.5.0
python-dotenv>=1.0.0
```

### Hardware

| Tier | VRAM | Speed | Cost |
|---|---|---|---|
| Minimum | 10 GB | ~15s/request | RunPod T4 ~$0.20/hr |
| Recommended | 16 GB | ~5s/request | RunPod A10G ~$0.75/hr |
| Dev/test | CPU only | ~120s/request | Free (slow but works) |

> Gemma 4 E2B has 5.1B total params (including embeddings). At bfloat16 this requires ~10–12 GB VRAM.

---

## 10. Portfolio Framing

**Project title:**
> Reverse-engineering SynthID: a study in LLM watermark fragility

**Portfolio description:**
> Built a web application that detects and removes Google DeepMind's SynthID watermarks from AI-generated text. The attack exploits the fundamental fragility of n-gram hash-based watermarking: because SynthID's signal lives in token selection probabilities rather than semantic content, a full paraphrase using Gemma 4 E2B-it completely regenerates the token sequence and destroys the watermark. Implemented quantitative verification via G-value scoring (pre/post attack), achieving consistent drops from ~0.71 → ~0.49 across test samples. Adapted the original `ParaphrasingAttack` class from `reverse-SynthID-text` for Gemma 4's updated API (`AutoModelForImageTextToText`, `AutoProcessor`, `parse_response()`). Backend: FastAPI. Frontend: Vanilla JS.

**What this signals to a hiring manager:**

- You understand LLM internals at the token-probability level, not just prompt engineering
- You can read and adapt real ML research code
- You know the Gemma 4 API differences from older versions — a specific, current, niche detail
- You can evaluate your own ML system quantitatively (G-value before/after), not just qualitatively
- You can ship a complete working product end to end

---

*End of PRD*
