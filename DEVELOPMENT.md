# SynthID Watermark Remover — Development Notes

## What This App Really Does

The SynthID Watermark Remover is an **ML-powered adversarial pipeline** designed to detect and reverse-engineer Google DeepMind's SynthID LLM text watermarks. Instead of relying on heuristic text tweaks, it uses a multi-stage attack combining **Gemma 4 E2B-it paraphrasing** and **token perturbation** to destroy the statistical n-gram hash signature of watermarked text while preserving semantic meaning.

### The Core Idea

When a user submits text (e.g. AI-generated article or essay), the system runs the following 4-step pipeline:

1. **Pre-Attack Watermark Detection** — Calculates the baseline statistical **G-value** of the input text using `synthid-text` and a GPT-2 tokenizer. A G-value >0.55 indicates high confidence of a SynthID watermark.
2. **Gemma 4 Paraphrasing Attack** — Passes the text to `google/gemma-4-E2B-it` via `AutoModelForImageTextToText`. With thinking mode enabled (`enable_thinking=True`), the model completely regenerates the token sequence under new probability distributions, erasing the original n-gram hash patterns. Reasoning tokens are automatically stripped using `processor.parse_response()`.
3. **Secondary Token Perturbation Attack** — Applies optional secondary transformations (synonym swapping and zero-width/filler token insertion) to disrupt any residual structural patterns.
4. **Post-Attack Verification & Quantification** — Re-evaluates the transformed text with the `WatermarkDetector` to confirm the G-value has dropped to baseline noise level (~0.49–0.51) and outputs the percentage reduction alongside a clean/watermarked verdict.

The entire process executes in ~5-15 seconds on GPU and produces a fully sanitized, un-watermarked text with quantitative verification.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla HTML5, Claymorphism CSS (soft 3D depth, dual drop/inset shadows), JavaScript (ES6+ Fetch API) |
| **Backend** | FastAPI, Python 3.11, PyTorch 2.2+, Uvicorn |
| **ML / LLM** | `google/gemma-4-E2B-it` via Hugging Face `AutoModelForImageTextToText` & `AutoProcessor` (`bfloat16`, `device_map="auto"`) |
| **Watermark Engine** | `synthid-text`, GPT-2 Tokenizer (`transformers`), `WatermarkDetector` |
| **UI Theme** | **Claymorphism** (Inflated 3D rounded cards, soft dual inner & outer shadows, pastel/dark clay palettes, tactile pill controls) |
| **Infrastructure** | RunPod / Single GPU (T4 or A10G VRAM >= 12GB), Docker |

---

## Recent Changes (July 2026)

### Gemma 4 E2B API Adaptation
- **Model Class Switch**: Upgraded paraphrasing pipeline from legacy `AutoModelForCausalLM` → `AutoModelForImageTextToText` to support Gemma 4 E2B.
- **Processor Integration**: Replaced standard tokenizer with `AutoProcessor`.
- **Thinking Mode Support**: Configured `enable_thinking=True` during generation and integrated `processor.parse_response()` to strip `<thought>` reasoning tags automatically.
- **Optimized Sampling Parameters**: Updated sampling config to Gemma 4 recommended defaults (`temperature=1.0`, `top_p=0.95`, `top_k=64`).

### Claymorphism UI Overhaul
- **3D Clay Design System**: Implemented a modern Claymorphism UI using pure CSS featuring inflated card containers, rounded corners (`border-radius: 16px`), and soft dual shadows (`box-shadow: inset ...` combined with subtle drop shadows).
- **Tactile Pill Controls**: Form inputs, dropdowns, and execution buttons styled with inflated 3D clay aesthetics and active press feedback (`transform: translateY(2px)`).
- **Responsive G-Value Gauge**: Visual pre/post G-value comparison bar with color-coded signal badges (🔴 Watermarked vs 🟢 Clean) and percentage reduction counters.

### FastAPI Async Lifespan Loader
- **Singleton Model Management**: Implemented `model_loader.py` using FastAPI's `@asynccontextmanager` lifespan event to load Gemma 4 weights once on server startup.
- **API Endpoints**: Built `/remove-watermark` POST route with Pydantic contract validation and `/health` diagnostic endpoint.

---

## Previous Changes

### Core Pipeline & Detection Setup
- **`WatermarkDetector` Wrapper**: Integrated `synthid-text` detection engine paired with GPT-2 tokenizer (`tokenizer_name="gpt2"`) for calculating mean G-values.
- **Multi-Attack Orchestration**: Created `pipeline.py` to chain `Gemma4ParaphrasingAttack` and `TokenPerturbationAttack` seamlessly.
- **CLI Validation Suite**: Developed standalone test scripts (`test_pipeline.py`) to verify G-value drop on sample watermarked texts prior to API deployment.

---

## Issues We Faced During Development

### 1. `AutoModelForCausalLM` vs `AutoModelForImageTextToText` Mismatch
- **Problem**: Gemma 4 E2B threw architecture initialization errors when loaded with the traditional `AutoModelForCausalLM` class used in older Gemma models.
- **Fix**: Updated `Gemma4ParaphrasingAttack` to use `AutoModelForImageTextToText` and paired it with `AutoProcessor` as required by the Gemma 4 specification.

### 2. Thinking Tokens Polluting Paraphrased Output
- **Problem**: When `enable_thinking=True` was enabled for higher paraphrase quality, Gemma 4 emitted internal reasoning blocks (`<thought>...</thought>`) directly into the final text.
- **Fix**: Wrapped model decoding with `processor.parse_response(decoded_text)`, which cleanly filters out reasoning steps, returning only the final rewritten output.

### 3. Missing GPT-2 Tokenizer Dependency for Detection
- **Problem**: `WatermarkDetector` failed when attempting to use Gemma's processor for token probability hashing.
- **Fix**: Separated the detection and paraphrasing components: `WatermarkDetector` explicitly loads `gpt2` tokenizer for G-value scoring while Gemma 4 uses its own native `AutoProcessor`.

### 4. GPU Out of Memory (OOM) on 16GB VRAM GPUs
- **Problem**: Loading Gemma 4 E2B in `float32` exceeded 20GB VRAM, causing CUDA OOM crashes during inference.
- **Fix**: Enforced `dtype=torch.bfloat16` and `device_map="auto"`, reducing memory footprint to ~10–12GB VRAM.

### 5. High Latency During Per-Request Model Re-initialization
- **Problem**: Early implementations instantiated the Gemma 4 model inside the API request handler, adding 30+ seconds to every request.
- **Fix**: Built a singleton loader in `model_loader.py` triggered during FastAPI app lifespan startup, allowing fast sub-10s request processing.

### 6. Claymorphism Shadow Blur & Contrast Issues
- **Problem**: Layered clay shadows (`inset` + outer drop-shadows) created blurry visual noise and poor text readability on dark themes.
- **Fix**: Standardized light sources in CSS using structured variables (`--clay-bg`, `--clay-shadow-light`, `--clay-shadow-dark`) and added crisp contrast borders.

### 7. Native System Role Chat Template Requirement
- **Problem**: Older Gemma 2 models failed when system prompts were included in the chat template array.
- **Fix**: Leveraged Gemma 4's native system role support in `messages = [{"role": "system", ...}, {"role": "user", ...}]`.

### 8. Sub-optimal Paraphrasing with Conservative Sampling
- **Problem**: Low temperature sampling (0.7) led to minimal rephrasing, leaving original n-gram hash sequences intact and keeping G-values above the detection threshold (>0.55).
- **Fix**: Adjusted generation parameters to Gemma 4 recommended values: `temperature=1.0, top_p=0.95, top_k=64`, achieving maximum token sequence diversity.

### 9. CORS & Client Timeout on Long Text Generation
- **Problem**: Processing large paragraphs (500+ words) caused browser fetch calls to time out after 10 seconds.
- **Fix**: Added explicit FastAPI `CORSMiddleware` and updated frontend `fetch()` logic with extended timeout bounds and an animated clay loading spinner.

### 10. Unicode Character Stripping in Token Perturbation
- **Problem**: Zero-width whitespace characters injected during `WhitespaceAttack` were automatically stripped by HTML form inputs.
- **Fix**: Used explicit Unicode escaping (`\u200B`, `\u200C`) and enforced `UTF-8` encoding across FastAPI response headers.

---

## Current Known Issues

### 🟡 High VRAM Footprint
Requires a dedicated GPU with at least 12GB VRAM (T4 / A10G / RunPod). Running on CPU is supported as fallback but takes ~60–120 seconds per request.

### 🟡 Cold Boot Loading Duration
Server startup requires ~30 seconds to load Gemma 4 weights into GPU memory before serving the first API request.

---

## Features We Can Add

### High Priority
- [ ] **Batch Processing Mode** — Drag-and-drop `.txt` or `.json` files to detect and sanitize watermarks from multiple documents in bulk.
- [ ] **Streaming Token Output (SSE)** — Implement Server-Sent Events to stream paraphrased text to the claymorphism UI token-by-token.

### Medium Priority
- [ ] **Interactive Attack Tuning** — UI sliders to dynamically adjust temperature, top_p, top_k, and token substitution rates.
- [ ] **PDF & Markdown Report Export** — Export before/after comparison reports featuring G-value reduction graphs as downloadable PDFs or MD files.
- [ ] **Multi-Model Paraphrase Selector** — Allow switching between Gemma 4 E2B, Gemma 2B, or Llama models for paraphrasing benchmark comparisons.

### Nice to Have
- [ ] **Multi-Scheme Watermark Detection** — Support detection for non-SynthID watermarking schemes (KGW, Unigram, distribution shift).
- [ ] **One-Click Cloud Deployment Script** — Automated Docker / RunPod deployment scripts.
- [ ] **API Key Authentication & Quota Management** — User authentication and rate limiting for hosted multi-user environments.

### Completed Features
- [x] **Core Paraphrase Engine** — Gemma 4 E2B integration using `AutoModelForImageTextToText` & `AutoProcessor`.
- [x] **Thinking Token Filtering** — Automatic reasoning tag stripping via `parse_response()`.
- [x] **G-Value Quantification** — Quantitative pre/post G-value scoring and % watermark reduction calculation.
- [x] **Claymorphism UI Design** — Tactile 3D clay aesthetic with inflated card containers and interactive controls.
- [x] **FastAPI Backend Architecture** — REST API with `/remove-watermark` and `/health` endpoints.
- [x] **Singleton Model Loader** — Non-blocking startup weight loading with FastAPI lifespan.

---

*Last updated: July 2026*
