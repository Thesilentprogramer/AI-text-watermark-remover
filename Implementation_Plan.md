# Implementation Plan — SynthID Watermark Remover
> What to build, in what order, with exact code structure for each piece.

---

## Before You Start — Folder Structure

Set this up on day 1 and never change it:

```
synthid-remover/
│
├── core/                        ← your original work lives here
│   ├── sanitizer.py             ← unicode attack (NEW)
│   ├── backtranslate.py         ← back-translation attack (NEW)
│   ├── entropy.py               ← linguistic entropy layer (NEW)
│   ├── gemma4_paraphrase.py     ← rewritten for Gemma 4 API (ADAPTED)
│   ├── detector.py              ← wraps repo detector + adds perplexity (EXTENDED)
│   ├── selector.py              ← confidence-based attack selector (NEW)
│   └── pipeline.py              ← orchestrates everything (NEW)
│
├── reverse_synthid/             ← cloned repo, touch nothing inside
│   └── reverse_synthid.py
│
├── api/
│   ├── main.py                  ← FastAPI app
│   ├── schemas.py               ← Pydantic models
│   └── routes.py                ← endpoint handlers
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── test_sanitizer.py
│   ├── test_detector.py
│   ├── test_pipeline.py
│   └── test_selector.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Phase 0 — Environment Setup (Day 1)

### 0.1 Install Python deps

```bash
pip install transformers torch accelerate synthid-text
pip install fastapi uvicorn pydantic python-dotenv requests
pip install sentencepiece sacremoses   # for back-translation
pip install nltk                        # for linguistic entropy
```

### 0.2 Clone the base repo

```bash
git clone https://github.com/aloshdenny/reverse-SynthID-text
cp reverse-SynthID-text/reverse_synthid.py ./reverse_synthid/reverse_synthid.py
```

### 0.3 Set up .env

```bash
# .env
GOOGLE_API_KEY=your_google_ai_studio_key
HF_TOKEN=your_huggingface_token
```

### 0.4 Set up .gitignore

```
.env
__pycache__/
*.pyc
.DS_Store
model_cache/
```

### 0.5 Verify base repo works

```python
# run this once to confirm the repo's detector loads correctly
from reverse_synthid.reverse_synthid import WatermarkDetector
detector = WatermarkDetector()
score, info = detector.compute_score("This is a test sentence to check the detector works.")
print(score, info)
```

If this runs without error, your base is good. Move to Phase 1.

---

## Phase 1 — Unicode Sanitizer (Day 2)

**File:** `core/sanitizer.py`

**What it does:** Strips hidden character-level watermarks before any other attack runs. Targets zero-width spaces, joiners, and other invisible unicode that ChatGPT-style models inject.

**Why it's new:** The repo has zero unicode handling. This is your first original contribution.

```python
# core/sanitizer.py

import re
import unicodedata
from dataclasses import dataclass

# every known invisible / zero-width unicode character
INVISIBLE_CHARS = [
    '\u200B',  # zero-width space
    '\u200C',  # zero-width non-joiner
    '\u200D',  # zero-width joiner
    '\u200E',  # left-to-right mark
    '\u200F',  # right-to-left mark
    '\u202F',  # narrow no-break space
    '\u2060',  # word joiner
    '\u2061',  # function application
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\uFEFF',  # byte order mark / zero-width no-break space
    '\u00AD',  # soft hyphen
    '\u180E',  # mongolian vowel separator
    '\u00A0',  # non-breaking space (replace with regular space)
]

@dataclass
class SanitizeResult:
    original_text: str
    clean_text: str
    chars_removed: int
    anomaly_score: float   # 0.0 = clean, 1.0 = heavily injected
    found_chars: list

class UnicodeSanitizer:

    def analyze(self, text: str) -> dict:
        """
        Scan text for hidden characters without modifying it.
        Returns anomaly score used by the selector later.
        """
        found = []
        for char in INVISIBLE_CHARS:
            count = text.count(char)
            if count > 0:
                found.append({"char": repr(char), "count": count})

        total_invisible = sum(f["count"] for f in found)
        # anomaly score: ratio of invisible chars to total chars, capped at 1.0
        anomaly_score = min(total_invisible / max(len(text), 1) * 100, 1.0)

        return {
            "found_chars": found,
            "total_invisible": total_invisible,
            "anomaly_score": round(anomaly_score, 4),
            "is_suspicious": anomaly_score > 0.01
        }

    def sanitize(self, text: str) -> SanitizeResult:
        """
        Remove all hidden characters and normalize unicode.
        """
        analysis = self.analyze(text)

        clean = text
        for char in INVISIBLE_CHARS:
            if char == '\u00A0':
                clean = clean.replace(char, ' ')   # replace NBSP with normal space
            else:
                clean = clean.replace(char, '')     # remove entirely

        # normalize to NFC — canonical composed form
        clean = unicodedata.normalize("NFC", clean)

        # collapse multiple spaces that may appear after removal
        clean = re.sub(r' {2,}', ' ', clean)
        clean = clean.strip()

        return SanitizeResult(
            original_text=text,
            clean_text=clean,
            chars_removed=analysis["total_invisible"],
            anomaly_score=analysis["anomaly_score"],
            found_chars=analysis["found_chars"]
        )
```

**Test it:**

```python
# tests/test_sanitizer.py
from core.sanitizer import UnicodeSanitizer

s = UnicodeSanitizer()

# inject some hidden chars manually
dirty = "This is\u200B a\u200D test\u202F sentence."
result = s.sanitize(dirty)

assert result.chars_removed == 3
assert "\u200B" not in result.clean_text
assert result.anomaly_score > 0
print("Sanitizer test passed:", result.chars_removed, "chars removed")
```

---

## Phase 2 — Extended Detector (Day 3)

**File:** `core/detector.py`

**What it does:** Wraps the repo's `WatermarkDetector` AND adds a second detection method — perplexity scoring. This is key because the repo only detects SynthID. You are adding detection of perplexity-based watermarks (GPTZero style) giving you dual-mode evaluation.

**Why it's new:** The repo has one detector. You have two. Your tool now verifies itself against two completely different watermarking paradigms.

```python
# core/detector.py

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from reverse_synthid.reverse_synthid import WatermarkDetector as RepoDetector

class ExtendedDetector:

    def __init__(self):
        self._repo_detector = RepoDetector()

        # GPT-2 for perplexity scoring — tiny model, CPU fine
        print("Loading GPT-2 for perplexity scoring...")
        self._ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._ppl_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self._ppl_model.eval()
        print("Done.")

    def _compute_perplexity(self, text: str) -> float:
        """
        Lower perplexity = more predictable = more likely AI generated.
        Human text: perplexity ~100-300
        AI text:    perplexity ~20-60
        """
        inputs = self._ppl_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self._ppl_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return round(math.exp(loss.item()), 2)

    def _perplexity_label(self, ppl: float) -> dict:
        if ppl < 50:
            return {"label": "likely AI", "confidence": "high"}
        elif ppl < 100:
            return {"label": "possibly AI", "confidence": "medium"}
        else:
            return {"label": "likely human", "confidence": "high"}

    def score(self, text: str) -> dict:
        """
        Run both detectors and return combined result.
        """
        # SynthID G-value (from repo)
        g_value, synthid_info = self._repo_detector.compute_score(text)

        # Perplexity (your addition)
        ppl = self._compute_perplexity(text)
        ppl_label = self._perplexity_label(ppl)

        return {
            "synthid": {
                "g_value": round(g_value, 4),
                "is_watermarked": synthid_info.get("likely_watermarked", g_value > 0.55),
                "threshold": 0.55,
            },
            "perplexity": {
                "score": ppl,
                "interpretation": ppl_label["label"],
                "confidence": ppl_label["confidence"],
            },
            "verdict": "watermarked" if (
                g_value > 0.55 or ppl < 60
            ) else "clean"
        }
```

**Test it:**

```python
# tests/test_detector.py
from core.detector import ExtendedDetector

d = ExtendedDetector()

# test on obviously human text
result = d.score("The sun rose slowly over the mountains as birds began to sing.")
print("G-value:", result["synthid"]["g_value"])
print("Perplexity:", result["perplexity"]["score"])
print("Verdict:", result["verdict"])
```

---

## Phase 3 — Back-Translation Attack (Day 4)

**File:** `core/backtranslate.py`

**What it does:** Translates text to German then back to English using Helsinki-NLP models from HuggingFace. This completely resets the token sequence — no API cost, no GPU needed, works on CPU.

**Why it's new:** The repo has no translation-based attack at all. This is your second new attack method.

```python
# core/backtranslate.py

from transformers import MarianMTModel, MarianTokenizer
from dataclasses import dataclass

@dataclass
class BackTranslateResult:
    original_text: str
    intermediate_text: str    # German
    final_text: str           # back to English
    pivot_language: str

class BackTranslationAttack:

    MODELS = {
        "de": {
            "forward": "Helsinki-NLP/opus-mt-en-de",   # English → German
            "backward": "Helsinki-NLP/opus-mt-de-en",  # German → English
        },
        "fr": {
            "forward": "Helsinki-NLP/opus-mt-en-fr",
            "backward": "Helsinki-NLP/opus-mt-fr-en",
        },
        "es": {
            "forward": "Helsinki-NLP/opus-mt-en-es",
            "backward": "Helsinki-NLP/opus-mt-es-en",
        },
    }

    def __init__(self, pivot_lang: str = "de"):
        assert pivot_lang in self.MODELS, f"Unsupported pivot language: {pivot_lang}"
        self.pivot_lang = pivot_lang
        model_ids = self.MODELS[pivot_lang]

        print(f"Loading translation models (pivot: {pivot_lang})...")
        self._fwd_tok = MarianTokenizer.from_pretrained(model_ids["forward"])
        self._fwd_model = MarianMTModel.from_pretrained(model_ids["forward"])
        self._bwd_tok = MarianTokenizer.from_pretrained(model_ids["backward"])
        self._bwd_model = MarianMTModel.from_pretrained(model_ids["backward"])
        print("Translation models loaded.")

    def _translate(self, text: str, tokenizer, model) -> str:
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = model.generate(**inputs, num_beams=4, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _chunk_text(self, text: str, max_words: int = 80) -> list:
        """
        Split long text into chunks to avoid translation model length limits.
        """
        words = text.split()
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    def attack(self, text: str) -> BackTranslateResult:
        chunks = self._chunk_text(text)

        intermediate_chunks = []
        final_chunks = []

        for chunk in chunks:
            # English → pivot language
            intermediate = self._translate(chunk, self._fwd_tok, self._fwd_model)
            intermediate_chunks.append(intermediate)

            # pivot language → English
            final = self._translate(intermediate, self._bwd_tok, self._bwd_model)
            final_chunks.append(final)

        return BackTranslateResult(
            original_text=text,
            intermediate_text=" ".join(intermediate_chunks),
            final_text=" ".join(final_chunks),
            pivot_language=self.pivot_lang
        )
```

**Test it:**

```python
# quick test
from core.backtranslate import BackTranslationAttack

bt = BackTranslationAttack(pivot_lang="de")
result = bt.attack("Artificial intelligence is transforming the way we work and communicate.")
print("Original: ", result.original_text)
print("German:   ", result.intermediate_text)
print("Back:     ", result.final_text)
```

---

## Phase 4 — Linguistic Entropy Layer (Day 5)

**File:** `core/entropy.py`

**What it does:** Two things the repo's `TokenPerturbationAttack` doesn't do — sentence length variation and passive/active voice shifting. These specifically target perplexity-based detectors by making the text feel less uniformly structured.

**Why it's new:** Repo only swaps synonyms. You are adding structural variation that defeats a completely different class of detector.

```python
# core/entropy.py

import re
import random
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)

class LinguisticEntropyLayer:

    def __init__(self, seed: int = 42):
        random.seed(seed)

    # ── Sentence Length Variation ─────────────────────────────────────────

    def _split_long_sentence(self, sentence: str) -> list:
        """
        Split sentences over 25 words at a conjunction or comma.
        """
        words = sentence.split()
        if len(words) < 25:
            return [sentence]

        connectors = [" and ", " but ", " which ", " because ", " however "]
        for connector in connectors:
            if connector in sentence.lower():
                idx = sentence.lower().index(connector)
                left = sentence[:idx].strip()
                right = sentence[idx + len(connector):].strip()
                if left and right:
                    right = right[0].upper() + right[1:]
                    return [left + ".", right]

        # fallback: split at comma
        if ", " in sentence:
            parts = sentence.split(", ", 1)
            return [parts[0] + ".", parts[1][0].upper() + parts[1][1:]]

        return [sentence]

    def _merge_short_sentences(self, sentences: list) -> list:
        """
        Merge consecutive short sentences (under 6 words) with a comma.
        """
        result = []
        i = 0
        while i < len(sentences):
            if (i + 1 < len(sentences) and
                len(sentences[i].split()) < 6 and
                len(sentences[i + 1].split()) < 6):
                merged = sentences[i].rstrip(".") + ", " + sentences[i + 1].lstrip()
                result.append(merged)
                i += 2
            else:
                result.append(sentences[i])
                i += 1
        return result

    def vary_sentence_lengths(self, text: str) -> str:
        sentences = nltk.sent_tokenize(text)

        # split long ones
        expanded = []
        for s in sentences:
            expanded.extend(self._split_long_sentence(s))

        # merge short ones
        merged = self._merge_short_sentences(expanded)

        return " ".join(merged)

    # ── Synonym Swap (extends repo's basic version) ───────────────────────

    def _get_synonym(self, word: str, pos_tag: str) -> str:
        pos_map = {"NN": wordnet.NOUN, "VB": wordnet.VERB,
                   "JJ": wordnet.ADJ, "RB": wordnet.ADV}
        wn_pos = pos_map.get(pos_tag[:2])
        if not wn_pos:
            return word

        synsets = wordnet.synsets(word, pos=wn_pos)
        candidates = []
        for syn in synsets:
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    candidates.append(candidate)

        return random.choice(candidates) if candidates else word

    def swap_synonyms(self, text: str, rate: float = 0.15) -> str:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        result = []

        for word, tag in tagged:
            if (random.random() < rate and
                tag[:2] in ["NN", "VB", "JJ", "RB"] and
                len(word) > 3):
                result.append(self._get_synonym(word, tag))
            else:
                result.append(word)

        # detokenize — rejoin punctuation properly
        text_out = " ".join(result)
        text_out = re.sub(r' ([.,!?;:])', r'\1', text_out)
        return text_out

    # ── Combined Apply ────────────────────────────────────────────────────

    def apply(self, text: str, synonym_rate: float = 0.15) -> str:
        text = self.vary_sentence_lengths(text)
        text = self.swap_synonyms(text, rate=synonym_rate)
        return text
```

---

## Phase 5 — Gemma 4 Paraphrasing via API (Day 6)

**File:** `core/gemma4_paraphrase.py`

**What it does:** Calls Google AI Studio API to paraphrase text using Gemma 3 (or Gemma 4 when available). This replaces the repo's local model loading with an API call, making it GPU-free.

```python
# core/gemma4_paraphrase.py

import os
import requests
from dataclasses import dataclass

@dataclass
class ParaphraseResult:
    original_text: str
    paraphrased_text: str
    model_used: str
    thinking_used: bool

class GemmaAPIParaphrase:

    GOOGLE_API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )

    def __init__(self, model: str = "gemma-3-27b-it"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        assert self.api_key, "GOOGLE_API_KEY not set in .env"
        self.model = model

    def _build_prompt(self, text: str) -> str:
        return (
            "Rewrite the following text completely in your own words. "
            "Preserve the exact meaning but change the sentence structure, "
            "word choices, and phrasing entirely. "
            "Output only the rewritten text, no explanation, no preamble.\n\n"
            f"{text}"
        )

    def paraphrase(self, text: str) -> ParaphraseResult:
        url = self.GOOGLE_API_URL.format(model=self.model)
        params = {"key": self.api_key}

        payload = {
            "contents": [{
                "parts": [{"text": self._build_prompt(text)}]
            }],
            "generationConfig": {
                "temperature": 1.0,
                "topP": 0.95,
                "topK": 64,
                "maxOutputTokens": 2048,
            }
        }

        response = requests.post(url, params=params, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        paraphrased = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        return ParaphraseResult(
            original_text=text,
            paraphrased_text=paraphrased,
            model_used=self.model,
            thinking_used=False
        )
```

---

## Phase 6 — Confidence-Based Attack Selector (Day 7)

**File:** `core/selector.py`

**What it does:** Automatically decides which attack(s) to run based on the input text's characteristics. This is the most original part of the whole project — no existing tool does this.

**Decision logic:**

```
Short text + low anomaly + moderate G   → back-translate only (fast)
Any text + high unicode anomaly         → sanitize first, then perturb
Long text + high G-value               → full paraphrase
Very high G-value (>0.75)              → combined (all layers)
Moderate G-value (0.55-0.65)           → back-translate + entropy
```

```python
# core/selector.py

from dataclasses import dataclass

@dataclass
class AttackPlan:
    attack_mode: str
    reasoning: str
    estimated_time: str
    layers: list   # ordered list of attack names to run

class AttackSelector:

    def select(self, text: str, pre_detection: dict, anomaly_score: float) -> AttackPlan:
        """
        Given the text and pre-detection scores, decide which attacks to run.

        pre_detection: output from ExtendedDetector.score()
        anomaly_score: output from UnicodeSanitizer.analyze()["anomaly_score"]
        """
        g_value = pre_detection["synthid"]["g_value"]
        perplexity = pre_detection["perplexity"]["score"]
        word_count = len(text.split())

        # Rule 1: High unicode anomaly → sanitize first regardless of G-value
        if anomaly_score > 0.05:
            if g_value < 0.55:
                return AttackPlan(
                    attack_mode="sanitize_only",
                    reasoning="High unicode anomaly detected, G-value below threshold. Character-level watermark suspected.",
                    estimated_time="< 1 second",
                    layers=["sanitize"]
                )
            else:
                return AttackPlan(
                    attack_mode="sanitize_then_paraphrase",
                    reasoning="High unicode anomaly AND high G-value. Applying full pipeline.",
                    estimated_time="10–20 seconds",
                    layers=["sanitize", "paraphrase", "entropy"]
                )

        # Rule 2: Very high G-value → full combined attack
        if g_value > 0.75:
            return AttackPlan(
                attack_mode="combined",
                reasoning=f"Very high G-value ({g_value}). Strong watermark detected. Running all attack layers.",
                estimated_time="15–25 seconds",
                layers=["sanitize", "paraphrase", "entropy"]
            )

        # Rule 3: Short text + moderate G-value → back-translate (fast, sufficient)
        if word_count < 150 and 0.55 <= g_value <= 0.65:
            return AttackPlan(
                attack_mode="backtranslate",
                reasoning=f"Short text ({word_count} words), moderate G-value ({g_value}). Back-translation sufficient.",
                estimated_time="5–10 seconds",
                layers=["sanitize", "backtranslate"]
            )

        # Rule 4: Long text + moderate G-value → paraphrase + entropy
        if word_count >= 150 and 0.55 <= g_value <= 0.75:
            return AttackPlan(
                attack_mode="paraphrase_entropy",
                reasoning=f"Long text ({word_count} words), moderate G-value ({g_value}). Paraphrase with entropy layer.",
                estimated_time="10–20 seconds",
                layers=["sanitize", "paraphrase", "entropy"]
            )

        # Rule 5: Low perplexity only (perplexity-based watermark, G-value clean)
        if perplexity < 60 and g_value < 0.55:
            return AttackPlan(
                attack_mode="entropy_only",
                reasoning=f"Low perplexity ({perplexity}), normal G-value. Targeting perplexity-based detection.",
                estimated_time="2–5 seconds",
                layers=["entropy"]
            )

        # Default: text appears clean
        return AttackPlan(
            attack_mode="none",
            reasoning=f"G-value ({g_value}) and perplexity ({perplexity}) both within normal range. Text appears clean.",
            estimated_time="0 seconds",
            layers=[]
        )
```

---

## Phase 7 — Main Pipeline (Day 8)

**File:** `core/pipeline.py`

**What it does:** Orchestrates every component in the right order using the selector's decision.

```python
# core/pipeline.py

import time
from core.sanitizer import UnicodeSanitizer
from core.detector import ExtendedDetector
from core.selector import AttackSelector
from core.backtranslate import BackTranslationAttack
from core.entropy import LinguisticEntropyLayer
from core.gemma4_paraphrase import GemmaAPIParaphrase

class WatermarkRemovalPipeline:

    def __init__(self):
        print("Initializing pipeline...")
        self.sanitizer   = UnicodeSanitizer()
        self.detector    = ExtendedDetector()
        self.selector    = AttackSelector()
        self.backtrans   = BackTranslationAttack(pivot_lang="de")
        self.entropy     = LinguisticEntropyLayer()
        self.paraphraser = GemmaAPIParaphrase()
        print("Pipeline ready.")

    def run(self, text: str) -> dict:
        start = time.time()

        # ── Step 1: Pre-scan ──────────────────────────────────────────────
        anomaly = self.sanitizer.analyze(text)
        pre_detection = self.detector.score(text)

        # ── Step 2: Select attack plan ────────────────────────────────────
        plan = self.selector.select(text, pre_detection, anomaly["anomaly_score"])

        # ── Step 3: Execute layers in order ──────────────────────────────
        current_text = text

        for layer in plan.layers:

            if layer == "sanitize":
                result = self.sanitizer.sanitize(current_text)
                current_text = result.clean_text

            elif layer == "paraphrase":
                result = self.paraphraser.paraphrase(current_text)
                current_text = result.paraphrased_text

            elif layer == "backtranslate":
                result = self.backtrans.attack(current_text)
                current_text = result.final_text

            elif layer == "entropy":
                current_text = self.entropy.apply(current_text)

        # ── Step 4: Post-detection ────────────────────────────────────────
        post_detection = self.detector.score(current_text)

        elapsed = round((time.time() - start) * 1000)

        return {
            "original_text": text,
            "clean_text": current_text,
            "attack_plan": {
                "mode": plan.attack_mode,
                "layers_applied": plan.layers,
                "reasoning": plan.reasoning,
            },
            "pre_attack": {
                "g_value": pre_detection["synthid"]["g_value"],
                "perplexity": pre_detection["perplexity"]["score"],
                "verdict": pre_detection["verdict"],
                "unicode_anomaly": anomaly["anomaly_score"],
            },
            "post_attack": {
                "g_value": post_detection["synthid"]["g_value"],
                "perplexity": post_detection["perplexity"]["score"],
                "verdict": post_detection["verdict"],
            },
            "g_value_reduction": round(
                pre_detection["synthid"]["g_value"] - post_detection["synthid"]["g_value"], 4
            ),
            "processing_time_ms": elapsed,
        }
```

---

## Phase 8 — FastAPI Layer (Day 9)

**File:** `api/main.py`, `api/schemas.py`, `api/routes.py`

```python
# api/schemas.py
from pydantic import BaseModel

class RemoveRequest(BaseModel):
    text: str

class DetectionScore(BaseModel):
    g_value: float
    perplexity: float
    verdict: str

class RemoveResponse(BaseModel):
    original_text: str
    clean_text: str
    attack_plan: dict
    pre_attack: dict
    post_attack: dict
    g_value_reduction: float
    processing_time_ms: int
```

```python
# api/routes.py
from fastapi import APIRouter, HTTPException
from api.schemas import RemoveRequest, RemoveResponse
from core.pipeline import WatermarkRemovalPipeline

router = APIRouter()
_pipeline: WatermarkRemovalPipeline = None

def get_pipeline() -> WatermarkRemovalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = WatermarkRemovalPipeline()
    return _pipeline

@router.post("/remove-watermark", response_model=RemoveResponse)
async def remove_watermark(req: RemoveRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 chars)")
    try:
        result = get_pipeline().run(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}
```

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import router

app = FastAPI(title="SynthID Watermark Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

**Run it:**

```bash
uvicorn api.main:app --reload --port 8000
```

---

## Phase 9 — Frontend (Day 10)

**File:** `frontend/index.html`

Keep it minimal. Three sections: input, scores, output.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SynthID Watermark Remover</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <h1>SynthID Watermark Remover</h1>
    <p>Detects and removes AI text watermarks using a layered attack pipeline</p>
  </header>

  <main>
    <!-- Input -->
    <section id="input-section">
      <label>Paste watermarked text</label>
      <textarea id="input-text" rows="10"
        placeholder="Paste AI-generated text here..."></textarea>
      <button id="run-btn" onclick="runAttack()">Run Attack</button>
    </section>

    <!-- Loading -->
    <div id="loading" class="hidden">
      <div class="spinner"></div>
      <p id="loading-msg">Analyzing text...</p>
    </div>

    <!-- Results -->
    <section id="results" class="hidden">

      <!-- Attack Plan -->
      <div class="card" id="plan-card">
        <h3>Attack Plan Selected</h3>
        <p id="plan-mode"></p>
        <p id="plan-reason" class="muted"></p>
      </div>

      <!-- Scores -->
      <div class="card" id="scores-card">
        <h3>Detection Scores</h3>
        <div class="score-row">
          <span>G-value before</span>
          <span id="g-before" class="score bad"></span>
        </div>
        <div class="score-row">
          <span>G-value after</span>
          <span id="g-after" class="score good"></span>
        </div>
        <div class="score-row">
          <span>Perplexity before</span>
          <span id="ppl-before" class="score"></span>
        </div>
        <div class="score-row">
          <span>Perplexity after</span>
          <span id="ppl-after" class="score"></span>
        </div>
        <div class="progress-bar">
          <div id="reduction-bar"></div>
        </div>
        <p id="reduction-label" class="muted"></p>
      </div>

      <!-- Output -->
      <div class="card" id="output-card">
        <h3>Clean Text</h3>
        <div id="output-text"></div>
        <button onclick="copyOutput()">Copy</button>
      </div>

    </section>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

```javascript
// frontend/app.js

async function runAttack() {
  const text = document.getElementById("input-text").value.trim();
  if (!text) return alert("Please paste some text first.");

  setLoading(true, "Analyzing text...");

  try {
    const res = await fetch("/remove-watermark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderResults(data);

  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  document.getElementById("results").classList.remove("hidden");

  // attack plan
  document.getElementById("plan-mode").textContent =
    "Mode: " + data.attack_plan.mode + " — Layers: " + data.attack_plan.layers_applied.join(" → ");
  document.getElementById("plan-reason").textContent = data.attack_plan.reasoning;

  // scores
  document.getElementById("g-before").textContent = data.pre_attack.g_value;
  document.getElementById("g-after").textContent  = data.post_attack.g_value;
  document.getElementById("ppl-before").textContent = data.pre_attack.perplexity;
  document.getElementById("ppl-after").textContent  = data.post_attack.perplexity;

  const reduction = Math.round(data.g_value_reduction / data.pre_attack.g_value * 100);
  document.getElementById("reduction-bar").style.width = reduction + "%";
  document.getElementById("reduction-label").textContent =
    `G-value reduced by ${reduction}% in ${data.processing_time_ms}ms`;

  // output
  document.getElementById("output-text").textContent = data.clean_text;
}

function setLoading(show, msg = "") {
  document.getElementById("loading").classList.toggle("hidden", !show);
  document.getElementById("loading-msg").textContent = msg;
  document.getElementById("run-btn").disabled = show;
}

function copyOutput() {
  const text = document.getElementById("output-text").textContent;
  navigator.clipboard.writeText(text);
}
```

---

## Phase 10 — Testing & README (Days 11–12)

### Run the full pipeline test

```python
# tests/test_pipeline.py
from core.pipeline import WatermarkRemovalPipeline

p = WatermarkRemovalPipeline()

sample = """
Artificial intelligence is rapidly transforming industries across the globe.
From healthcare to finance, machine learning models are being deployed at scale.
The implications for the workforce are significant and multifaceted.
"""

result = p.run(sample)
print("Attack mode:", result["attack_plan"]["mode"])
print("Layers:", result["attack_plan"]["layers_applied"])
print("G before:", result["pre_attack"]["g_value"])
print("G after:", result["post_attack"]["g_value"])
print("Reduction:", result["g_value_reduction"])
print("Time:", result["processing_time_ms"], "ms")
```

### README sections to write

1. What SynthID is and how it works at the token level
2. The three watermark types this tool targets
3. The confidence-based selector — explain the decision logic
4. Setup instructions
5. API docs
6. Results table — before/after G-values on 5 test samples

---

## Complete Build Order Summary

| Day | Phase | Deliverable |
|---|---|---|
| 1 | Environment setup | Repo cloned, deps installed, detector verified |
| 2 | Sanitizer | `core/sanitizer.py` tested |
| 3 | Extended detector | `core/detector.py` with dual scoring |
| 4 | Back-translation | `core/backtranslate.py` tested |
| 5 | Entropy layer | `core/entropy.py` tested |
| 6 | Gemma API | `core/gemma4_paraphrase.py` tested |
| 7 | Selector | `core/selector.py` logic verified on 5 inputs |
| 8 | Pipeline | `core/pipeline.py` end-to-end tested |
| 9 | FastAPI | Server running, endpoints tested with curl |
| 10 | Frontend | Browser demo working end-to-end |
| 11–12 | Tests + README | Documented and pushed to GitHub |
