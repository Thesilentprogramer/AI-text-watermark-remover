"""
Gemma4ParaphrasingAttack
Rewrites token sequences using google/gemma-4-E2B-it to completely destroy SynthID n-gram hash bias.
Supports AutoModelForImageTextToText local inference, API fallback, and structural fallback.
"""

import os
import re
import logging
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("gemma4_paraphrase")

# Preference order when auto-selecting from ListModels (4b not on all API keys yet).
PREFERRED_GEMMA_API_MODELS = (
    "gemma-4-4b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
)


class Gemma4ParaphrasingAttack:
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or os.getenv("MODEL_ID", "google/gemma-4-E2B-it")
        self.hf_token = os.getenv("HF_TOKEN")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self._configured_gemma_api_model = (
            os.getenv("GEMMA_API_MODEL")
            or os.getenv("GEMINI_API_MODEL")  # legacy alias
        )
        if os.getenv("GEMINI_API_MODEL") and not os.getenv("GEMMA_API_MODEL"):
            logger.warning(
                "GEMINI_API_MODEL is deprecated; use GEMMA_API_MODEL (Gemma 4 only)."
            )
        self._available_gemma_api_models = None
        self.gemma_api_model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None
        self.loaded = False
        self.last_paraphrase_source = "skipped"

    def load_local_model(self):
        """Loads Gemma 4 E2B weights locally using AutoModelForImageTextToText."""
        if self.loaded:
            return

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Loading {self.model_id} on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.hf_token
            )

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                token=self.hf_token,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda":
                self.model.to(self.device)

            self.loaded = True
            logger.info("Gemma 4 model successfully loaded locally.")
        except Exception as e:
            logger.warning(f"Local Gemma 4 model load skipped/failed: {e}")
            self.loaded = False

    def paraphrase(
        self,
        text: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64
    ) -> str:
        """
        Step 3 - ATTACK
        Paraphrases text completely to destroy n-gram hash watermarking patterns.
        """
        if not text or len(text.strip()) == 0:
            self.last_paraphrase_source = "skipped"
            return ""

        backend = os.getenv("PARAPHRASE_BACKEND", "auto").lower()
        if backend == "heuristic":
            self.last_paraphrase_source = "heuristic"
            return self._paraphrase_heuristic_fallback(text)

        # 1. Attempt Local Inference if model loaded
        if self.loaded and self.model and self.processor:
            res = self._paraphrase_local(text, temperature, top_p, top_k)
            if res and len(res) > 0 and res != text:
                self.last_paraphrase_source = "local"
                return res

        # 2. Attempt Gemma 4 API (Google AI Studio free tier) if key is configured
        if backend in ("auto", "api") and self.google_api_key:
            api_res = self._paraphrase_gemma_api(text)
            if api_res and self._is_usable_paraphrase(text, api_res):
                self.last_paraphrase_source = "api" if getattr(self, "_api_paraphrase_used", False) else "heuristic"
                logger.info(
                    "Gemma 4 paraphrase ok (%d -> %d chars, source=%s)",
                    len(text),
                    len(api_res),
                    self.last_paraphrase_source,
                )
                return api_res
            logger.warning(
                "Gemma 4 API paraphrase unusable (in=%d out=%d); using heuristic fallback.",
                len(text),
                len(api_res or ""),
            )

        # 3. Structural fallback (synonym + perturb + shuffle)
        self.last_paraphrase_source = "heuristic"
        return self._paraphrase_heuristic_fallback(text)

    def _paraphrase_local(self, text: str, temperature: float, top_p: float, top_k: int) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert text rewriter. Paraphrase the user's text completely using distinct vocabulary "
                    "and altered sentence structures while preserving the exact original core meaning. "
                    "Output ONLY the final rewritten text."
                )
            },
            {
                "role": "user",
                "content": f"Rewrite the following text:\n\n{text}"
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True
                )

            raw_decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

            if hasattr(self.processor, "parse_response"):
                clean_output = self.processor.parse_response(raw_decoded)
            else:
                clean_output = self._strip_thinking_tags(raw_decoded)

            return clean_output.strip()
        except Exception as e:
            logger.error(f"Error during local Gemma 4 inference: {e}")
            return ""

    def _fetch_available_gemma_api_models(self) -> list:
        """List Gemma 4 models this API key can call via generateContent."""
        if self._available_gemma_api_models is not None:
            return self._available_gemma_api_models

        models = []
        if not self.google_api_key:
            self._available_gemma_api_models = models
            return models

        try:
            import requests

            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": self.google_api_key},
                timeout=15,
            )
            if resp.status_code == 200:
                for entry in resp.json().get("models", []):
                    name = entry.get("name", "").replace("models/", "")
                    methods = entry.get("supportedGenerationMethods", [])
                    if "gemma-4" in name and "generateContent" in methods:
                        models.append(name)
        except Exception as e:
            logger.warning("Could not list Gemma API models: %s", e)

        self._available_gemma_api_models = models
        return models

    def _resolve_gemma_api_model(self) -> str:
        """Pick a Gemma 4 model available on this Gemini API key."""
        if self.gemma_api_model:
            return self.gemma_api_model

        available = self._fetch_available_gemma_api_models()
        configured = self._configured_gemma_api_model

        if configured:
            if configured in available:
                self.gemma_api_model = configured
            else:
                logger.warning(
                    "GEMMA_API_MODEL=%s not available on this API key. Available Gemma 4: %s",
                    configured,
                    ", ".join(available) or "(none)",
                )

        if not self.gemma_api_model:
            for candidate in PREFERRED_GEMMA_API_MODELS:
                if candidate in available:
                    self.gemma_api_model = candidate
                    break

        if not self.gemma_api_model and available:
            self.gemma_api_model = available[0]

        if self.gemma_api_model:
            logger.info("Resolved Gemma 4 API model: %s", self.gemma_api_model)
        else:
            logger.warning("No Gemma 4 models available on this Gemini API key.")

        return self.gemma_api_model

    def _extract_gemma_api_text(self, data: dict) -> str:
        """Pull final answer text from generateContent response (skip thought parts)."""
        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("Gemma 4 API: response has no candidates")
            return ""

        parts = candidates[0].get("content", {}).get("parts") or []
        answer_parts = []
        for part in parts:
            if part.get("thought"):
                continue
            text = part.get("text", "").strip()
            if text:
                answer_parts.append(text)

        if answer_parts:
            return self._strip_thinking_tags(answer_parts[-1]).strip()

        # Model spent all tokens on reasoning — try to salvage a final line from thought text.
        for part in reversed(parts):
            if part.get("thought") and part.get("text"):
                lines = [ln.strip() for ln in part["text"].splitlines() if ln.strip()]
                for line in reversed(lines):
                    if line.startswith(('"', "'", "**", "Rewrite:", "Output:")):
                        continue
                    if len(line) > 40 and not line.startswith("*"):
                        return self._strip_thinking_tags(line).strip()

        logger.warning(
            "Gemma 4 API: no non-thought text parts (parts=%d, finish=%s)",
            len(parts),
            candidates[0].get("finishReason"),
        )
        return ""

    def _is_usable_paraphrase(self, original: str, result: str) -> bool:
        """Reject empty or severely truncated API output (common on long inputs)."""
        if not result or not result.strip():
            return False
        out_len = len(result.strip())
        in_len = len(original.strip())
        if in_len < 400:
            return out_len >= 20
        return out_len >= int(in_len * 0.35)

    def _split_for_api_chunks(self, text: str, max_chunk_chars: int = 1000) -> list:
        """Split long text into API-sized chunks at paragraph/sentence boundaries."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks = []
        current = ""

        def flush():
            nonlocal current
            if current.strip():
                chunks.append(current.strip())
            current = ""

        for para in paragraphs:
            if len(para) > max_chunk_chars:
                flush()
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                buf = ""
                for sentence in sentences:
                    if buf and len(buf) + len(sentence) + 1 > max_chunk_chars:
                        chunks.append(buf.strip())
                        buf = sentence
                    else:
                        buf = f"{buf} {sentence}".strip() if buf else sentence
                if buf:
                    chunks.append(buf.strip())
            elif current and len(current) + len(para) + 2 > max_chunk_chars:
                flush()
                current = para
            else:
                current = f"{current}\n\n{para}".strip() if current else para

        flush()
        return chunks or [text.strip()]

    def _paraphrase_gemma_api(self, text: str) -> str:
        """Gemma 4 via the Gemini API (same GOOGLE_API_KEY / generativelanguage.googleapis.com)."""
        self._api_paraphrase_used = False
        model = self._resolve_gemma_api_model()
        if not model:
            return None

        single_max = int(os.getenv("GEMMA_SINGLE_CALL_MAX_CHARS", "8000"))
        chunk_threshold = int(os.getenv("GEMMA_CHUNK_THRESHOLD_CHARS", "1200"))
        chunking = os.getenv("GEMMA_API_CHUNKING", "auto").lower()

        # Free-tier default: one API call (fast). Chunk only if output is bad or forced.
        if chunking != "always" and len(text) <= single_max:
            result = self._call_gemma_api_once(model, text)
            if result and self._is_usable_paraphrase(text, result):
                self._api_paraphrase_used = True
                logger.info("Gemma 4 single-call paraphrase ok (%d chars)", len(text))
                return result
            if chunking == "never" or len(text) <= chunk_threshold:
                return None
            logger.warning(
                "Single-call paraphrase unusable for %d chars; retrying with chunks.",
                len(text),
            )

        if len(text) > chunk_threshold or chunking == "always":
            parts = []
            for chunk in self._split_for_api_chunks(text):
                rewritten = self._call_gemma_api_once(model, chunk)
                if rewritten and self._is_usable_paraphrase(chunk, rewritten):
                    self._api_paraphrase_used = True
                    parts.append(rewritten.strip())
                else:
                    logger.warning(
                        "Chunk paraphrase failed (%d chars); using heuristic for chunk.",
                        len(chunk),
                    )
                    parts.append(self._paraphrase_heuristic_fallback(chunk).strip())
            combined = "\n\n".join(p for p in parts if p)
            return combined if combined else None

        result = self._call_gemma_api_once(model, text)
        if result:
            self._api_paraphrase_used = True
        return result

    def _call_gemma_api_once(self, model: str, text: str) -> str:
        try:
            import requests

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.google_api_key}"
            )
            payload = {
                "systemInstruction": {
                    "parts": [{
                        "text": (
                            "You are an expert text rewriter. Paraphrase the user's text completely using "
                            "distinct vocabulary and altered sentence structures while preserving the exact "
                            "original core meaning. Output ONLY the final rewritten text with no preamble."
                        )
                    }]
                },
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": f"Rewrite the following text:\n\n{text}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 1.0,
                    "topP": 0.95,
                    "topK": 64,
                    "maxOutputTokens": 8192,
                },
            }

            logger.info("Paraphrasing via Gemma 4 API model: %s (%d chars)", model, len(text))
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                result = self._extract_gemma_api_text(resp.json())
                if result:
                    return result
                logger.warning("Gemma 4 API model=%s returned empty answer text", model)
                return None
            logger.warning(
                "Gemma 4 API error model=%s status=%s body=%s",
                model,
                resp.status_code,
                resp.text[:500],
            )
        except Exception as e:
            logger.warning(f"Gemma 4 API fallback error: {e}")
        return None

    def _strip_thinking_tags(self, text: str) -> str:
        """Strips <thought>...</thought> tags manually if parse_response is unavailable."""
        cleaned = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        return cleaned.replace('<eos>', '').replace('</s>', '').strip()

    def _apply_synonym_map(self, text: str) -> str:
        """Word-level synonym substitution from curated map."""
        synonym_map = {
            'first': 'initial', 'second': 'subsequent', 'third': 'final',
            'greatest': 'most extraordinary', 'legendary': 'iconic', 'miracle': 'triumph',
            'stadium': 'arena', 'packed': 'filled to capacity', 'witness': 'observe',
            'history': 'records', 'facing': 'competing against', 'received': 'collected',
            'technology': 'framework', 'embeds': 'incorporates', 'statistical': 'quantitative',
            'watermark': 'statistical signal', 'generated': 'produced', 'text': 'content',
            'processing': 'pipeline', 'demonstrates': 'illustrates', 'provides': 'offers',
            'system': 'architecture', 'attack': 'transformation', 'paraphrase': 'rephrase',
            'model': 'neural network', 'detect': 'identify', 'invisible': 'imperceptible',
            'imperceptible': 'undetectable', 'biasing': 'altering', 'token': 'lexical unit',
            'selection': 'sampling', 'probabilities': 'distributions', 'signal': 'signature',
            'resides': 'exists', 'patterns': 'structures', 'survives': 'withstands',
            'editing': 'modification', 'techniques': 'methods', 'google': 'DeepMind',
            'deepmind': 'Google DeepMind', 'article': 'passage', 'analysts': 'observers',
        }

        sentences = re.split(r'(?<=[.!?])\s+', text)
        paraphrased_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            words = sentence.split()
            new_words = []
            for w in words:
                clean_w = re.sub(r'[^a-zA-Z]', '', w).lower()
                if clean_w in synonym_map:
                    replacement = synonym_map[clean_w]
                    if w[0].isupper():
                        replacement = replacement.capitalize()
                    punct = re.findall(r'[^a-zA-Z]+$', w)
                    if punct:
                        replacement += punct[0]
                    new_words.append(replacement)
                else:
                    new_words.append(w)
            paraphrased_sentences.append(' '.join(new_words))

        return ' '.join(paraphrased_sentences)

    def _paraphrase_heuristic_fallback(self, text: str) -> str:
        """Structural rephrasing: synonyms, perturbation, and sentence shuffle."""
        from reverse_synthid.reverse_synthid import TokenPerturbationAttack
        from attacks.sentence_shuffling import SentenceShufflingAttack

        result = self._apply_synonym_map(text)
        try:
            perturb = TokenPerturbationAttack()
            result = perturb.substitute_synonyms(result, substitution_rate=0.2)
        except Exception as e:
            logger.warning(f"Perturbation in heuristic fallback failed: {e}")

        try:
            shuffle = SentenceShufflingAttack()
            result = shuffle.transform(result)
        except Exception as e:
            logger.warning(f"Shuffle in heuristic fallback failed: {e}")

        return result if result.strip() else text
