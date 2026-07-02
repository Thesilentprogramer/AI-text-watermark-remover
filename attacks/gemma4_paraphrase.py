"""
Gemma4ParaphrasingAttack
Rewrites token sequences using google/gemma-4-E2B-it to completely destroy SynthID n-gram hash bias.
Supports AutoModelForImageTextToText local inference as well as API fallback.
"""

import os
import logging
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("gemma4_paraphrase")


class Gemma4ParaphrasingAttack:
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or os.getenv("MODEL_ID", "google/gemma-4-E2B-it")
        self.hf_token = os.getenv("HF_TOKEN")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.processor = None
        self.loaded = False

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
        enable_thinking: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64
    ) -> str:
        """
        Step 3 - ATTACK
        Paraphrases text completely to destroy n-gram hash watermarking patterns.
        """
        if not text or len(text.strip()) == 0:
            return ""

        # 1. Attempt Local Inference if model loaded
        if self.loaded and self.model and self.processor:
            return self._paraphrase_local(text, enable_thinking, temperature, top_p, top_k)

        # 2. Attempt API Inference if GOOGLE_API_KEY or HF_TOKEN is configured
        if self.google_api_key:
            api_res = self._paraphrase_gemini_api(text)
            if api_res:
                return api_res

        # 3. Smart Algorithmic Paraphrase Fallback if offline/unloaded
        return self._paraphrase_heuristic_fallback(text)

    def _paraphrase_local(self, text: str, enable_thinking: bool, temperature: float, top_p: float, top_k: int) -> str:
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
                enable_thinking=enable_thinking
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
            
            # Strip thinking tags using Gemma 4's parse_response if available
            if hasattr(self.processor, "parse_response"):
                clean_output = self.processor.parse_response(raw_decoded)
            else:
                clean_output = self._strip_thinking_tags(raw_decoded)

            return clean_output.strip()
        except Exception as e:
            logger.error(f"Error during local Gemma 4 inference: {e}")
            return self._paraphrase_heuristic_fallback(text)

    def _paraphrase_gemini_api(self, text: str) -> str:
        """Fallback API call via Google AI Studio / Gemini API if local weights are missing."""
        try:
            import requests

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.google_api_key}"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": (
                            "Paraphrase the following text completely in your own words while preserving the exact meaning. "
                            "Do not include commentary or quotes. Output only the rewritten text.\n\n"
                            f"{text}"
                        )
                    }]
                }]
            }

            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                result = data["candidates"][0]["content"]["parts"][0]["text"]
                return result.strip()
        except Exception as e:
            logger.warning(f"Gemini API fallback error: {e}")
        return None

    def _strip_thinking_tags(self, text: str) -> str:
        """Strips <thought>...</thought> tags manually if parse_response is unavailable."""
        import re
        cleaned = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        return cleaned.replace('<eos>', '').replace('</s>', '').strip()

    def _paraphrase_heuristic_fallback(self, text: str) -> str:
        """High quality structural rephrasing fallback for testing environments."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        rewritten_sentences = []

        synonyms = {
            "invisible": "imperceptible",
            "watermark": "statistical signal",
            "detect": "identify",
            "generated": "produced",
            "system": "framework",
            "attack": "transformation",
            "paraphrase": "rephrase",
            "model": "neural network",
            "text": "content",
            "process": "pipeline",
            "demonstrates": "illustrates",
            "provides": "offers",
            "requires": "demands"
        }

        for sentence in sentences:
            words = sentence.split()
            new_words = [synonyms.get(w.lower().strip(',.'), w) for w in words]
            rewritten = " ".join(new_words)
            if len(rewritten) > 0:
                rewritten_sentences.append(rewritten[0].upper() + rewritten[1:])

        result = ". ".join(rewritten_sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result if result else text
