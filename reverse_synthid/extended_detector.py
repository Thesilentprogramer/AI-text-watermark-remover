"""
ExtendedDetector — wraps upstream SynthID WatermarkDetector and adds GPT-2 perplexity scoring.
"""

import math
import logging
import os

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

from reverse_synthid.reverse_synthid import WatermarkDetector as UpstreamDetector

logger = logging.getLogger("extended_detector")


class ExtendedDetector:
    def __init__(self, tokenizer_name: str = "gpt2"):
        self._repo_detector = UpstreamDetector(tokenizer_name=tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self._token_counter = self._repo_detector.tokenizer
        self._ppl_tokenizer = None
        self._ppl_model = None
        self._perplexity_enabled = os.getenv("ENABLE_PERPLEXITY", "true").lower() == "true"

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        tokens = self._token_counter(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).input_ids
        return int(tokens.shape[1])

    def _ensure_perplexity_model(self):
        if self._ppl_model is not None or not self._perplexity_enabled:
            return
        try:
            logger.info("Loading GPT-2 for perplexity scoring...")
            self._ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._ppl_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self._ppl_model.eval()
            logger.info("GPT-2 perplexity model loaded.")
        except Exception as e:
            logger.warning(f"Perplexity model load failed: {e}")
            self._perplexity_enabled = False

    def _compute_perplexity(self, text: str) -> float:
        self._ensure_perplexity_model()
        if not self._perplexity_enabled or self._ppl_model is None:
            return 0.0

        inputs = self._ppl_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._ppl_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return round(math.exp(loss.item()), 2)

    def _perplexity_label(self, ppl: float) -> str:
        if ppl <= 0:
            return "unknown"
        if ppl < 50:
            return "likely AI"
        if ppl < 100:
            return "possibly AI"
        return "likely human"

    def score(self, text: str) -> dict:
        mean_g, synthid_info = self._repo_detector.compute_score(text)
        g_value = round(float(mean_g), 4)
        token_count = synthid_info.get("num_tokens", self.count_tokens(text))
        perplexity = self._compute_perplexity(text)

        is_watermarked = g_value >= 0.55 or (perplexity > 0 and perplexity < 60)
        if g_value >= 0.65:
            confidence = "high"
        elif g_value >= 0.55:
            confidence = "medium"
        else:
            confidence = "low"

        verdict = "watermarked" if (g_value > 0.55 or (perplexity > 0 and perplexity < 60)) else "clean"

        return {
            "g_value": g_value,
            "is_watermarked": is_watermarked,
            "confidence": confidence,
            "sample_tokens": token_count,
            "perplexity": perplexity,
            "perplexity_label": self._perplexity_label(perplexity),
            "verdict": verdict,
        }

    def detect(self, text: str) -> dict:
        """Pipeline-compatible detection result."""
        result = self.score(text)
        return {
            "g_value": result["g_value"],
            "is_watermarked": result["is_watermarked"],
            "confidence": result["confidence"],
            "sample_tokens": result["sample_tokens"],
            "perplexity": result["perplexity"],
            "perplexity_label": result["perplexity_label"],
            "verdict": result["verdict"],
        }
