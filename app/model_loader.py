"""
Singleton Model Loader for Gemma 4 Paraphrasing Attack & Extended Watermark Detector
Prevents model re-initialization per request.
"""

import logging
import os

from attacks.gemma4_paraphrase import Gemma4ParaphrasingAttack
from reverse_synthid.extended_detector import ExtendedDetector

logger = logging.getLogger("model_loader")

_paraphrase_engine: Gemma4ParaphrasingAttack = None
_detector_engine: ExtendedDetector = None
_backtranslate_engine = None
_entropy_engine = None


def init_engines():
    global _paraphrase_engine, _detector_engine
    logger.info("Initializing ML engines (ExtendedDetector & Gemma4ParaphrasingAttack)...")

    if _detector_engine is None:
        _detector_engine = ExtendedDetector(tokenizer_name="gpt2")
        if os.getenv("ENABLE_PERPLEXITY", "true").lower() == "true":
            logger.info("Preloading GPT-2 perplexity model...")
            _detector_engine._ensure_perplexity_model()

    if _paraphrase_engine is None:
        _paraphrase_engine = Gemma4ParaphrasingAttack()
        if os.getenv("ENABLE_LOCAL_GEMMA", "false").lower() == "true":
            _paraphrase_engine.load_local_model()

    logger.info("Engines successfully initialized.")


def get_paraphrase_engine() -> Gemma4ParaphrasingAttack:
    global _paraphrase_engine
    if _paraphrase_engine is None:
        init_engines()
    return _paraphrase_engine


def get_detector_engine() -> ExtendedDetector:
    global _detector_engine
    if _detector_engine is None:
        init_engines()
    return _detector_engine


def get_backtranslate_engine():
    global _backtranslate_engine
    if _backtranslate_engine is None:
        from attacks.backtranslate import BackTranslationAttack
        _backtranslate_engine = BackTranslationAttack(pivot_lang="de")
    return _backtranslate_engine


def get_entropy_engine():
    global _entropy_engine
    if _entropy_engine is None:
        from attacks.entropy import LinguisticEntropyLayer
        _entropy_engine = LinguisticEntropyLayer()
    return _entropy_engine
