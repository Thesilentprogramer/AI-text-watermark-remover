"""
Singleton Model Loader for Gemma 4 Paraphrasing Attack & Watermark Detector
Prevents model re-initialization per request.
"""

import logging
from attacks.gemma4_paraphrase import Gemma4ParaphrasingAttack
from reverse_synthid.detector import WatermarkDetector

logger = logging.getLogger("model_loader")

_paraphrase_engine: Gemma4ParaphrasingAttack = None
_detector_engine: WatermarkDetector = None


def init_engines():
    global _paraphrase_engine, _detector_engine
    logger.info("Initializing ML engines (WatermarkDetector & Gemma4ParaphrasingAttack)...")

    if _detector_engine is None:
        _detector_engine = WatermarkDetector(tokenizer_name="gpt2")

    if _paraphrase_engine is None:
        _paraphrase_engine = Gemma4ParaphrasingAttack()
        # Optionally load local model weights if hardware supports CUDA/bfloat16
        # _paraphrase_engine.load_local_model()

    logger.info("Engines successfully initialized.")


def get_paraphrase_engine() -> Gemma4ParaphrasingAttack:
    global _paraphrase_engine
    if _paraphrase_engine is None:
        init_engines()
    return _paraphrase_engine


def get_detector_engine() -> WatermarkDetector:
    global _detector_engine
    if _detector_engine is None:
        init_engines()
    return _detector_engine
