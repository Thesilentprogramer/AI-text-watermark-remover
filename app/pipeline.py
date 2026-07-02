"""
SynthID Attack Pipeline Orchestrator
Executes: Step 1 (Sanitize) -> Step 2 (Detect Pre) -> Step 3 (Attack/Paraphrase) -> Step 4 (Perturb) -> Step 5 (Detect Post)
"""

import time
import logging
from attacks.sanitizer import sanitize_text
from attacks.token_perturbation import TokenPerturbationAttack
from app.model_loader import get_paraphrase_engine, get_detector_engine
from app.schemas import WatermarkRequest, WatermarkResponse, DetectionScore

logger = logging.getLogger("pipeline")


class AttackPipeline:
    def __init__(self):
        self.perturb_engine = TokenPerturbationAttack()

    def run(self, request: WatermarkRequest) -> WatermarkResponse:
        start_time = time.time()

        raw_text = request.text
        attack_mode = request.attack_mode.lower() if request.attack_mode else "combined"

        # Step 1 — SANITIZE
        sanitization_res = sanitize_text(raw_text)
        sanitized_text = sanitization_res["sanitized_text"]
        removed_chars = sanitization_res["removed_count"]

        detector = get_detector_engine()
        paraphraser = get_paraphrase_engine()

        # Step 2 — DETECT (before)
        pre_detect_dict = detector.detect(sanitized_text)
        pre_score = DetectionScore(**pre_detect_dict)

        current_text = sanitized_text

        # Step 3 — ATTACK (Gemma 4 Paraphrasing)
        if attack_mode in ["combined", "paraphrase"]:
            current_text = paraphraser.paraphrase(
                text=current_text,
                enable_thinking=request.enable_thinking
            )

        # Step 4 — PERTURB (Secondary Pass)
        if attack_mode in ["combined", "perturb"]:
            current_text = self.perturb_engine.perturb(
                text=current_text,
                rate=request.substitution_rate
            )

        clean_text = current_text.strip()

        # Step 5 — DETECT (after)
        post_detect_dict = detector.detect(clean_text)
        post_score = DetectionScore(**post_detect_dict)

        # Calculate Watermark Reduction Percentage
        pre_g = pre_score.g_value
        post_g = post_score.g_value
        if pre_g > 0:
            reduction_pct = round(max(0.0, ((pre_g - post_g) / pre_g) * 100), 2)
        else:
            reduction_pct = 0.0

        elapsed_ms = int((time.time() - start_time) * 1000)

        return WatermarkResponse(
            clean_text=clean_text,
            sanitized_char_count=removed_chars,
            pre_attack=pre_score,
            post_attack=post_score,
            watermark_reduction_pct=reduction_pct,
            attack_used=attack_mode,
            processing_time_ms=elapsed_ms
        )
