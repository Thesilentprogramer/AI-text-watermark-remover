"""
SynthID Attack Pipeline Orchestrator
Executes: Step 1 (Sanitize) -> Step 2 (Detect Pre) -> Step 3 (Attack/Paraphrase/Homoglyph/Shuffle) -> Step 4 (Perturb) -> Step 5 (Detect Post)
"""

import time
import logging
from attacks.sanitizer import sanitize_text
from attacks.token_perturbation import TokenPerturbationAttack
from attacks.homoglyph import HomoglyphAttack
from attacks.sentence_shuffling import SentenceShufflingAttack
from app.model_loader import get_paraphrase_engine, get_detector_engine
from app.schemas import WatermarkRequest, WatermarkResponse, DetectionScore

logger = logging.getLogger("pipeline")


class AttackPipeline:
    def __init__(self):
        self.perturb_engine = TokenPerturbationAttack()
        self.homoglyph_engine = HomoglyphAttack()
        self.shuffle_engine = SentenceShufflingAttack()

    def run(self, request: WatermarkRequest) -> WatermarkResponse:
        start_time = time.time()
        step_logs = []

        raw_text = request.text
        attack_mode = request.attack_mode.lower() if request.attack_mode else "combined"

        # Step 1 — SANITIZE
        sanitization_res = sanitize_text(raw_text)
        sanitized_text = sanitization_res["sanitized_text"]
        removed_chars = sanitization_res["removed_count"]

        if removed_chars > 0:
            step_logs.append(f"Step 1 (Sanitize): Removed {removed_chars} zero-width unicode characters.")
        else:
            step_logs.append("Step 1 (Sanitize): No hidden unicode characters detected.")

        detector = get_detector_engine()
        paraphraser = get_paraphrase_engine()

        # Step 2 — DETECT (before)
        pre_detect_dict = detector.detect(sanitized_text)
        pre_score = DetectionScore(**pre_detect_dict)

        pre_status = "WATERMARKED" if pre_score.is_watermarked else "UNWATERMARKED"
        step_logs.append(f"Step 2 (Detect Pre): G-Value = {pre_score.g_value:.4f} → Verdict: {pre_status} ({pre_score.confidence.upper()} confidence)")

        current_text = sanitized_text

        # Step 3 & 4 — ATTACK EXECUTION
        if attack_mode in ["combined", "paraphrase"]:
            paraphrased = paraphraser.paraphrase(
                text=current_text,
                enable_thinking=request.enable_thinking
            )
            if paraphrased != current_text:
                step_logs.append("Step 3 (Attack): Gemma 4 E2B paraphrased token sequence successfully.")
                current_text = paraphrased
            else:
                step_logs.append("Step 3 (Attack): Gemma 4 paraphrase completed.")

        if attack_mode == "homoglyph":
            current_text = self.homoglyph_engine.transform(current_text, rate=request.substitution_rate or 0.25)
            step_logs.append("Step 3 (Homoglyph): Replaced ASCII characters with Cyrillic lookalikes.")

        if attack_mode == "shuffle":
            current_text = self.shuffle_engine.transform(current_text)
            step_logs.append("Step 3 (Sentence Shuffle): Reordered sentence structure.")

        if attack_mode in ["combined", "perturb"]:
            current_text = self.perturb_engine.perturb(
                text=current_text,
                rate=request.substitution_rate
            )
            step_logs.append(f"Step 4 (Perturb): Secondary synonym substitution pass applied (rate={request.substitution_rate}).")

        clean_text = current_text.strip()

        # Step 5 — DETECT (after)
        post_detect_dict = detector.detect(clean_text)
        post_score = DetectionScore(**post_detect_dict)

        post_status = "WATERMARKED" if post_score.is_watermarked else "CLEAN"
        step_logs.append(f"Step 5 (Detect Post): G-Value = {post_score.g_value:.4f} → Verdict: {post_status}")

        # Calculate Watermark Reduction Percentage
        pre_g = pre_score.g_value
        post_g = post_score.g_value
        if pre_g > 0:
            reduction_pct = round(max(0.0, ((pre_g - post_g) / pre_g) * 100), 2)
        else:
            reduction_pct = 0.0

        is_clean = not post_score.is_watermarked

        # Generate prominent user verdict title
        if is_clean:
            verdict_title = f"🟢 WATERMARK SUCCESSFULLY REMOVED (-{reduction_pct:.1f}% Signal Drop)"
        elif reduction_pct > 15.0:
            verdict_title = f"🟡 WATERMARK SIGNAL REDUCED BY {reduction_pct:.1f}% (Partial Removal)"
        else:
            verdict_title = "🔴 WATERMARK SIGNAL STILL DETECTED (Try Combined Mode)"

        elapsed_ms = int((time.time() - start_time) * 1000)
        step_logs.append(f"Pipeline finished in {elapsed_ms} ms.")

        return WatermarkResponse(
            clean_text=clean_text,
            sanitized_char_count=removed_chars,
            pre_attack=pre_score,
            post_attack=post_score,
            watermark_reduction_pct=reduction_pct,
            is_clean=is_clean,
            verdict_title=verdict_title,
            step_logs=step_logs,
            attack_used=attack_mode,
            processing_time_ms=elapsed_ms
        )
