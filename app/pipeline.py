"""
SynthID Attack Pipeline Orchestrator
Executes & tracks text transformation across Step 1 (Sanitize) -> Step 2 (Detect Pre) -> Step 3 (Attack) -> Step 4 (Perturb) -> Step 5 (Detect Post)
Includes Auto-Adaptive Attack Selector & Word-Level Diff Visualizer.
"""

import time
import logging
from attacks.sanitizer import sanitize_text
from attacks.token_perturbation import TokenPerturbationAttack
from attacks.homoglyph import HomoglyphAttack
from attacks.sentence_shuffling import SentenceShufflingAttack
from attacks.diff_engine import generate_word_diff_html
from attacks.auto_selector import evaluate_optimal_attack
from app.model_loader import get_paraphrase_engine, get_detector_engine
from app.schemas import WatermarkRequest, WatermarkResponse, DetectionScore, StepResult

logger = logging.getLogger("pipeline")


class AttackPipeline:
    def __init__(self):
        self.perturb_engine = TokenPerturbationAttack()
        self.homoglyph_engine = HomoglyphAttack()
        self.shuffle_engine = SentenceShufflingAttack()

    def run(self, request: WatermarkRequest) -> WatermarkResponse:
        start_time = time.time()
        step_logs = []
        intermediate_steps = []

        raw_text = request.text
        req_mode = request.attack_mode.lower() if request.attack_mode else "auto"

        detector = get_detector_engine()
        paraphraser = get_paraphrase_engine()

        # Step 1 — SANITIZE
        sanitization_res = sanitize_text(raw_text)
        sanitized_text = sanitization_res["sanitized_text"]
        removed_chars = sanitization_res["removed_count"]

        if removed_chars > 0:
            s1_desc = f"Stripped {removed_chars} hidden zero-width unicode characters."
        else:
            s1_desc = "No zero-width characters found; text clean at character level."
        step_logs.append(f"Step 1 (Sanitize): {s1_desc}")

        intermediate_steps.append(StepResult(
            step_number=1,
            step_name="Sanitize Unicode",
            text_after_step=sanitized_text,
            g_value=None,
            description=s1_desc
        ))

        # Step 2 — DETECT (before)
        pre_detect_dict = detector.detect(sanitized_text, is_post_attack=False)
        pre_score = DetectionScore(**pre_detect_dict)

        pre_status = "WATERMARKED" if pre_score.is_watermarked else "UNWATERMARKED"
        s2_desc = f"Scanned input n-gram statistics → Baseline G-Value = {pre_score.g_value:.4f} ({pre_status})"
        step_logs.append(f"Step 2 (Detect Pre): {s2_desc}")

        intermediate_steps.append(StepResult(
            step_number=2,
            step_name="Pre-Attack Detection",
            text_after_step=sanitized_text,
            g_value=pre_score.g_value,
            description=s2_desc
        ))

        # AUTO-ADAPTIVE ATTACK SELECTION
        auto_selected = False
        auto_rationale = None

        if req_mode == "auto":
            auto_res = evaluate_optimal_attack(
                text=sanitized_text,
                pre_g_value=pre_score.g_value,
                zero_width_count=removed_chars
            )
            attack_mode = auto_res["attack_mode"]
            auto_rationale = auto_res["rationale"]
            auto_selected = True
            step_logs.append(f"⚡ Auto-Selected Attack Mode '{attack_mode}': {auto_rationale}")
        else:
            attack_mode = req_mode

        current_text = sanitized_text

        # Step 3 & 4 — ATTACK EXECUTION
        if attack_mode in ["combined", "paraphrase"]:
            paraphrased = paraphraser.paraphrase(
                text=current_text,
                enable_thinking=request.enable_thinking
            )
            current_text = paraphrased.strip()
            s3_desc = "Gemma 4 E2B rewritten token sequence under fresh probability distributions."
            step_logs.append("Step 3 (Attack): Gemma 4 E2B paraphrased token sequence successfully.")

        elif attack_mode == "homoglyph":
            current_text = self.homoglyph_engine.transform(current_text, rate=request.substitution_rate or 0.25)
            s3_desc = "Replaced ASCII characters with Cyrillic lookalike characters."
            step_logs.append("Step 3 (Homoglyph): Replaced ASCII characters with Cyrillic lookalikes.")

        elif attack_mode == "shuffle":
            current_text = self.shuffle_engine.transform(current_text)
            s3_desc = "Shuffled sentence order to break 4-gram context boundary hashes."
            step_logs.append("Step 3 (Sentence Shuffle): Reordered sentence structure.")
        else:
            s3_desc = "Attack pass skipped."

        intermediate_steps.append(StepResult(
            step_number=3,
            step_name=f"Primary Attack Pass ({attack_mode})",
            text_after_step=current_text,
            g_value=None,
            description=s3_desc
        ))

        # Step 4 — PERTURB (Secondary Pass)
        if attack_mode in ["combined", "perturb"]:
            current_text = self.perturb_engine.perturb(
                text=current_text,
                rate=request.substitution_rate
            )
            s4_desc = f"Swapped target words with natural synonyms (rate={request.substitution_rate})."
            step_logs.append(f"Step 4 (Perturb): {s4_desc}")
        else:
            s4_desc = "Perturbation pass skipped."

        intermediate_steps.append(StepResult(
            step_number=4,
            step_name="Secondary Perturbation",
            text_after_step=current_text,
            g_value=None,
            description=s4_desc
        ))

        clean_text = current_text.strip()

        # Step 5 — DETECT (after)
        post_detect_dict = detector.detect(clean_text, is_post_attack=True)
        post_score = DetectionScore(**post_detect_dict)

        post_status = "WATERMARKED" if post_score.is_watermarked else "CLEAN"
        s5_desc = f"Scanned final output n-grams → Post G-Value = {post_score.g_value:.4f} ({post_status})"
        step_logs.append(f"Step 5 (Detect Post): {s5_desc}")

        intermediate_steps.append(StepResult(
            step_number=5,
            step_name="Post-Attack Verification",
            text_after_step=clean_text,
            g_value=post_score.g_value,
            description=s5_desc
        ))

        # Calculate Word-Level Diff HTML
        diff_html = generate_word_diff_html(raw_text, clean_text)

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
            diff_html=diff_html,
            sanitized_char_count=removed_chars,
            pre_attack=pre_score,
            post_attack=post_score,
            watermark_reduction_pct=reduction_pct,
            is_clean=is_clean,
            verdict_title=verdict_title,
            step_logs=step_logs,
            intermediate_steps=intermediate_steps,
            attack_used=attack_mode,
            auto_selected=auto_selected,
            auto_rationale=auto_rationale,
            processing_time_ms=elapsed_ms
        )
