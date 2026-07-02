"""
SynthID Attack Pipeline Orchestrator
Executes & tracks text transformation across sanitize → detect(pre) → attack → detect(post).
Includes confidence-based attack selector and word-level diff visualizer.
"""

import time
import logging

from attacks.sanitizer import sanitize_text
from attacks.diff_engine import generate_word_diff_html
from attacks.auto_selector import select_attack, AttackPlan
from reverse_synthid.reverse_synthid import TokenPerturbationAttack, HomoglyphAttack
from attacks.sentence_shuffling import SentenceShufflingAttack
from app.model_loader import (
    get_paraphrase_engine,
    get_detector_engine,
    get_backtranslate_engine,
    get_entropy_engine,
)
from app.schemas import WatermarkRequest, WatermarkResponse, DetectionScore, StepResult

logger = logging.getLogger("pipeline")

MANUAL_LAYER_MAP = {
    "combined": ["paraphrase", "perturb", "entropy"],
    "paraphrase": ["paraphrase"],
    "perturb": ["perturb"],
    "sanitize_perturb": ["perturb"],
    "backtranslate": ["backtranslate"],
    "homoglyph": ["homoglyph"],
    "shuffle": ["shuffle"],
    "none": [],
    "sanitize": [],
}


class AttackPipeline:
    def __init__(self):
        self.perturb_engine = TokenPerturbationAttack()
        self.homoglyph_engine = HomoglyphAttack()
        self.shuffle_engine = SentenceShufflingAttack()

    def _execute_layers(
        self,
        layers: list,
        current_text: str,
        request: WatermarkRequest,
        step_logs: list,
        intermediate_steps: list,
        attack_mode: str,
    ) -> str:
        paraphraser = get_paraphrase_engine()
        step_num = 3

        for layer in layers:
            if layer == "paraphrase":
                current_text = paraphraser.paraphrase(
                    text=current_text,
                    enable_thinking=request.enable_thinking,
                ).strip()
                desc = "Gemma 4 E2B paraphrased token sequence under fresh probability distributions."
                step_logs.append(f"Step {step_num} (Paraphrase): {desc}")

            elif layer == "backtranslate":
                bt = get_backtranslate_engine()
                result = bt.attack(current_text)
                current_text = result.final_text.strip()
                desc = f"Back-translated via {result.pivot_language.upper()} pivot language."
                step_logs.append(f"Step {step_num} (Back-translate): {desc}")

            elif layer == "perturb":
                rate = request.substitution_rate or 0.15
                current_text = self.perturb_engine.substitute_synonyms(
                    current_text,
                    substitution_rate=rate,
                )
                desc = f"Applied synonym perturbation (rate={rate})."
                step_logs.append(f"Step {step_num} (Perturb): {desc}")

            elif layer == "entropy":
                entropy = get_entropy_engine()
                current_text = entropy.apply(current_text, synonym_rate=request.substitution_rate or 0.15)
                desc = "Applied linguistic entropy layer (sentence variation + synonym swap)."
                step_logs.append(f"Step {step_num} (Entropy): {desc}")

            elif layer == "homoglyph":
                current_text = self.homoglyph_engine.apply_homoglyphs(
                    current_text,
                    rate=request.substitution_rate or 0.25,
                )
                desc = "Replaced ASCII characters with Cyrillic homoglyphs."
                step_logs.append(f"Step {step_num} (Homoglyph): {desc}")

            elif layer == "shuffle":
                current_text = self.shuffle_engine.transform(current_text)
                desc = "Shuffled sentence order to break n-gram context boundaries."
                step_logs.append(f"Step {step_num} (Shuffle): {desc}")

            else:
                continue

            intermediate_steps.append(StepResult(
                step_number=step_num,
                step_name=f"Attack Layer ({layer})",
                text_after_step=current_text,
                g_value=None,
                description=desc,
            ))
            step_num += 1

        if not layers:
            step_logs.append(f"Step 3 (Attack): No attack layers applied for mode '{attack_mode}'.")

        return current_text

    def run(self, request: WatermarkRequest) -> WatermarkResponse:
        start_time = time.time()
        step_logs = []
        intermediate_steps = []

        raw_text = request.text
        req_mode = request.attack_mode.lower() if request.attack_mode else "auto"

        detector = get_detector_engine()

        # Step 1 — SANITIZE
        sanitization_res = sanitize_text(raw_text)
        sanitized_text = sanitization_res["sanitized_text"]
        removed_chars = sanitization_res["removed_count"]
        anomaly_score = sanitization_res.get("anomaly_score", 0.0)

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
            description=s1_desc,
        ))

        # Step 2 — DETECT (before)
        pre_detect_dict = detector.detect(sanitized_text)
        pre_score = DetectionScore(**pre_detect_dict)
        token_count = detector.count_tokens(sanitized_text)

        pre_status = "WATERMARKED" if pre_score.is_watermarked else "UNWATERMARKED"
        ppl_note = ""
        if pre_score.perplexity is not None and pre_score.perplexity > 0:
            ppl_note = f", Perplexity={pre_score.perplexity:.1f}"
        s2_desc = (
            f"Scanned input n-gram statistics → G-Value={pre_score.g_value:.4f} "
            f"({pre_status}, {token_count} tokens{ppl_note})"
        )
        step_logs.append(f"Step 2 (Detect Pre): {s2_desc}")

        intermediate_steps.append(StepResult(
            step_number=2,
            step_name="Pre-Attack Detection",
            text_after_step=sanitized_text,
            g_value=pre_score.g_value,
            description=s2_desc,
        ))

        # Attack selection
        auto_selected = False
        auto_rationale = None
        attack_plan: AttackPlan = None

        if req_mode == "auto":
            attack_plan = select_attack(
                pre_g_value=pre_score.g_value,
                unicode_anomaly_score=anomaly_score,
                zero_width_count=removed_chars,
                token_count=token_count,
                perplexity=pre_score.perplexity or 0.0,
            )
            attack_mode = attack_plan.attack_mode
            layers = attack_plan.layers
            auto_rationale = attack_plan.rationale
            auto_selected = True
            step_logs.append(
                f"⚡ Auto-Selected '{attack_mode}' ({attack_plan.estimated_time}): {auto_rationale}"
            )
        else:
            attack_mode = req_mode
            layers = MANUAL_LAYER_MAP.get(attack_mode, ["paraphrase"])

        current_text = sanitized_text
        current_text = self._execute_layers(
            layers=layers,
            current_text=current_text,
            request=request,
            step_logs=step_logs,
            intermediate_steps=intermediate_steps,
            attack_mode=attack_mode,
        )

        clean_text = current_text.strip()

        # Step 5 — DETECT (after)
        post_detect_dict = detector.detect(clean_text)
        post_score = DetectionScore(**post_detect_dict)

        post_status = "WATERMARKED" if post_score.is_watermarked else "CLEAN"
        post_ppl = ""
        if post_score.perplexity is not None and post_score.perplexity > 0:
            post_ppl = f", Perplexity={post_score.perplexity:.1f}"
        s5_desc = (
            f"Scanned final output → Post G-Value={post_score.g_value:.4f} "
            f"({post_status}{post_ppl})"
        )
        step_logs.append(f"Step 5 (Detect Post): {s5_desc}")

        intermediate_steps.append(StepResult(
            step_number=5,
            step_name="Post-Attack Verification",
            text_after_step=clean_text,
            g_value=post_score.g_value,
            description=s5_desc,
        ))

        diff_html = generate_word_diff_html(raw_text, clean_text)

        pre_g = pre_score.g_value
        post_g = post_score.g_value
        if pre_g > 0:
            reduction_pct = round(max(0.0, ((pre_g - post_g) / pre_g) * 100), 2)
        else:
            reduction_pct = 0.0

        is_clean = not post_score.is_watermarked

        if attack_mode == "none":
            verdict_title = "🟢 TEXT APPEARS CLEAN — NO ATTACK APPLIED"
        elif is_clean:
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
            processing_time_ms=elapsed_ms,
        )
