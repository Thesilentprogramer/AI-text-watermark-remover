"""
Confidence-Based Attack Selector
Automatically selects the optimal adversarial attack based on input text characteristics.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class AttackPlan:
    attack_mode: str
    layers: List[str]
    rationale: str
    estimated_time: str


def select_attack(
    pre_g_value: float,
    unicode_anomaly_score: float,
    zero_width_count: int,
    token_count: int,
    perplexity: float = 0.0,
) -> AttackPlan:
    """
    Select attack mode based on pre-detection metrics.

    Rules (priority order):
    1. G > 0.75 → combined (all layers)
    2. Unicode anomaly → sanitize + perturb
    3. G >= 0.55 and tokens < 200 → backtranslate
    4. G >= 0.55 and tokens >= 200 → paraphrase
    5. Low perplexity (< 60) with normal G → paraphrase/backtranslate (likely AI, no SynthID)
    6. Clean input → none
    """
    if pre_g_value > 0.75:
        return AttackPlan(
            attack_mode="combined",
            layers=["paraphrase", "perturb", "entropy"],
            rationale=(
                f"Very high G-value ({pre_g_value:.2f} > 0.75). "
                "Running combined mode: paraphrase + perturb + entropy for total signal destruction."
            ),
            estimated_time="15–25 seconds",
        )

    if zero_width_count > 0 or unicode_anomaly_score > 0.01:
        return AttackPlan(
            attack_mode="sanitize_perturb",
            layers=["perturb"],
            rationale=(
                f"Unicode anomaly detected ({zero_width_count} hidden chars, "
                f"score {unicode_anomaly_score:.4f}). Sanitize then apply synonym perturbation."
            ),
            estimated_time="2–5 seconds",
        )

    if pre_g_value >= 0.55 and token_count < 200:
        return AttackPlan(
            attack_mode="backtranslate",
            layers=["backtranslate"],
            rationale=(
                f"Moderate watermark signal (G={pre_g_value:.2f}) on short text "
                f"({token_count} tokens < 200). Back-translation is fast and sufficient."
            ),
            estimated_time="5–10 seconds",
        )

    if pre_g_value >= 0.55 and token_count >= 200:
        return AttackPlan(
            attack_mode="paraphrase",
            layers=["paraphrase"],
            rationale=(
                f"Moderate watermark signal (G={pre_g_value:.2f}) on long text "
                f"({token_count} tokens >= 200). Full Gemma 4 paraphrase for thorough removal."
            ),
            estimated_time="10–20 seconds",
        )

    if pre_g_value < 0.55 and perplexity > 0 and perplexity < 60:
        if token_count >= 200:
            return AttackPlan(
                attack_mode="paraphrase",
                layers=["paraphrase"],
                rationale=(
                    f"Low perplexity ({perplexity:.1f}) suggests AI-generated text with no SynthID signal "
                    f"(G={pre_g_value:.2f}). Paraphrasing long text ({token_count} tokens)."
                ),
                estimated_time="10–20 seconds",
            )
        return AttackPlan(
            attack_mode="paraphrase",
            layers=["paraphrase"],
            rationale=(
                f"Low perplexity ({perplexity:.1f}) suggests AI-generated text with no SynthID signal "
                f"(G={pre_g_value:.2f}). Paraphrasing short text ({token_count} tokens)."
            ),
            estimated_time="10–20 seconds",
        )

    if os.getenv("FORCE_ATTACK", "false").lower() == "true" and token_count >= 50:
        mode = "paraphrase" if token_count >= 200 else "paraphrase"
        return AttackPlan(
            attack_mode=mode,
            layers=["paraphrase"],
            rationale=(
                f"FORCE_ATTACK enabled — running paraphrase on {token_count} tokens "
                f"(G={pre_g_value:.2f}, PPL={perplexity:.1f})."
            ),
            estimated_time="10–20 seconds",
        )

    # Perplexity unavailable but long text with borderline G (typical Gemini baseline ~0.50)
    if perplexity == 0 and token_count >= 100 and 0.48 <= pre_g_value < 0.55:
        return AttackPlan(
            attack_mode="paraphrase",
            layers=["paraphrase"],
            rationale=(
                f"Borderline G-value ({pre_g_value:.2f}) on long text ({token_count} tokens). "
                "Perplexity unavailable — paraphrasing to reset token sequence."
            ),
            estimated_time="10–20 seconds",
        )

    return AttackPlan(
        attack_mode="none",
        layers=[],
        rationale=(
            f"G-value ({pre_g_value:.2f}) and unicode profile are within normal range. "
            "No attack required."
        ),
        estimated_time="0 seconds",
    )


def evaluate_optimal_attack(
    text: str,
    pre_g_value: float,
    zero_width_count: int,
    token_count: int = None,
    unicode_anomaly_score: float = None,
    perplexity: float = 0.0,
) -> dict:
    """Backward-compatible wrapper returning dict for pipeline integration."""
    if token_count is None:
        token_count = len(text.split())
    if unicode_anomaly_score is None:
        unicode_anomaly_score = zero_width_count / max(len(text), 1)

    plan = select_attack(
        pre_g_value=pre_g_value,
        unicode_anomaly_score=unicode_anomaly_score,
        zero_width_count=zero_width_count,
        token_count=token_count,
        perplexity=perplexity,
    )
    return {
        "attack_mode": plan.attack_mode,
        "layers": plan.layers,
        "rationale": plan.rationale,
        "estimated_time": plan.estimated_time,
    }
