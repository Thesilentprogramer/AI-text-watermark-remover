"""
Auto-Adaptive Confidence-Based Attack Selector
Analyzes input text metrics (token length, zero-width unicode anomaly count, pre-attack G-value score)
and automatically selects the optimal adversarial attack strategy.
"""

from typing import Dict, Any


def evaluate_optimal_attack(text: str, pre_g_value: float, zero_width_count: int) -> Dict[str, Any]:
    """
    Evaluates optimal attack mode based on text metrics.

    Decision Rules:
    1. Zero-width unicode anomalies detected -> 'combined' (Sanitize + Combined Pass)
    2. Watermark signal detected (G >= 0.55) -> 'combined' (Gemma 4 Paraphrase + Secondary Perturbation)
    3. Low watermark signal (G < 0.55) -> 'paraphrase' (Fast structural rephrase)
    """
    words = text.split()
    word_count = len(words)

    # Rule 1: Zero-Width Characters Detected
    if zero_width_count > 0:
        return {
            "attack_mode": "combined",
            "rationale": f"Zero-width unicode anomalies ({zero_width_count} chars) detected. Applied Unicode Sanitizer followed by Combined Pass."
        }

    # Rule 2: Watermark Signal Detected (G >= 0.55)
    if pre_g_value >= 0.55:
        return {
            "attack_mode": "combined",
            "rationale": f"Watermark signal detected (G-Value {pre_g_value:.2f} >= 0.55). Selected Combined Mode (Paraphrase + Secondary Perturbation) for total signal destruction."
        }

    # Rule 3: Clean/Unwatermarked Input
    return {
        "attack_mode": "combined",
        "rationale": f"Analyzed {word_count} words. Applied Combined Mode for baseline security optimization."
    }
