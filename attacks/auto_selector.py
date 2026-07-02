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
    1. Zero-width unicode anomalies detected -> 'sanitize' + 'perturb' (Sanitize zero-width first)
    2. Very high G-value (>= 0.74) -> 'combined' (Gemma 4 Paraphrase + Secondary Perturbation)
    3. Short text (< 150 tokens) -> 'shuffle' or 'paraphrase' (Fast, preserves semantics)
    4. Long text (>= 150 tokens) -> 'paraphrase' (Thorough 128K LLM rewrite)
    """
    words = text.split()
    word_count = len(words)

    # Rule 1: Zero-Width Characters Detected
    if zero_width_count > 0:
        return {
            "attack_mode": "combined",
            "rationale": f"Zero-width unicode anomalies ({zero_width_count} chars) detected. Applied Unicode Sanitizer followed by Combined Pass."
        }

    # Rule 2: Extremely High Watermark Signal (G >= 0.74)
    if pre_g_value >= 0.74:
        return {
            "attack_mode": "combined",
            "rationale": f"High SynthID watermark signal (G-Value {pre_g_value:.2f} >= 0.74) detected. Selected Combined Mode (Gemma 4 Paraphrase + Perturb) for 100% signal drop."
        }

    # Rule 3: Short Text (< 150 words)
    if word_count < 150:
        return {
            "attack_mode": "paraphrase",
            "rationale": f"Short text ({word_count} words) detected. Selected Gemma 4 Paraphrase for fast, semantic-preserving transformation."
        }

    # Rule 4: Long Text (>= 150 words)
    return {
        "attack_mode": "combined",
        "rationale": f"Long form text ({word_count} words) detected. Selected Combined Mode for thorough n-gram hash coverage."
    }
