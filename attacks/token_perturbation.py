"""
TokenPerturbationAttack
Secondary perturbation pass applying synonym swapping and word structure variations.
Runs locally to add extra noise on top of the primary paraphrase.
"""

import random


SYNONYM_DICT = {
    "important": ["crucial", "vital", "significant", "essential"],
    "use": ["utilize", "apply", "employ", "adopt"],
    "create": ["generate", "produce", "build", "construct"],
    "method": ["approach", "technique", "strategy", "procedure"],
    "show": ["demonstrate", "display", "reveal", "exhibit"],
    "help": ["assist", "aid", "support", "facilitate"],
    "change": ["modify", "alter", "transform", "adjust"],
    "result": ["outcome", "consequence", "effect", "output"],
    "strong": ["robust", "powerful", "intense", "potent"],
    "clear": ["evident", "explicit", "lucid", "apparent"],
    "fast": ["rapid", "swift", "quick", "speedy"],
    "large": ["substantial", "sizeable", "considerable", "extensive"],
    "small": ["minor", "modest", "slight", "compact"],
    "good": ["favorable", "advantageous", "beneficial", "positive"],
    "bad": ["adverse", "detrimental", "unfavorable", "suboptimal"]
}


class TokenPerturbationAttack:
    def __init__(self, substitution_rate: float = 0.15):
        self.substitution_rate = substitution_rate

    def perturb(self, text: str, rate: float = None) -> str:
        """
        Step 4 - PERTURB
        Swaps selected words with natural synonyms to break residual n-gram hash patterns.
        """
        if not text:
            return ""

        sub_rate = rate if rate is not None else self.substitution_rate
        words = text.split()
        perturbed_words = []

        for word in words:
            clean_word = word.strip(".,!?;:\"'()[]{}").lower()

            if clean_word in SYNONYM_DICT and random.random() < sub_rate:
                candidates = SYNONYM_DICT[clean_word]
                replacement = random.choice(candidates)

                # Match original capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()

                # Preserve trailing punctuation
                punctuation = ""
                if word and not word[-1].isalnum():
                    punctuation = word[-1]

                perturbed_words.append(replacement + punctuation)
            else:
                perturbed_words.append(word)

        return " ".join(perturbed_words)
