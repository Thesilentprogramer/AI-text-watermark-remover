"""
SentenceShufflingAttack
Reorders independent sentences to break n-gram context boundaries.
"""

import re
import random


class SentenceShufflingAttack:
    def transform(self, text: str) -> str:
        """
        Splits text into sentences and shuffles their order.
        """
        if not text:
            return ""

        # Split into sentences while preserving trailing punctuation
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        if len(sentences) <= 1:
            return text

        shuffled = list(sentences)
        random.shuffle(shuffled)

        # Rejoin shuffled sentences
        return " ".join(shuffled)
