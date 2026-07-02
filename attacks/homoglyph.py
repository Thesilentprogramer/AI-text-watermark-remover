"""
HomoglyphAttack
Replaces ASCII characters with visually identical Unicode lookalikes (Cyrillic/Greek).
Destroys tokenizer n-gram indexing without changing visual appearance to humans.
"""

import random

# Homoglyph mapping dictionary (ASCII -> Cyrillic/Greek lookalikes)
HOMOGLYPH_MAP = {
    'a': 'а',  # Cyrillic small letter a
    'c': 'с',  # Cyrillic small letter es
    'e': 'е',  # Cyrillic small letter ie
    'i': 'і',  # Cyrillic small letter byelorussian-ukrainian i
    'o': 'о',  # Cyrillic small letter o
    'p': 'р',  # Cyrillic small letter er
    's': 'ѕ',  # Cyrillic small letter dze
    'x': 'х',  # Cyrillic small letter ha
    'y': 'у',  # Cyrillic small letter straight u
    'A': 'А',  # Cyrillic capital letter A
    'B': 'В',  # Cyrillic capital letter Ve
    'C': 'С',  # Cyrillic capital letter Es
    'E': 'Е',  # Cyrillic capital letter IE
    'H': 'Н',  # Cyrillic capital letter En
    'I': 'І',  # Cyrillic capital letter Byelorussian-Ukrainian I
    'K': 'К',  # Cyrillic capital letter Ka
    'M': 'М',  # Cyrillic capital letter Em
    'O': 'О',  # Cyrillic capital letter O
    'P': 'Р',  # Cyrillic capital letter Er
    'T': 'Т',  # Cyrillic capital letter Te
    'X': 'Х',  # Cyrillic capital letter Ha
}


class HomoglyphAttack:
    def __init__(self, substitution_rate: float = 0.25):
        self.substitution_rate = substitution_rate

    def transform(self, text: str, rate: float = None) -> str:
        """
        Replaces selected ASCII characters with homoglyphs.
        """
        if not text:
            return ""

        sub_rate = rate if rate is not None else self.substitution_rate
        chars = list(text)

        for i, char in enumerate(chars):
            if char in HOMOGLYPH_MAP and random.random() < sub_rate:
                chars[i] = HOMOGLYPH_MAP[char]

        return "".join(chars)
