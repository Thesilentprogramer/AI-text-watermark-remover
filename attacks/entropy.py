"""
LinguisticEntropyLayer — sentence length variation and synonym swapping.
Uses regex-based tokenization to avoid NLTK download dependencies at runtime.
"""

import re
import random


SYNONYM_MAP = {
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
    "bad": ["adverse", "detrimental", "unfavorable", "suboptimal"],
    "text": ["content", "passage", "copy", "material"],
    "system": ["framework", "platform", "setup", "architecture"],
    "model": ["network", "engine", "system", "framework"],
    "process": ["pipeline", "workflow", "procedure", "operation"],
}


class LinguisticEntropyLayer:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _split_long_sentence(self, sentence: str) -> list:
        words = sentence.split()
        if len(words) < 25:
            return [sentence]

        connectors = [" and ", " but ", " which ", " because ", " however "]
        lower = sentence.lower()
        for connector in connectors:
            if connector in lower:
                idx = lower.index(connector)
                left = sentence[:idx].strip()
                right = sentence[idx + len(connector):].strip()
                if left and right:
                    right = right[0].upper() + right[1:]
                    return [left + ".", right]

        if ", " in sentence:
            parts = sentence.split(", ", 1)
            return [parts[0] + ".", parts[1][0].upper() + parts[1][1:]]

        return [sentence]

    def _merge_short_sentences(self, sentences: list) -> list:
        result = []
        i = 0
        while i < len(sentences):
            if (
                i + 1 < len(sentences)
                and len(sentences[i].split()) < 6
                and len(sentences[i + 1].split()) < 6
            ):
                merged = sentences[i].rstrip(".") + ", " + sentences[i + 1].lstrip()
                result.append(merged)
                i += 2
            else:
                result.append(sentences[i])
                i += 1
        return result

    def vary_sentence_lengths(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        expanded = []
        for s in sentences:
            expanded.extend(self._split_long_sentence(s))
        merged = self._merge_short_sentences(expanded)
        return " ".join(merged)

    def swap_synonyms(self, text: str, rate: float = 0.15) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean in SYNONYM_MAP and random.random() < rate:
                replacement = random.choice(SYNONYM_MAP[clean])
                if word and word[0].isupper():
                    replacement = replacement.capitalize()
                punct = re.findall(r'[^a-zA-Z]+$', word)
                if punct:
                    replacement += punct[0]
                result.append(replacement)
            else:
                result.append(word)
        text_out = " ".join(result)
        return re.sub(r" ([.,!?;:])", r"\1", text_out)

    def apply(self, text: str, synonym_rate: float = 0.15) -> str:
        text = self.vary_sentence_lengths(text)
        text = self.swap_synonyms(text, rate=synonym_rate)
        return text
