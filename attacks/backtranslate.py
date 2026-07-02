"""
BackTranslationAttack — EN → pivot language → EN via Helsinki-NLP MarianMT models.
"""

from dataclasses import dataclass

from transformers import MarianMTModel, MarianTokenizer


@dataclass
class BackTranslateResult:
    original_text: str
    intermediate_text: str
    final_text: str
    pivot_language: str


class BackTranslationAttack:
    MODELS = {
        "de": {
            "forward": "Helsinki-NLP/opus-mt-en-de",
            "backward": "Helsinki-NLP/opus-mt-de-en",
        },
        "fr": {
            "forward": "Helsinki-NLP/opus-mt-en-fr",
            "backward": "Helsinki-NLP/opus-mt-fr-en",
        },
        "es": {
            "forward": "Helsinki-NLP/opus-mt-en-es",
            "backward": "Helsinki-NLP/opus-mt-es-en",
        },
    }

    def __init__(self, pivot_lang: str = "de"):
        if pivot_lang not in self.MODELS:
            raise ValueError(f"Unsupported pivot language: {pivot_lang}")
        self.pivot_lang = pivot_lang
        model_ids = self.MODELS[pivot_lang]

        self._fwd_tok = MarianTokenizer.from_pretrained(model_ids["forward"])
        self._fwd_model = MarianMTModel.from_pretrained(model_ids["forward"])
        self._bwd_tok = MarianTokenizer.from_pretrained(model_ids["backward"])
        self._bwd_model = MarianMTModel.from_pretrained(model_ids["backward"])

    def _translate(self, text: str, tokenizer, model) -> str:
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = model.generate(**inputs, num_beams=4, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _chunk_text(self, text: str, max_words: int = 80) -> list:
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def attack(self, text: str) -> BackTranslateResult:
        chunks = self._chunk_text(text) or [text]
        intermediate_chunks = []
        final_chunks = []

        for chunk in chunks:
            intermediate = self._translate(chunk, self._fwd_tok, self._fwd_model)
            intermediate_chunks.append(intermediate)
            final = self._translate(intermediate, self._bwd_tok, self._bwd_model)
            final_chunks.append(final)

        return BackTranslateResult(
            original_text=text,
            intermediate_text=" ".join(intermediate_chunks),
            final_text=" ".join(final_chunks),
            pivot_language=self.pivot_lang,
        )
