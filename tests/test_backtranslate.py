"""Smoke tests for back-translation attack."""

import pytest

from attacks.backtranslate import BackTranslationAttack


@pytest.mark.slow
def test_backtranslate_roundtrip():
    bt = BackTranslationAttack(pivot_lang="de")
    text = "Artificial intelligence is transforming industries."
    result = bt.attack(text)
    assert result.final_text
    assert result.final_text != ""
    assert result.pivot_language == "de"
