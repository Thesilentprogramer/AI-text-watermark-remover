"""Tests for linguistic entropy layer."""

from attacks.entropy import LinguisticEntropyLayer


def test_entropy_apply():
    layer = LinguisticEntropyLayer()
    text = "This is a simple test sentence. It has multiple parts for variation."
    result = layer.apply(text)
    assert result
    assert len(result) > 0
