"""
Unit tests for ExtendedDetector & Sanitizer
"""

from attacks.sanitizer import sanitize_text, analyze_unicode_anomaly
from reverse_synthid.extended_detector import ExtendedDetector


def test_sanitize_text():
    text_with_zero_width = "Hello\u200B World\uFEFF!"
    res = sanitize_text(text_with_zero_width)
    assert res["sanitized_text"] == "Hello World!"
    assert res["removed_count"] == 2
    assert res["anomaly_score"] > 0


def test_analyze_unicode_anomaly():
    dirty = "This is\u200B a test."
    analysis = analyze_unicode_anomaly(dirty)
    assert analysis["total_invisible"] == 1
    assert analysis["is_suspicious"]


def test_detector_basic():
    detector = ExtendedDetector()
    res = detector.detect("This is a test sentence to verify detector scoring.")
    assert "g_value" in res
    assert "is_watermarked" in res
    assert "confidence" in res
    assert "perplexity" in res


def test_count_tokens():
    detector = ExtendedDetector()
    count = detector.count_tokens("This is a short test sentence.")
    assert count > 0
