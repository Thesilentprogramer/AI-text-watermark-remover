"""
Unit tests for WatermarkDetector & Sanitizer
"""

from attacks.sanitizer import sanitize_text
from reverse_synthid.detector import WatermarkDetector


def test_sanitize_text():
    text_with_zero_width = "Hello\u200B World\uFEFF!"
    res = sanitize_text(text_with_zero_width)
    assert res["sanitized_text"] == "Hello World!"
    assert res["removed_count"] == 2


def test_detector_basic():
    detector = WatermarkDetector()
    res = detector.detect("This is a test sentence to verify detector scoring.")
    assert "g_value" in res
    assert "is_watermarked" in res
    assert "confidence" in res
    assert res["g_value"] >= 0.0
