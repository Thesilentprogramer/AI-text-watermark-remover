from attacks.auto_selector import evaluate_optimal_attack


def test_zero_width_unicode_selection():
    res = evaluate_optimal_attack("Sample text", pre_g_value=0.50, zero_width_count=3)
    assert res["attack_mode"] == "combined"
    assert "Zero-width" in res["rationale"]


def test_high_g_value_selection():
    res = evaluate_optimal_attack("Sample text", pre_g_value=0.76, zero_width_count=0)
    assert res["attack_mode"] == "combined"
    assert "High SynthID watermark signal" in res["rationale"]


def test_short_text_selection():
    short_text = "This is a short sentence for testing auto mode."
    res = evaluate_optimal_attack(short_text, pre_g_value=0.50, zero_width_count=0)
    assert res["attack_mode"] == "paraphrase"
    assert "Short text" in res["rationale"]


def test_long_text_selection():
    long_text = " ".join(["word"] * 200)
    res = evaluate_optimal_attack(long_text, pre_g_value=0.50, zero_width_count=0)
    assert res["attack_mode"] == "combined"
    assert "Long form text" in res["rationale"]
