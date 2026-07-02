from attacks.auto_selector import evaluate_optimal_attack


def test_zero_width_unicode_selection():
    res = evaluate_optimal_attack("Sample text", pre_g_value=0.50, zero_width_count=3)
    assert res["attack_mode"] == "combined"
    assert "Zero-width" in res["rationale"]


def test_high_g_value_selection():
    res = evaluate_optimal_attack("Sample text", pre_g_value=0.76, zero_width_count=0)
    assert res["attack_mode"] == "combined"
    assert "Watermark signal detected" in res["rationale"]


def test_clean_input_selection():
    text = "This is a short sentence for testing auto mode."
    res = evaluate_optimal_attack(text, pre_g_value=0.50, zero_width_count=0)
    assert res["attack_mode"] == "combined"
    assert "Combined Mode" in res["rationale"]
