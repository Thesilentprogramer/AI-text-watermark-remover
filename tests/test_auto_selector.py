from attacks.auto_selector import select_attack, evaluate_optimal_attack


def test_very_high_g_combined():
    plan = select_attack(pre_g_value=0.80, unicode_anomaly_score=0.0, zero_width_count=0, token_count=50)
    assert plan.attack_mode == "combined"
    assert "paraphrase" in plan.layers
    assert "perturb" in plan.layers


def test_unicode_anomaly_perturb():
    plan = select_attack(pre_g_value=0.50, unicode_anomaly_score=0.05, zero_width_count=3, token_count=100)
    assert plan.attack_mode == "sanitize_perturb"
    assert plan.layers == ["perturb"]
    assert "Unicode anomaly" in plan.rationale


def test_short_text_backtranslate():
    plan = select_attack(pre_g_value=0.60, unicode_anomaly_score=0.0, zero_width_count=0, token_count=80)
    assert plan.attack_mode == "backtranslate"
    assert plan.layers == ["backtranslate"]


def test_long_text_paraphrase():
    plan = select_attack(pre_g_value=0.60, unicode_anomaly_score=0.0, zero_width_count=0, token_count=300)
    assert plan.attack_mode == "paraphrase"
    assert plan.layers == ["paraphrase"]


def test_clean_no_attack():
    plan = select_attack(
        pre_g_value=0.50, unicode_anomaly_score=0.0, zero_width_count=0,
        token_count=100, perplexity=120.0,
    )
    assert plan.attack_mode == "none"
    assert plan.layers == []


def test_low_perplexity_long_text_paraphrase():
    plan = select_attack(
        pre_g_value=0.50, unicode_anomaly_score=0.0, zero_width_count=0,
        token_count=300, perplexity=35.0,
    )
    assert plan.attack_mode == "paraphrase"
    assert "perplexity" in plan.rationale.lower()


def test_low_perplexity_short_text_backtranslate():
    plan = select_attack(
        pre_g_value=0.50, unicode_anomaly_score=0.0, zero_width_count=0,
        token_count=80, perplexity=40.0,
    )
    assert plan.attack_mode == "backtranslate"


def test_g_overrides_length():
    plan = select_attack(pre_g_value=0.80, unicode_anomaly_score=0.0, zero_width_count=0, token_count=300)
    assert plan.attack_mode == "combined"


def test_evaluate_optimal_attack_wrapper():
    res = evaluate_optimal_attack("Sample text", pre_g_value=0.76, zero_width_count=0, token_count=50)
    assert res["attack_mode"] == "combined"
    assert "layers" in res
