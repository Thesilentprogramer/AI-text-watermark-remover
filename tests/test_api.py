"""
Integration tests for FastAPI endpoints
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_remove_watermark_endpoint():
    payload = {
        "text": "Google DeepMind SynthID embeds statistical watermarks into generated text.",
        "attack_mode": "combined",
        "substitution_rate": 0.15
    }
    response = client.post("/remove-watermark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "clean_text" in data
    assert "pre_attack" in data
    assert "post_attack" in data
    assert "watermark_reduction_pct" in data
    assert "perplexity" in data["pre_attack"]


def test_auto_mode_endpoint():
    payload = {
        "text": "Google DeepMind SynthID embeds statistical watermarks into generated text.",
        "attack_mode": "auto",
    }
    response = client.post("/remove-watermark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["auto_selected"] is True
    assert data["auto_rationale"]


def test_homoglyph_mode_endpoint():
    payload = {
        "text": "Google DeepMind SynthID embeds statistical watermarks into generated text.",
        "attack_mode": "homoglyph",
        "substitution_rate": 0.25,
    }
    response = client.post("/remove-watermark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["attack_used"] == "homoglyph"
    assert data["clean_text"]
