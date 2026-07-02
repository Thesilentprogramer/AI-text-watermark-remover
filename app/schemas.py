from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class WatermarkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The watermarked input text")
    attack_mode: Optional[str] = Field(
        default="combined",
        description="Attack strategy: 'combined', 'paraphrase', 'perturb', or 'sanitize'"
    )
    enable_thinking: Optional[bool] = Field(
        default=True,
        description="Enable thinking mode for Gemma 4 paraphrasing"
    )
    substitution_rate: Optional[float] = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Token substitution rate for secondary perturbation pass"
    )


class DetectionScore(BaseModel):
    g_value: float
    is_watermarked: bool
    confidence: str
    sample_tokens: int


class WatermarkResponse(BaseModel):
    clean_text: str
    sanitized_char_count: int
    pre_attack: DetectionScore
    post_attack: DetectionScore
    watermark_reduction_pct: float
    attack_used: str
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    engine: str
