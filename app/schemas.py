from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class WatermarkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The watermarked input text")
    attack_mode: Optional[str] = Field(
        default="auto",
        description="Attack strategy: 'auto' (intelligent adaptive selector), 'combined', 'paraphrase', 'perturb', 'homoglyph', 'shuffle', or 'sanitize'"
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
    perplexity: Optional[float] = None
    perplexity_label: Optional[str] = None
    verdict: Optional[str] = None


class StepResult(BaseModel):
    step_number: int
    step_name: str
    text_after_step: str
    g_value: Optional[float] = None
    description: str


class WatermarkResponse(BaseModel):
    clean_text: str
    diff_html: str
    sanitized_char_count: int
    pre_attack: DetectionScore
    post_attack: DetectionScore
    watermark_reduction_pct: float
    is_clean: bool
    verdict_title: str
    step_logs: List[str]
    intermediate_steps: List[StepResult]
    attack_used: str
    auto_selected: bool = False
    auto_rationale: Optional[str] = None
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    engine: str
