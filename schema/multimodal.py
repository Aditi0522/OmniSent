from pydantic import BaseModel
from typing import Dict, List, Optional


class FusedWindow(BaseModel):
    start_ms: int
    end_ms: int
    fused_emotions: Dict[str, float]
    confidence: float
    dominant_emotion: str
    modalities_used: List[str]


class FusionConfig(BaseModel):
    window_ms: int
    smoothing_window: int


class FusionProcessingInfo(BaseModel):
    modalities_used: List[str]
    total_windows: int
    processing_time_sec: float


class MultimodalResponse(BaseModel):
    source_video: str
    config: FusionConfig
    processing: FusionProcessingInfo
    summary: Dict
    usability_metrics: Dict[str, float]
    usability_events: List[Dict]
    segments: List[Dict]
    fused_windows: List[FusedWindow]
    per_modality: Optional[Dict] = None

