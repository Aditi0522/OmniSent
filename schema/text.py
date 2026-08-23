from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TextSegmentPrediction(BaseModel):
    timestamp_ms: int
    end_ms: int
    text: str
    emotions: Dict[str, float]
    confidence: float
    dominant_emotion: str


class TextAnalysisConfig(BaseModel):
    smoothing_window: int
    min_segment_ms: int


class TextProcessingInfo(BaseModel):
    total_segments: int
    processing_time_sec: float


class TextAnalysisResponse(BaseModel):
    """Response from text emotion analysis."""
    source_audio: Optional[str] = None
    source_video: Optional[str] = None
    transcript: Optional[List[Dict]] = None
    config: TextAnalysisConfig
    processing: TextProcessingInfo
    summary: Dict
    usability_metrics: Dict[str, float]
    usability_events: List[Dict]
    segments: List[Dict]
    segment_predictions: List[TextSegmentPrediction]


class TextHealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    classifier_loaded: bool

