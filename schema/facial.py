from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class EmotionProbabilities(BaseModel):
    Angry: float = 0.0
    Disgusted: float = 0.0
    Fearful: float = 0.0
    Happy: float = 0.0
    Neutral: float = 0.0
    Sad: float = 0.0
    Surprised: float = 0.0

class UsabilityMetrics(BaseModel):
    frustration_index: float = 0.0
    frustration_index_max: Optional[float] = None
    confusion_score: float = 0.0
    confusion_score_max: Optional[float] = None
    engagement_level: float = 0.0
    engagement_level_max: Optional[float] = None
    satisfaction_score: float = 0.0
    satisfaction_score_max: Optional[float] = None
    boredom_score: float = 0.0
    boredom_score_max: Optional[float] = None
    stress_level: float = 0.0
    stress_level_max: Optional[float] = None

class UsabilityEvent(BaseModel):
    type: str
    timestamp_ms: int = 0
    severity: Optional[str] = None
    detail: Optional[str] = None
    from_emotion: Optional[str] = Field(None, alias="from")
    to_emotion: Optional[str] = Field(None, alias="to")
    magnitude: Optional[float] = None

class EmotionSegment(BaseModel):
    start_ms: int
    end_ms: int
    duration_ms: int
    dominant_emotion: str
    emotions: Dict[str, float]
    confidence: float

class FramePrediction(BaseModel):
    timestamp_ms: int
    smoothed_emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float

class AnalysisConfig(BaseModel):
    process_fps: int
    smoothing_window: int
    min_segment_ms: int

class ProcessingInfo(BaseModel):
    total_frames_processed: int
    faces_detected: int
    faces_missed: int
    detection_rate: float
    processing_time_sec: float

class AnalysisSummary(BaseModel):
    overall_emotions: Dict[str, float]
    dominant_emotion: str
    dominant_confidence: float
    time_distribution: Dict[str, float]
    total_frames: int
    total_duration_ms: float
    avg_confidence: float

class VideoAnalysisResponse(BaseModel):
    """Response from POST /analyze/facial (video upload)."""
    video: str
    config: AnalysisConfig
    processing: ProcessingInfo
    summary: AnalysisSummary
    usability_metrics: Dict[str, float]
    usability_events: List[UsabilityEvent]
    segments: List[EmotionSegment]
    frame_predictions: List[FramePrediction]

class FrameAnalysisResponse(BaseModel):
    """Response from single-frame analysis."""
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    det_score: float
    usability_metrics: UsabilityMetrics

class ErrorResponse(BaseModel):
    error: str

class HealthResponse(BaseModel):
    status: str
    device: str
    face_detector_loaded: bool
    emotion_model_loaded: bool

