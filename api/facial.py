import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from schema.facial import VideoAnalysisResponse, FrameAnalysisResponse, HealthResponse
from facial_module.facial_analyzer import FacialEmotionAnalyzer

router = APIRouter(prefix="/analyze/facial", tags=["Facial Analysis"])
logger = logging.getLogger(__name__)

analyzer: FacialEmotionAnalyzer = None


def init_analyzer(model_path: str, **kwargs):
    global analyzer
    analyzer = FacialEmotionAnalyzer(model_path=model_path, **kwargs)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if models are loaded and ready."""
    if analyzer is None:
        return HealthResponse(
            status="not_initialized",
            device="unknown",
            face_detector_loaded=False,
            emotion_model_loaded=False,
        )
    return HealthResponse(
        status="ready",
        device=str(analyzer.emotion_predictor.device),
        face_detector_loaded=analyzer.face_detector.model is not None,
        emotion_model_loaded=analyzer.emotion_predictor.model is not None,
    )


@router.post("/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    process_fps: int = Query(10, ge=1, le=30),
    smoothing_window: int = Query(5, ge=1, le=20),
    min_segment_ms: int = Query(500, ge=100, le=5000),
):
    """Upload a video file and get full facial emotion analysis."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")

    # Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Override config if user passed custom values
        analyzer.process_fps = process_fps
        analyzer.smoothing_window = smoothing_window
        analyzer.min_segment_ms = min_segment_ms

        result = analyzer.analyze_video(tmp_path)

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

