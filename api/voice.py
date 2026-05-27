import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from schema.facial import VideoAnalysisResponse, FrameAnalysisResponse, HealthResponse
from voice_module.voice_analyzer import VoiceEmotionAnalyzer

router = APIRouter(prefix="/analyze/voice", tags=["Voice Analysis"])
logger = logging.getLogger(__name__)

analyzer: VoiceEmotionAnalyzer = None

def init_analyzer(model_path: str, **kwargs):
    global analyzer
    analyzer = VoiceEmotionAnalyzer(model_path=model_path, **kwargs)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if models are loaded and ready."""
    if analyzer is None:
        return {"status": "not_initialized"}
    return {
        "status": "ready",
        "device": str(analyzer.predictor.device),
        "model_loaded": analyzer.predictor.model is not None,
    }


@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...),
                        segment_duration: float = Query(2.0, ge=0.5, le=10.0),
                        smoothing_window: int = Query(3, ge=1, le=10),):
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Voice analyzer not initialized")
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        analyzer.segment_duration = segment_duration
        analyzer.smoothing_window = smoothing_window
        return analyzer.analyze_audio(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@router.post("/from-video")
async def analyze_from_video(
    file: UploadFile = File(...),
    segment_duration: float = Query(2.0, ge=0.5, le=10.0),
    smoothing_window: int = Query(3, ge=1, le=10),
):
    """Upload a video file — extracts audio then runs voice emotion analysis."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Voice analyzer not initialized")
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        analyzer.segment_duration = segment_duration
        analyzer.smoothing_window = smoothing_window
        return analyzer.analyze_from_video(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)




