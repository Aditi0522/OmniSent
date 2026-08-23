import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from schema.text import TextAnalysisResponse, TextHealthResponse
from text_module.text_analyzer import TextEmotionAnalyzer

router = APIRouter(prefix="/analyze/text", tags=["Text Analysis"])
logger = logging.getLogger(__name__)

analyzer: TextEmotionAnalyzer = None


def init_analyzer(**kwargs):
    global analyzer
    analyzer = TextEmotionAnalyzer(**kwargs)


@router.get("/health", response_model=TextHealthResponse)
async def health_check():
    if analyzer is None:
        return TextHealthResponse(
            status="not_initialized",
            whisper_loaded=False,
            classifier_loaded=False,
        )
    return TextHealthResponse(
        status="ready",
        whisper_loaded=analyzer.transcriber.model is not None,
        classifier_loaded=analyzer.predictor.classifier.model is not None,
    )


@router.post("/from-audio", response_model=TextAnalysisResponse)
async def analyze_from_audio(
    file: UploadFile = File(...),
    smoothing_window: int = Query(3, ge=1, le=10),
):
    """Upload an audio file — transcribes with Whisper then runs text emotion analysis."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Text analyzer not initialized")

    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        analyzer.smoothing_window = smoothing_window
        return analyzer.analyze_from_audio(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/from-video", response_model=TextAnalysisResponse)
async def analyze_from_video(
    file: UploadFile = File(...),
    smoothing_window: int = Query(3, ge=1, le=10),
):
    """Upload a video — extracts audio, transcribes, then runs text emotion analysis."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Text analyzer not initialized")

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        analyzer.smoothing_window = smoothing_window
        return analyzer.analyze_from_video(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

