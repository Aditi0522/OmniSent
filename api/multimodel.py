import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from schema.multimodal import MultimodalResponse
from fusion.fusion_analyzer import MultimodalFusionAnalyzer

# Import the module-level analyzers from each API
from api import facial as facial_api
from api import voice as voice_api
from api import text as text_api

router = APIRouter(prefix="/analyze", tags=["Multimodal Analysis"])
logger = logging.getLogger(__name__)

fusion_analyzer: MultimodalFusionAnalyzer = None


def init_fusion(**kwargs):
    global fusion_analyzer
    fusion_analyzer = MultimodalFusionAnalyzer(**kwargs)


@router.post("/multimodal", response_model=MultimodalResponse)
async def analyze_multimodal(
    file: UploadFile = File(...),
    process_fps: int = Query(10, ge=1, le=30),
    segment_duration: float = Query(2.0, ge=0.5, le=10.0),
    window_ms: int = Query(2000, ge=500, le=10000),
    include_per_modality: bool = Query(False),
):
    """
    Upload a video file and run all available modalities (face, voice, text).
    Results are fused into a single unified emotional timeline.
    Gracefully degrades if a modality fails (e.g., no face detected, no audio track).
    """
    if fusion_analyzer is None:
        raise HTTPException(status_code=503, detail="Fusion analyzer not initialized")

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use: {allowed}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        facial_result = None
        voice_result = None
        text_result = None

        # ── Run facial module ──
        if facial_api.analyzer is not None:
            try:
                facial_api.analyzer.process_fps = process_fps
                facial_result = facial_api.analyzer.analyze_video(tmp_path)
                if "error" in facial_result:
                    logger.warning(f"Facial module returned error: {facial_result['error']}")
                    facial_result = None
            except Exception as e:
                logger.warning(f"Facial module failed (non-fatal): {e}")

        # ── Run voice module ──
        if voice_api.analyzer is not None:
            try:
                voice_api.analyzer.segment_duration = segment_duration
                voice_result = voice_api.analyzer.analyze_from_video(tmp_path)
            except Exception as e:
                logger.warning(f"Voice module failed (non-fatal): {e}")

        # ── Run text module ──
        if text_api.analyzer is not None:
            try:
                text_result = text_api.analyzer.analyze_from_video(tmp_path)
            except Exception as e:
                logger.warning(f"Text module failed (non-fatal): {e}")

        # ── Check at least one modality succeeded ──
        if not any([facial_result, voice_result, text_result]):
            raise HTTPException(
                status_code=422,
                detail="All modalities failed. Check logs for details.",
            )

        # ── Fuse ──
        fusion_analyzer.window_ms = window_ms
        result = fusion_analyzer.fuse(
            facial_result=facial_result,
            voice_result=voice_result,
            text_result=text_result,
        )

        result["source_video"] = file.filename

        if include_per_modality:
            result["per_modality"] = {
                "facial": facial_result,
                "voice": voice_result,
                "text": text_result,
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

