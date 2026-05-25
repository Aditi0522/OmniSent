import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.facial import router as facial_router, init_analyzer
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="OmniSent — Multimodal Sentiment Analysis Engine",
    description="Facial, voice, and text emotion analysis for usability studies",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Load models once on server start."""
    init_analyzer(
        model_path=settings.EMOTION_MODEL_PATH,
        process_fps=settings.PROCESS_FPS,
        smoothing_window=settings.SMOOTHING_WINDOW,
        min_segment_ms=settings.MIN_SEGMENT_MS,
        det_size=settings.DET_SIZE,
    )

app.include_router(facial_router)

@app.get("/")
async def root():
    return {"service": "OmniSent", "status": "running"}

