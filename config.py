from os import environ
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    EMOTION_MODEL_PATH: Path = BASE_DIR/ "facial_module" /"models" / "best_model.pt"
    PROCESS_FPS: int = 5
    SMOOTHING_WINDOW: int = 10
    MIN_SEGMENT_MS: int = 500
    DET_SIZE: tuple[int,int] = (640,640)

    VOICE_MODEL_PATH: Path = BASE_DIR/ "voice_module" / "models" / "best_model.pt"
    SEGMENT_DURATION: float = 2.0
    VOICE_SMOOTHING_WINDOW: int = 3

    TEXT_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    WHISPER_SIZE: str = "base"
    TEXT_SMOOTHING_WINDOW: int = 3
    TEXT_MIN_SEGMENT_MS: int = 2000

    FUSION_WINDOW_MS: int = 2000
    FUSION_SMOOTHING_WINDOW: int = 5
    FUSION_MIN_SEGMENT_MS: int = 1000

    model_config = SettingsConfigDict(
            env_file = ".env",
            extra = "ignore"
            )

settings = Settings()

