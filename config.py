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

    model_config = SettingsConfigDict(
            env_file = ".env",
            extra = "ignore"
            )

settings = Settings()


