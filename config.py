from os import environ
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    MODEL_PATH: str

    EMOTION_MODEL_PATH: Path = BASE_DIR/ "facial_module" /"models" / "best_model.pt"

    model_config = SettingsConfigDict(
            env_file = ".env",
            extra = "ignore"
            )

settings = Settings()


