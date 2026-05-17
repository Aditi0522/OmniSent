from pydantic import BaseModel

class ModalityOutput(BaseModel):
    timestamp: float
    duration: float
    emotions: dict
    confidence: float
    modality: str
    face_detected: bool

