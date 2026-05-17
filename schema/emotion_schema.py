from pydantic import BaseModel

class EmotionalMOdel(BaseModel):
    Angry: float
    Disgust: float
    Fear: float
    Happy: float
    Frustrated: float
    Neutral: float
    Sad: float

