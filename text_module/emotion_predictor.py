import logging
from utils.utils import EMOTIONS
from .model import TextEmotionClassifier

class TextEmotionPredictor:
    def __init__(self, model_name=None, device=None):
        self.logger = logging.getLogger(__name__)
        self.classifier = TextEmotionClassifier(
            model_name=model_name, device=device
        ) if model_name else TextEmotionClassifier(device=device)

    def predict(self, text: str) -> tuple[dict, float]:
        if not text or not text.strip():
            fallback = {e: round(1.0/7, 4) for e in EMOTIONS}
            return fallback, 0.0
        return self.classifier.predict(text)

    def predict_segments(self, segments: list[dict]) -> list[dict]:
        results = []
        for seg in segments:
            emotions, confidence = self.predict(seg["text"])
            dominant = max(emotions, key=emotions.get)
            results.append({
                "timestamp_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg["text"],
                "emotions": emotions,
                "confidence": confidence,
                "dominant_emotion": dominant,
            })
        return results

