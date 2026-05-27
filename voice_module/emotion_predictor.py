import torch
import logging
from utils.utils import EMOTIONS
from torch.cuda import is_available
from .model import load_voice_model

class VoiceEmotionPredictor:
    def __init__(self,model_path,device = None):
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_voice_model(model_path,self.device)
    def predict_emotion(self, waveform_tensor) -> tuple[dict,float]:
        try:
            input_values = waveform_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_values)
                probs = torch.softmax(logits, dim =1)[0]
            emotions = {
                    EMOTIONS[i]: round(float(probs[i]),4)
                    for i in range(len(EMOTIONS))
                    }
            confidence = float(probs.max())
            return emotions, confidence
        except Exception as e:
            self.logger.error(f"Error prediction emotion from voice module: {e}")
            fallback = {em: round(1.0 / 7, 4) for em in EMOTIONS}
            return fallback, 0.0
    def predict_segment(self,segments) -> list[dict]:
        results = []
        for seg in segments:
            emotions, confidence = self.predict_emotion(seg["waveform"])
            dominant = max(emotions, key=emotions.get)
            results.append({
                "timestamp_ms": seg["timestamp_ms"],
                "emotions": emotions,
                "confidence": confidence,
                "dominant_emotion": dominant,
            })
        return results
