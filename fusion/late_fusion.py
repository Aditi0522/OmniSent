from utils.utils import EMOTIONS
import numpy as np

class LateFusionEngine:
    """Confidence-weighted late fusion of modality predictions."""

    def __init__(self, default_weights=None):
        # Default equal weights; can be learned via grid search later
        self.default_weights = default_weights or {
            "facial": 1.0, "voice": 1.0, "text": 1.0
        }

    def fuse_window(self, modalities: dict) -> tuple[dict, float]:
        """
        modalities: {"facial": {"emotions": {...}, "confidence": float}, ...}
        Returns: fused_emotions, fused_confidence
        """
        fused = {e: 0.0 for e in EMOTIONS}
        total_weight = 0.0

        for mod_name, mod_data in modalities.items():
            w = self.default_weights.get(mod_name, 1.0) * mod_data["confidence"]
            for e in EMOTIONS:
                fused[e] += mod_data["emotions"].get(e, 0) * w
            total_weight += w

        if total_weight > 0:
            fused = {e: round(v / total_weight, 4) for e, v in fused.items()}

        confidence = round(max(fused.values()), 4) if fused else 0.0
        return fused, confidence

