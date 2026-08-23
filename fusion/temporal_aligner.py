from utils.utils import EMOTIONS
import numpy as np

class TemporalAligner:
    """Aligns predictions from different modalities to common time windows."""

    def align(self, facial_preds: list, voice_preds: list, text_preds: list,
              window_ms: int = 2000) -> list[dict]:
        """
        Each pred list has dicts with 'timestamp_ms' and 'emotions'.
        Returns windows with aligned predictions from all available modalities.
        """
        # Find time range
        all_ts = []
        for preds in [facial_preds, voice_preds, text_preds]:
            if preds:
                all_ts.extend([p["timestamp_ms"] for p in preds])
        if not all_ts:
            return []

        start = min(all_ts)
        end = max(all_ts)
        windows = []

        for t in range(start, end + 1, window_ms):
            window = {"start_ms": t, "end_ms": t + window_ms, "modalities": {}}
            for name, preds in [("facial", facial_preds), ("voice", voice_preds),
                                ("text", text_preds)]:
                if not preds:
                    continue
                matched = [p for p in preds if t <= p["timestamp_ms"] < t + window_ms]
                if matched:
                    avg_emotions = {e: 0.0 for e in EMOTIONS}
                    avg_conf = 0.0
                    for m in matched:
                        for e in EMOTIONS:
                            avg_emotions[e] += m["emotions"].get(e, 0)
                        avg_conf += m.get("confidence", 0)
                    n = len(matched)
                    avg_emotions = {e: round(v/n, 4) for e, v in avg_emotions.items()}
                    avg_conf = round(avg_conf / n, 4)
                    window["modalities"][name] = {
                        "emotions": avg_emotions, "confidence": avg_conf
                    }
            if window["modalities"]:
                windows.append(window)
        return windows

