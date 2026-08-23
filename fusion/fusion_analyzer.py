import time, logging
import numpy as np
from .temporal_aligner import TemporalAligner
from .late_fusion import LateFusionEngine
from utils.temporal import TemporalSmoother, EmotionTimeline, aggregate_summary
from utils.usability_metrics import UsabilityMetricsCalculator

class MultimodalFusionAnalyzer:
    def __init__(self, smoothing_window=5, min_segment_ms=1000, window_ms=2000):
        self.logger = logging.getLogger(__name__)
        self.smoothing_window = smoothing_window
        self.min_segment_ms = min_segment_ms
        self.window_ms = window_ms
        self.aligner = TemporalAligner()
        self.fusion_engine = LateFusionEngine()
        self.ux_calc = UsabilityMetricsCalculator()

    def fuse(self, facial_result: dict = None, voice_result: dict = None,
             text_result: dict = None) -> dict:
        start_time = time.time()

        # Extract frame/segment predictions from each module's result
        facial_preds = self._extract_preds(facial_result, "frame_predictions",
                                            "smoothed_emotions")
        voice_preds = self._extract_preds(voice_result, "segment_predictions",
                                           "emotions")
        text_preds = self._extract_preds(text_result, "segment_predictions",
                                          "emotions")

        modalities_present = []
        if facial_preds: modalities_present.append("facial")
        if voice_preds: modalities_present.append("voice")
        if text_preds: modalities_present.append("text")

        if not modalities_present:
            return {"error": "No modality data provided"}

        # Align to common time windows
        windows = self.aligner.align(facial_preds, voice_preds, text_preds,
                                      self.window_ms)

        # Fuse + build timeline
        smoother = TemporalSmoother(window_size=self.smoothing_window)
        timeline = EmotionTimeline(smoother)
        prev_emotions = None
        all_ux_metrics, all_ux_events = [], []
        fused_windows = []

        for win in windows:
            fused_emotions, fused_conf = self.fusion_engine.fuse_window(
                win["modalities"])
            timeline.add_frame(win["start_ms"], fused_emotions, fused_conf)

            ux = self.ux_calc.compute(fused_emotions, fused_conf)
            all_ux_metrics.append(ux)
            events = self.ux_calc.detect_events(fused_emotions, fused_conf,
                                                 prev_emotions)
            for ev in events:
                ev["timestamp_ms"] = win["start_ms"]
                all_ux_events.append(ev)
            prev_emotions = fused_emotions

            fused_windows.append({
                "start_ms": win["start_ms"], "end_ms": win["end_ms"],
                "fused_emotions": fused_emotions, "confidence": fused_conf,
                "dominant_emotion": max(fused_emotions, key=fused_emotions.get),
                "modalities_used": list(win["modalities"].keys()),
            })

        segments = timeline.build_segments(min_segment_ms=self.min_segment_ms)
        summary = aggregate_summary(timeline)

        avg_ux = {}
        if all_ux_metrics:
            for key in all_ux_metrics[0]:
                values = [m[key] for m in all_ux_metrics]
                avg_ux[key] = round(float(np.mean(values)), 4)
                avg_ux[f"{key}_max"] = round(float(np.max(values)), 4)

        return {
            "config": {"window_ms": self.window_ms,
                       "smoothing_window": self.smoothing_window},
            "processing": {"modalities_used": modalities_present,
                           "total_windows": len(fused_windows),
                           "processing_time_sec": round(time.time()-start_time, 2)},
            "summary": summary,
            "usability_metrics": avg_ux,
            "usability_events": all_ux_events,
            "segments": segments,
            "fused_windows": fused_windows,
        }

    def _extract_preds(self, result, key, emotions_key):
        if not result or key not in result:
            return []
        preds = []
        for p in result[key]:
            preds.append({
                "timestamp_ms": p.get("timestamp_ms", 0),
                "emotions": p.get(emotions_key, p.get("emotions", {})),
                "confidence": p.get("confidence", 0),
            })
        return preds

