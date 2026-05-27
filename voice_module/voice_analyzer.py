import os
import time
import logging
import numpy as np
from utils.utils import EMOTIONS
from .audio_preprocessor import AudioPreprocessor
from .emotion_predictor import VoiceEmotionPredictor
from utils.usability_metrics import UsabilityMetricsCalculator
from utils.temporal import TemporalSmoother, EmotionTimeline, aggregate_summary

class VoiceEmotionAnalyzer:
    def __init__(self,model_path,segment_duration=2.0, smoothing_window=3, min_segment_ms=1000):
        self.logger = logging.getLogger(__name__)
        self.segment_duration = segment_duration
        self.smoothing_window = smoothing_window
        self.min_segment_ms = min_segment_ms
        self.preprocessor = AudioPreprocessor(segment_duration=segment_duration)
        self.predictor = VoiceEmotionPredictor(model_path=model_path)
        self.ux_calc = UsabilityMetricsCalculator()
        self.logger.info("Voice Emotion Analyzer initialized")

    def analyze_audio(self,audio_path) -> dict:
        start_time = time.time()
        waveform = self.preprocessor.load_audio(audio_path)
        segments = self.preprocessor.segment_audio(waveform)
        total_duration_ms = int((len(waveform)/self.preprocessor.target_sr) *1000)
        smoother = TemporalSmoother(window_size=self.smoothing_window)
        timeline = EmotionTimeline(smoother)

        prev_emotions = None
        all_ux_metrics = []
        all_ux_events = []

        predictions = self.predictor.predict_segments(segments)
        for pred in predictions:
            emotions = pred["emotions"]
            confidence = pred["confidence"]
            timestamp_ms = pred["timestamp_ms"]

            timeline.add_frame(timestamp_ms, emotions, confidence)

            ux = self.ux_calc.compute(emotions, confidence)
            all_ux_metrics.append(ux)

            events = self.ux_calc.detect_events(emotions, confidence, prev_emotions)
            for ev in events:
                ev["timestamp_ms"] = timestamp_ms
                all_ux_events.append(ev)
            prev_emotions = emotions
        processing_time = time.time() - start_time

        emotion_segments = timeline.build_segments(min_segment_ms=self.min_segment_ms)
        summary = aggregate_summary(timeline)

        avg_ux = {}
        if all_ux_metrics:
            for key in all_ux_metrics[0]:
                values = [m[key] for m in all_ux_metrics]
                avg_ux[key] = round(float(np.mean(values)), 4)
                avg_ux[f"{key}_max"] = round(float(np.max(values)), 4)
        return {
            "audio": audio_path,
            "config": {
                "segment_duration": self.segment_duration,
                "smoothing_window": self.smoothing_window,
                "min_segment_ms": self.min_segment_ms,
            },
            "processing": {
                "total_segments": len(segments),
                "total_duration_ms": total_duration_ms,
                "processing_time_sec": round(processing_time, 2),
            },
            "summary": summary,
            "usability_metrics": avg_ux,
            "usability_events": all_ux_events,
            "segments": emotion_segments,
            "segment_predictions": predictions,
        }

    def analyze_from_video(self,video_path) -> dict:
        audio_path = self.preprocessor.extract_audio_from_video(video_path)
        try:
            result = self.analyze_audio(audio_path)
            result["source_video"] = video_path
            return result
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
