import os, time, logging
import numpy as np
from .transcriber import WhisperTranscriber
from .emotion_predictor import TextEmotionPredictor
from utils.temporal import TemporalSmoother, EmotionTimeline, aggregate_summary
from utils.usability_metrics import UsabilityMetricsCalculator

class TextEmotionAnalyzer:
    def __init__(self, model_name=None, whisper_size="base",
                 smoothing_window=3, min_segment_ms=2000):
        self.logger = logging.getLogger(__name__)
        self.smoothing_window = smoothing_window
        self.min_segment_ms = min_segment_ms
        self.transcriber = WhisperTranscriber(model_size=whisper_size)
        self.predictor = TextEmotionPredictor(model_name=model_name)
        self.ux_calc = UsabilityMetricsCalculator()
        self.logger.info("TextEmotionAnalyzer initialized")

    def analyze_transcript(self, transcript_segments: list[dict]) -> dict:
        start_time = time.time()
        smoother = TemporalSmoother(window_size=self.smoothing_window)
        timeline = EmotionTimeline(smoother)
        prev_emotions = None
        all_ux_metrics, all_ux_events = [], []

        predictions = self.predictor.predict_segments(transcript_segments)
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
            "config": {"smoothing_window": self.smoothing_window,
                       "min_segment_ms": self.min_segment_ms},
            "processing": {"total_segments": len(transcript_segments),
                           "processing_time_sec": round(processing_time, 2)},
            "summary": summary,
            "usability_metrics": avg_ux,
            "usability_events": all_ux_events,
            "segments": emotion_segments,
            "segment_predictions": predictions,
        }

    def analyze_from_audio(self, audio_path: str) -> dict:
        transcript = self.transcriber.transcribe(audio_path)
        result = self.analyze_transcript(transcript)
        result["source_audio"] = audio_path
        result["transcript"] = transcript
        return result

    def analyze_from_video(self, video_path: str) -> dict:
        from voice_module.audio_preprocessor import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        audio_path = preprocessor.extract_audio_from_video(video_path)
        try:
            result = self.analyze_from_audio(audio_path)
            result["source_video"] = video_path
            return result
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

