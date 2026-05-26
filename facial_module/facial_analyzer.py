## Heart of the module, class which links all the functions related
## to facial emotion detection.load_model

## 1. Open video
## 2. Create TemporalSmoother + EmotionTimeline + UsabilityMetricsCalculator
## 3. Frame loop:
##   a. Read frame, skip based on process_fps
##   b. Detect face (SCRFD) → get face_crop
##   c. Predict emotions (your EfficientNet-B0) → 7-class probabilities
##   d. Feed into timeline (smoothing happens inside)
##   e. Compute UX metrics for this frame
##   f. Detect UX events (frustration spike, etc.)
## 4. Build segments from timeline
## 5. Aggregate summary
## 6. Return the full result dict

## Orchestrates face detection, emotion prediction, temporal
## smoothing, and usability metrics into a single pipeline.

import cv2
import time
import logging
import numpy as np
from .face_detector import SCRFDFaceDetector
from .emotion_predictor import EmotionPredictor
from utils.temporal import TemporalSmoother, EmotionTimeline, aggregate_summary
from utils.usability_metrics import UsabilityMetricsCalculator


class FacialEmotionAnalyzer:
    def __init__(self, model_path: str, process_fps=10,
                 smoothing_window=5, min_segment_ms=500, det_size=(640,640)):
        self.logger = logging.getLogger(__name__)
        self.process_fps = process_fps
        self.smoothing_window = smoothing_window
        self.min_segment_ms = min_segment_ms

        self.face_detector = SCRFDFaceDetector(det_size=det_size)
        self.emotion_predictor = EmotionPredictor(model_path=model_path)
        self.ux_calc = UsabilityMetricsCalculator()
        self.logger.info("FacialEmotionAnalyzer initialized")

    def analyze_video(self, video_path: str) -> dict:
        """Full pipeline: video → face detection → emotion → smoothing → UX metrics → JSON."""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return {"error": f"Cannot open video: {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / self.process_fps))

        # Fresh smoother + timeline for each video
        smoother = TemporalSmoother(window_size=self.smoothing_window)
        timeline = EmotionTimeline(smoother)

        frame_idx = 0
        processed = 0
        faces_found = 0
        faces_missed = 0
        prev_emotions = None
        all_ux_metrics = []
        all_ux_events = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp_ms = int((frame_idx / video_fps) * 1000)
            detected = self.face_detector.detect_faces(frame)

            if detected:
                faces_found += 1
                face_crop = detected[0]["face_crop"]

                emotions, confidence = self.emotion_predictor.predict_emotion(face_crop)
                timeline.add_frame(timestamp_ms, emotions, confidence)

                ux = self.ux_calc.compute(emotions, confidence)
                all_ux_metrics.append(ux)

                events = self.ux_calc.detect_events(emotions, confidence, prev_emotions)
                for ev in events:
                    ev["timestamp_ms"] = timestamp_ms
                    all_ux_events.append(ev)

                prev_emotions = emotions
            else:
                faces_missed += 1

            processed += 1
            frame_idx += 1

        cap.release()
        processing_time = time.time() - start_time

        # Build segments + summary
        segments = timeline.build_segments(min_segment_ms=self.min_segment_ms)
        summary = aggregate_summary(timeline)

        # Average UX metrics
        avg_ux = {}
        if all_ux_metrics:
            for key in all_ux_metrics[0]:
                values = [m[key] for m in all_ux_metrics]
                avg_ux[key] = round(float(np.mean(values)), 4)
                avg_ux[f"{key}_max"] = round(float(np.max(values)), 4)

        return {
            "video": video_path,
            "config": {
                "process_fps": self.process_fps,
                "smoothing_window": self.smoothing_window,
                "min_segment_ms": self.min_segment_ms,
            },
            "processing": {
                "total_frames_processed": processed,
                "faces_detected": faces_found,
                "faces_missed": faces_missed,
                "detection_rate": round(faces_found / max(processed, 1), 4),
                "processing_time_sec": round(processing_time, 2),
            },
            "summary": summary,
            "usability_metrics": avg_ux,
            "usability_events": all_ux_events,
            "segments": segments,
            "frame_predictions": [
                {
                    "timestamp_ms": f["timestamp_ms"],
                    "smoothed_emotions": f["smoothed_emotions"],
                    "dominant_emotion": f["dominant_emotion"],
                    "confidence": f["confidence"],
                }
                for f in timeline.raw_frames
            ],
        }

    def analyze_frame(self, frame) -> dict:
        """Single-frame analysis, no temporal smoothing."""
        detected = self.face_detector.detect_faces(frame)
        if not detected:
            return {"error": "No face detected"}

        face_crop = detected[0]["face_crop"]
        emotions, confidence = self.emotion_predictor.predict_emotion(face_crop)
        ux = self.ux_calc.compute(emotions, confidence)
        dominant = max(emotions, key=emotions.get)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant,
            "confidence": round(confidence, 4),
            "det_score": detected[0]["det_score"],
            "usability_metrics": ux,
        }

