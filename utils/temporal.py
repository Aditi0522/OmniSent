## Temporal smoothing is a technique used to reduce noise and short Term
## fluctuation in data collected over time by aggregating or averaging 
## values across a time window. This provides us with a better
## metric for emotion detection analysis.

import numpy as np
from facial_module.model import EMOTIONS
from collections import deque

class TemporalSmoother:
    def __init__(self, window_size = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen = window_size)

    def smooth(self,emotions:dict, confidence:float) -> tuple:
        self.buffer.append((emotions,confidence))
        smoothed = {e: 0.0 for e in EMOTIONS}
        conf_sum = 0.0
        for emo,conf in self.buffer:
            for emotion in EMOTIONS:
                smoothed[emotion] += emo.get(emotion,0) * conf
            conf_sum += conf
        if conf_sum>0:
            smoothed = {k:round(v/conf_sum,4) for k,v in smoothed.items()}
        smoothed_confidence = round(max(smoothed.values()),4)
        return smoothed,smoothed_confidence

    def reset(self):
        self.buffer.clear()

class EmotionTimeline:
    def __init__(self, smoother: TemporalSmoother):
        self.smoother = smoother
        self.raw_frames = []
        self.segments = []

    def add_frame(self, timestamp_ms: int, raw_emotions: dict, confidence: float):
        """Add one frame's raw prediction. Smoothing is applied internally."""
        smoothed_emotions, smoothed_confidence = self.smoother.smooth(
            raw_emotions, confidence
        )
        dominant = max(smoothed_emotions, key=smoothed_emotions.get)

        self.raw_frames.append({
            'timestamp_ms': timestamp_ms,
            'raw_emotions': raw_emotions,
            'smoothed_emotions': smoothed_emotions,
            'confidence': smoothed_confidence,
            'dominant_emotion': dominant,
        })
    def build_segments(self, min_segment_ms: int = 500) -> list:
        """
        Group consecutive same-emotion frames into segments.
        Merge micro-segments (< min_segment_ms) into neighbors.
        """
        if not self.raw_frames:
            return []

        segments = []
        cur = {
            'start_ms': self.raw_frames[0]['timestamp_ms'],
            'end_ms': self.raw_frames[0]['timestamp_ms'],
            'dominant_emotion': self.raw_frames[0]['dominant_emotion'],
            'avg_emotions': {e: [] for e in EMOTIONS},
            'avg_confidence': [],
        }

        for frame in self.raw_frames:
            if frame['dominant_emotion'] == cur['dominant_emotion']:
                cur['end_ms'] = frame['timestamp_ms']
                for e in EMOTIONS:
                    cur['avg_emotions'][e].append(frame['smoothed_emotions'].get(e, 0))
                cur['avg_confidence'].append(frame['confidence'])
            else:
                segments.append(self._finalize(cur))
                cur = {
                    'start_ms': frame['timestamp_ms'],
                    'end_ms': frame['timestamp_ms'],
                    'dominant_emotion': frame['dominant_emotion'],
                    'avg_emotions': {e: [frame['smoothed_emotions'].get(e, 0)] for e in EMOTIONS},
                    'avg_confidence': [frame['confidence']],
                }

        segments.append(self._finalize(cur))
        segments = self._merge_short(segments, min_segment_ms)
        self.segments = segments
        return segments

    def _finalize(self, seg):
        return {
            'start_ms': seg['start_ms'],
            'end_ms': seg['end_ms'],
            'duration_ms': seg['end_ms'] - seg['start_ms'],
            'dominant_emotion': seg['dominant_emotion'],
            'emotions': {e: round(float(np.mean(v)), 4) for e, v in seg['avg_emotions'].items()},
            'confidence': round(float(np.mean(seg['avg_confidence'])), 4),
        }

    def _merge_short(self, segments, min_ms):
        if len(segments) <= 1:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg['duration_ms'] < min_ms:
                merged[-1]['end_ms'] = seg['end_ms']
                merged[-1]['duration_ms'] = merged[-1]['end_ms'] - merged[-1]['start_ms']
            else:
                merged.append(seg)
        return merged

def aggregate_summary(timeline: EmotionTimeline) -> dict:
    """
    Compute a single overall emotion summary from all frames.
    High-confidence frames contribute more than low-confidence frames.
    """
    frames = timeline.raw_frames
    if not frames:
        return {'error': 'No frames processed'}

    weighted = {e: 0.0 for e in EMOTIONS}
    total_w = 0.0

    for f in frames:
        w = f['confidence']
        for e in EMOTIONS:
            weighted[e] += f['smoothed_emotions'].get(e, 0) * w
        total_w += w

    if total_w > 0:
        overall = {e: round(v / total_w, 4) for e, v in weighted.items()}
    else:
        overall = {e: round(1 / 7, 4) for e in EMOTIONS}

    dominant = max(overall, key=overall.get)

    # Time distribution: what % of total time was each emotion dominant?
    total_time = frames[-1]['timestamp_ms'] - frames[0]['timestamp_ms']
    time_dist = {e: 0 for e in EMOTIONS}
    for seg in timeline.segments:
        time_dist[seg['dominant_emotion']] += seg['duration_ms']
    if total_time > 0:
        time_dist = {e: round(t / total_time, 3) for e, t in time_dist.items()}

    return {
        'overall_emotions': overall,
        'dominant_emotion': dominant,
        'dominant_confidence': overall[dominant],
        'time_distribution': time_dist,
        'total_frames': len(frames),
        'total_duration_ms': total_time,
        'avg_confidence': round(float(np.mean([f['confidence'] for f in frames])), 3),
    }

