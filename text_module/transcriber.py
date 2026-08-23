import logging
import whisper

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        self.logger = logging.getLogger(__name__)
        self.model = whisper.load_model(model_size)
        self.logger.info(f"Whisper {model_size} model loaded")

    def transcribe(self, audio_path: str) -> list[dict]:
        """Returns timestamped segments: [{"text": ..., "start_ms": ..., "end_ms": ...}]"""
        result = self.model.transcribe(audio_path, word_timestamps=False)
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
            })
        return segments

