import os
import torch
import tempfile
import logging
import torchaudio
import subprocess

TARGET_SR = 16000

class AudioPreprocessor:
    def __init__(self, target_sr=TARGET_SR, segment_duration=2.0):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.logger = logging.getLogger(__name__)

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video to a temp .wav file using ffmpeg."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = temp_file.name
        temp_file.close()

        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.target_sr),
            "-ac", "1",
            audio_path,
            "-y",
        ]

        try:
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            raise RuntimeError(f"ffmpeg failed: {e}")

        self.logger.info(f"Audio extracted to {audio_path}")
        return audio_path

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file, convert to mono 16kHz, return 1D tensor."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def segment_audio(self, waveform: torch.Tensor) -> list[dict]:
        """
        Split waveform into fixed-length chunks with timestamps.
        Returns: [{"waveform": tensor, "timestamp_ms": int}, ...]
        """
        segment_len = int(self.target_sr * self.segment_duration)
        segments = []

        for i in range(0, len(waveform), segment_len):
            chunk = waveform[i:i + segment_len]

            # Pad last chunk if shorter
            if len(chunk) < segment_len:
                chunk = torch.nn.functional.pad(chunk, (0, segment_len - len(chunk)))

            timestamp_ms = int((i / self.target_sr) * 1000)

            segments.append({
                "waveform": chunk,
                "timestamp_ms": timestamp_ms,
            })

        return segments

    def pad_or_truncate(self, waveform: torch.Tensor, max_length: int) -> torch.Tensor:
        if waveform.shape[0] > max_length:
            return waveform[:max_length]
        elif waveform.shape[0] < max_length:
            return torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[0]))
        return waveform




