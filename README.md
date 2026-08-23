# 🧠 OmniSent — Multimodal Sentiment Analysis Engine

OmniSent is a multimodal sentiment analysis engine for usability studies. Analyzes facial expressions, voice prosody, and speech content simultaneously to map a user's emotional state throughout a usability session, fusing all three signals into a timestamped emotional timeline and actionable UX metrics.

> **Infer emotional states from usability testing videos through facial expressions, voice tone, and spoken text — generating actionable UX insights.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![GSoC](https://img.shields.io/badge/GSoC-2026-fbbc04?logo=google&logoColor=white)](https://summerofcode.withgoogle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

OmniSent processes video recordings of usability testing sessions to analyze participants' emotional responses through **three modalities** — facial expressions, voice tone, and spoken text — then fuses them into a single unified emotional timeline.

Unlike general-purpose emotion detection tools, OmniSent is purpose-built for UX research:

- **Multimodal fusion** — combines face, voice, and text signals with confidence-weighted late fusion for more accurate analysis than any single modality
- **Temporal analysis** — tracks how emotions evolve *over time*, not just single-frame snapshots
- **UX-specific metrics** — translates raw emotions into actionable scores like frustration index, confusion score, and engagement level
- **Event detection** — automatically flags moments of frustration spikes, delight, or emotional shifts
- **Graceful degradation** — if a modality fails (no face detected, no audio), the remaining modalities still produce results

---

## Architecture

```
                         ┌─────────────────────┐
                         │   Usability Video   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │    FastAPI Backend  │
                         │      (main.py)      │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼─────────┐ ┌────────▼────────┐  ┌─────────▼─────────┐
    │  Facial Module     │ │  Voice Module   │  │  Text Module      │
    │                    │ │                 │  │  (Coming Soon)    │
    │  SCRFD Detection   │ │  ffmpeg Extract │  │                   │
    │  EfficientNet-B0   │ │  WavLM-Base+    │  │  Whisper STT      │
    │  7-class Emotion   │ │  7-class Emotion│  │  GoEmotions       │
    └─────────┬──────────┘ └────────┬────────┘  └─────────┬─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │    Shared Components      │
                      │                           │
                      │  Temporal Smoother        │
                      │  Emotion Timeline         │
                      │  Usability Metrics Calc   │
                      └─────────────┬─────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   JSON Response     │
                         │  • Emotion Summary  │
                         │  • UX Metrics       │
                         │  • Timeline Segments│
                         │  • Event Flags      │
                         └─────────────────────┘
```

---

## Facial Module

Detects faces in video frames and classifies emotions at 10 FPS with 100ms temporal resolution.

| Component | Technology | Detail |
|---|---|---|
| Face Detection | SCRFD (InsightFace `buffalo_s`) | 640×640 input, largest face selection, 10% padded crop |
| Emotion Model | Fine-tuned EfficientNet-B0 | 7-class classifier, trained on 70,000+ images |
| Training Data | AffectNet + RAF-DB + FER+ + Custom | Webcam-specific augmentations (compression, blur, noise) |
| Post-processing | Temporal Smoother | Confidence-weighted moving average (window=5) |

**Emotion Classes:** `Angry` · `Disgusted` · `Fearful` · `Happy` · `Neutral` · `Sad` · `Surprised`

---

## Voice Module

Extracts audio from video and analyzes emotional tone in 2-second segments.

| Component | Technology | Detail |
|---|---|---|
| Audio Extraction | ffmpeg | Video → 16kHz mono PCM WAV |
| Emotion Model | Fine-tuned WavLM-Base-Plus | 7-class classifier, mean-pooled hidden states |
| Training Data | RAVDESS + CREMA-D + TESS | 17,500+ audio clips, 2-phase training |
| Segmentation | torchaudio | 2-second non-overlapping chunks with timestamp tracking |

### Text Module
Transcribes speech from audio and classifies emotional content per utterance.
| Component | Technology | Detail |
|---|---|---|
| Speech-to-Text | OpenAI Whisper | Timestamped transcript segments |
| Emotion Model | DistilRoBERTa (HuggingFace) | 7-class classifier, pretrained on emotion data |
| Input | Whisper transcript segments | Each utterance analyzed independently |
| Post-processing | Temporal Smoother | Same shared pipeline as facial and voice |
### Multimodal Fusion
Combines predictions from all three modalities into a unified emotional timeline.
| Component | Method | Detail |
|---|---|---|
| Temporal Alignment | Window-based | Aligns face (10fps), voice (2s), text (variable) to common 2-second windows |
| Fusion Strategy | Confidence-weighted late fusion | Higher-confidence modalities automatically dominate |
| Missing Modalities | Graceful degradation | Weight renormalization when a modality is absent |
**Emotion Classes (all modules):** `Angry` · `Disgusted` · `Fearful` · `Happy` · `Neutral` · `Sad` · `Surprised`

## Usability Metrics

Raw 7-class emotions are transformed into composite UX scores:

| Metric | Formula | Use Case |
|---|---|---|
| Frustration Index | `Angry×0.5 + Disgusted×0.3 + Sad×0.2` | Identifies pain points in UI flow |
| Confusion Score | `Fearful×0.4 + Surprised×0.3 + (1−conf)×0.3` | Flags unclear navigation or instructions |
| Engagement Level | `Happy×0.3 + Surprised×0.3 + (1−Neutral)×0.4` | Measures active user involvement |
| Satisfaction Score | `Happy×0.6 + Neutral×0.3×conf + Surprised×0.1` | Overall positive experience indicator |
| Boredom Score | `(Neutral×0.5 + Sad×0.3 + Disgusted×0.2) × (1−intensity)` | Detects disengagement |
| Stress Level | `Fearful×0.5 + Angry×0.3 + Surprised×0.2` | Identifies high-pressure moments |

**Automatic Event Detection:** Frustration spikes, confusion moments, delight peaks, and significant emotional shifts are flagged with timestamps.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **Facial ML** | PyTorch, timm (EfficientNet-B0), InsightFace (SCRFD), OpenCV, ONNX Runtime |
| **Voice ML** | PyTorch, HuggingFace Transformers (WavLM-Base-Plus), torchaudio |
| **Text ML** | HuggingFace Transformers (DistilRoBERTa), OpenAI Whisper |
| **Audio Processing** | ffmpeg, torchaudio |
| **Data Augmentation** | albumentations |
| **Evaluation** | scikit-learn (classification report, confusion matrix, F1 score) |
| **Training Infra** | Kaggle (P100/T4 GPU) |

## Quick Start

### Prerequisites

- Python 3.10+
- ffmpeg installed (`sudo apt install ffmpeg`)
- Model weights (see [Model Setup](#model-setup))

### Installation

```bash
git clone https://github.com/Aditi0522/OmniSent.git
cd OmniSent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model Setup

Place your fine-tuned model weights:

```
facial_module/models/best_model.pt
voice_module/models/best_voice_model.pt
```

### Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive API documentation.

---
## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/analyze/facial/health` | Facial module health check |
| POST | `/analyze/facial/video` | Upload video → facial emotion analysis |
| GET | `/analyze/voice/health` | Voice module health check |
| POST | `/analyze/voice/audio` | Upload audio → voice emotion analysis |
| POST | `/analyze/voice/from-video` | Upload video → extract audio → voice analysis |
| GET | `/analyze/text/health` | Text module health check |
| POST | `/analyze/text/from-audio` | Upload audio → transcribe → text emotion analysis |
| POST | `/analyze/text/from-video` | Upload video → extract audio → transcribe → text analysis |
| POST | `/analyze/multimodal` | Upload video → run all modalities → fused analysis |

### Example — Analyze a Video (Facial)

```bash
curl -X POST "http://localhost:8000/analyze/facial/video?process_fps=10" \
     -F "file=@usability_session.mp4"
```

<details>
<summary><strong>📋 Sample Response</strong></summary>

```json
{
  "video": "usability_session.mp4",
  "config": {
    "process_fps": 10,
    "smoothing_window": 5,
    "min_segment_ms": 500
  },
  "processing": {
    "total_frames_processed": 1200,
    "faces_detected": 1150,
    "faces_missed": 50,
    "detection_rate": 0.9583,
    "processing_time_sec": 12.34
  },
  "summary": {
    "overall_emotions": {
      "Angry": 0.0412,
      "Disgusted": 0.0198,
      "Fearful": 0.0567,
      "Happy": 0.3241,
      "Neutral": 0.4102,
      "Sad": 0.0834,
      "Surprised": 0.0646
    },
    "dominant_emotion": "Neutral",
    "dominant_confidence": 0.8234
  },
  "usability_metrics": {
    "frustration_index": 0.0453,
    "confusion_score": 0.1234,
    "engagement_level": 0.4521,
    "satisfaction_score": 0.5678,
    "boredom_score": 0.2341,
    "stress_level": 0.0891
  },
  "usability_events": [
    {
      "type": "frustration_spike",
      "timestamp_ms": 45200,
      "severity": "medium"
    }
  ],
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 15000,
      "duration_ms": 15000,
      "dominant_emotion": "Neutral",
      "confidence": 0.82
    }
  ]
}
```
</details>

---

## Contributing

Please do! if you find this project up ur alley or u simply find this to be ur niche, would love to collaborate with anyone interested. Kindly adhere to the contribution guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-module`)
3. Commit your changes (`git commit -m 'Add new module'`)
4. Push to the branch (`git push origin feature/new-module`)
5. Open a Pull Request

---

## Author

**Aditi Soni** — [GitHub](https://github.com/Aditi0522) · [LinkedIn](https://www.linkedin.com/in/aditisoni05/)

