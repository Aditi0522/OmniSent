## OmniSent

OmniSent is a multimodal sentiment analysis engine for usability studies. Analyzes facial expressions, voice prosody, and speech content simultaneously to map a user's emotional state throughout a usability session, fusing all three signals into a timestamped emotional timeline and actionable UX metrics.

# 🧠 OmniSent — Multimodal Sentiment Analysis Engine

> **Infer emotional states from usability testing videos through facial expressions and voice tone, generating actionable UX insights.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GSoC](https://img.shields.io/badge/GSoC-2026-blue?logo=google&logoColor=white)](https://summerofcode.withgoogle.com)

**Google Summer of Code 2026** · [AOSSIE](https://aossie.org) · Mentored under the RUXAILAB project

---

## 📖 Overview

OmniSent is a multimodal sentiment analysis engine designed for **usability studies**. It processes video recordings of usability testing sessions to analyze participants' emotional responses through multiple modalities — facial expressions, voice tone, and (upcoming) spoken text.

Unlike general-purpose emotion detection tools, OmniSent is purpose-built for UX research:

- **Temporal analysis** — tracks how emotions evolve *over time*, not just single-frame snapshots
- **UX-specific metrics** — translates raw emotions into actionable scores like frustration index, confusion score, and engagement level
- **Event detection** — automatically flags moments of frustration spikes, delight, or emotional shifts
- **Modular architecture** — each modality works independently and can be combined through late fusion

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
                         │   Usability Video    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │    FastAPI Backend   │
                         │      (main.py)       │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼─────────┐ ┌────────▼────────┐  ┌─────────▼─────────┐
    │  Facial Module     │ │  Voice Module   │  │  Text Module      │
    │                    │ │                 │  │  (Coming Soon)     │
    │  SCRFD Detection   │ │  ffmpeg Extract │  │                   │
    │  EfficientNet-B0   │ │  WavLM-Base+    │  │  Whisper STT      │
    │  7-class Emotion   │ │  7-class Emotion│  │  GoEmotions        │
    └─────────┬──────────┘ └────────┬────────┘  └─────────┬─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │    Shared Components       │
                      │                            │
                      │  Temporal Smoother          │
                      │  Emotion Timeline           │
                      │  Usability Metrics Calc     │
                      └─────────────┬──────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   JSON Response      │
                         │  • Emotion Summary   │
                         │  • UX Metrics        │
                         │  • Timeline Segments │
                         │  • Event Flags       │
                         └─────────────────────┘
```

---

## 🧑 Facial Module

Detects faces in video frames and classifies emotions at 10 FPS with 100ms temporal resolution.

| Component | Technology | Detail |
|---|---|---|
| Face Detection | SCRFD (InsightFace `buffalo_s`) | 640×640 input, largest face selection, 10% padded crop |
| Emotion Model | Fine-tuned EfficientNet-B0 | 7-class classifier, trained on 70,000+ images |
| Training Data | AffectNet + RAF-DB + FER+ + Custom | Webcam-specific augmentations (compression, blur, noise) |
| Post-processing | Temporal Smoother | Confidence-weighted moving average (window=5) |

**Emotion Classes:** `Angry` · `Disgusted` · `Fearful` · `Happy` · `Neutral` · `Sad` · `Surprised`

---

## 🎙️ Voice Module

Extracts audio from video and analyzes emotional tone in 2-second segments.

| Component | Technology | Detail |
|---|---|---|
| Audio Extraction | ffmpeg | Video → 16kHz mono PCM WAV |
| Emotion Model | Fine-tuned WavLM-Base-Plus | 7-class classifier, mean-pooled hidden states |
| Training Data | RAVDESS + CREMA-D + TESS | 17,500+ audio clips, 2-phase training |
| Segmentation | torchaudio | 2-second non-overlapping chunks with timestamp tracking |

---

## 📊 Usability Metrics

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

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **Facial ML** | PyTorch, timm (EfficientNet-B0), InsightFace (SCRFD), OpenCV |
| **Voice ML** | PyTorch, HuggingFace Transformers (WavLM), torchaudio |
| **Audio Processing** | ffmpeg, torchaudio |
| **Data Augmentation** | albumentations |
| **Training** | Kaggle (GPU), scikit-learn |

---

## 🚀 Quick Start

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

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/analyze/facial/health` | Facial module health check |
| `POST` | `/analyze/facial/video` | Upload video → facial emotion analysis |
| `GET` | `/analyze/voice/health` | Voice module health check |
| `POST` | `/analyze/voice/audio` | Upload audio → voice emotion analysis |
| `POST` | `/analyze/voice/from-video` | Upload video → extract audio → voice analysis |

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

## 📁 Project Structure

```
OmniSent/
├── main.py                          # FastAPI app, startup, routing
├── config.py                        # Centralized settings (.env support)
├── requirements.txt
│
├── api/
│   ├── facial.py                    # POST /analyze/facial/video
│   └── voice.py                     # POST /analyze/voice/audio, /from-video
│
├── facial_module/
│   ├── model.py                     # EmotiEffLibFineTuned (EfficientNet-B0)
│   ├── face_detector.py             # SCRFDFaceDetector (InsightFace)
│   ├── emotion_predictor.py         # Face crop → 7-class probabilities
│   ├── analyzer.py                  # FacialEmotionAnalyzer (orchestrator)
│   └── models/                      # Fine-tuned weights (.gitignored)
│
├── voice_module/
│   ├── model.py                     # WavLMEmotionClassifier
│   ├── audio_preprocessor.py        # ffmpeg, resample, segment
│   ├── emotion_predictor.py         # Waveform → 7-class probabilities
│   ├── analyzer.py                  # VoiceEmotionAnalyzer (orchestrator)
│   └── models/                      # Fine-tuned weights (.gitignored)
│
├── shared/
│   ├── constants.py                 # EMOTIONS list (single source of truth)
│   ├── temporal.py                  # TemporalSmoother + EmotionTimeline
│   └── usability_metrics.py         # UX scores + event detection
│
├── schema/
│   └── facial.py                    # Pydantic response models
│
├── training/                        # Offline training scripts
│   ├── fine_tuning.py
│   ├── benchmark.py
│   └── augmentation.py
│
└── tests/
```

---

## 📈 Current Progress

- [x] Facial sentiment pipeline (SCRFD + EfficientNet-B0)
- [x] Voice sentiment pipeline (WavLM-Base-Plus)
- [x] Temporal smoothing (confidence-weighted moving average)
- [x] Emotion timeline with segment detection
- [x] Usability metrics computation (6 composite scores)
- [x] Automatic UX event detection
- [x] FastAPI backend with REST endpoints
- [x] Model fine-tuning on 70,000+ images / 17,500+ audio clips
- [ ] Text sentiment module (Whisper + GoEmotions)
- [ ] Multimodal late fusion
- [ ] Dashboard / visualization frontend
- [ ] IEMOCAP dataset integration for voice
- [ ] End-to-end evaluation on usability study recordings

---

## 🗺️ Roadmap

| Phase | Timeline | Deliverable |
|---|---|---|
| ~~Facial Module~~ | ~~Week 1-3~~ | ~~SCRFD + EfficientNet-B0 pipeline~~ ✅ |
| ~~Voice Module~~ | ~~Week 3-5~~ | ~~WavLM pipeline + fine-tuning~~ ✅ |
| Text Module | Week 5-7 | Whisper STT → GoEmotions classification |
| Multimodal Fusion | Week 7-9 | Late fusion (confidence-weighted avg across modalities) |
| Dashboard | Week 9-11 | Visualization frontend for UX researchers |
| Evaluation | Week 11-12 | End-to-end testing on real usability study data |

---

## 🤝 Contributing

This project is developed as part of **Google Summer of Code 2026** under **AOSSIE**. Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-module`)
3. Commit your changes (`git commit -m 'Add new module'`)
4. Push to the branch (`git push origin feature/new-module`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Aditi Soni** — [GitHub](https://github.com/Aditi0522) · [LinkedIn](https://linkedin.com/in/aditi-soni)

GSoC 2026 Contributor @ AOSSIE

