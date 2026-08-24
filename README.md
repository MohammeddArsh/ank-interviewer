# Ank — AI Mock Interviewer

[![CI](https://github.com/MohammeddArsh/ank-interviewer/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammeddArsh/ank-interviewer/actions/workflows/ci.yml)

> A full-stack AI mock interviewer with a voice interface — paste a **job description** and your **resume**, and a realistic AI interviewer conducts a natural, sectioned interview with follow-up questions and a full scorecard at the end. Powered by **OpenRouter free-tier open-source LLMs** + local faster-whisper.
>
> Pick an interviewer persona, watch them speak (animated avatar with lip-sync or a live circular waveform), answer by voice, and get an honest evaluation.

![Ank Demo](docs/demo.png)

---

## Live Demo

| Platform | Link | Status |
|---|---|---|
| ☁️ **Render** (primary) | [ank-voice-assistant.onrender.com](https://ank-voice-assistant.onrender.com) | Always on |
| ☁️ **Azure Container Apps** (CI/CD) | [ank-voice-assistant.kindriver-c8b791e4.germanywestcentral.azurecontainerapps.io](https://ank-voice-assistant.kindriver-c8b791e4.germanywestcentral.azurecontainerapps.io) | Docker image deployed via GitHub Actions — scales to zero on the free tier |

> **Note:** Render is the primary deployment, kept always on using UptimeRobot health checks. Railway is a backup that runs until the free trial expires in March 2026. The Azure Container App demonstrates the Docker + CI/CD pipeline — it cold-starts on first request (scale-to-zero).

---

## Overview

Ank is a full-stack AI virtual mock interviewer built on **OpenRouter's free-tier open-source LLMs** (no credit card) plus local faster-whisper and free gTTS. It reads a job description and a resume, generates a tailored **sectioned interview plan**, then conducts the interview conversationally — asking one question at a time, probing with **follow-up questions** on your answers, and moving between sections with natural spoken transitions. When the interview wraps up, it produces a **scorecard** (0–100) with strengths, improvement areas and a verdict.

**Live interaction flow:**

```
Your voice → faster-whisper transcription → OpenRouter interviewer → gTTS → spoken reply (subtitle + lip-sync)
```

---

## Features

- **AI mock interviewer** — realistic, sectioned interviews built from the JD + your resume
- **Natural conversation** — warm greeting, section transitions that reference your answers, one probing follow-up per answer, closing with "any questions for me?"
- **Adjustable length** — pick **Quick (~8 min)**, **Standard (~18 min)** or **Deep (~30 min)** during setup; controls total questions and whether probing follow-ups are asked
- **Real-time transcription** — answers are streamed to `/ws/transcribe` and transcribed live word-by-word (Moonshine ONNX, tiny+quantized by default to fit small free-tier containers — set `MOONSHINE_MODEL=base MOONSHINE_PRECISION=float` for max quality; faster-whisper optional via `STT_BACKEND`), with a WAV-upload fallback
- **Final scorecard** — 0–100 score, strengths, areas to improve, and a spoken verdict
- **Animated interviewer avatar** — illustrated personas (Recruiter, Technical Lead, HR, Executive) with lip-sync, blinking and head motion while speaking; toggle to a live circular waveform
- **Live subtitles** — the interviewer's speech appears as subtitles in sync with each sentence
- **Resume & JD input** — paste text or upload PDF / DOCX / TXT (parsed server-side)
- **Voice answers** — click the mic or hold Spacebar to answer; your transcript is shown back
- **Skip / end early** — move past a question or finish the interview any time
- **Free stack** — OpenRouter free-tier open-source LLMs (with automatic model fallbacks) + local Moonshine ONNX / faster-whisper for transcription, gTTS for speech
- **Dual mode** — `openrouter` (cloud, free, default) or `local` (Ollama + faster-whisper, fully offline)
- **Mobile responsive** — works on phones and tablets
- **Auto cleanup** — temp audio/files removed on startup, shutdown and reset

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (UI)                      │
│  Paste/upload JD + resume → fetch(/interview/start) │
│  AudioWorklet → PCM frames → WS /ws/transcribe       │
│  (live partials) → /interview/answer-text on stop    │
│  ← JSON { utterance, segments, state }               │
│  Audio() playback ← /audio/{file} + lip-sync avatar  │
└────────────────────────┬─────────────────────────────┘
                         │ HTTP
┌────────────────────────▼─────────────────────────────┐
│                FastAPI Backend                        │
│                                                       │
│  /interview/*                                        │
│    ├── interview/extractor.py  (PDF/DOCX/TXT parse) │
│    ├── interview/plan.py       (sections + questions)│
│    ├── interview/engine.py     (follow-ups,          │
│    │                            transitions, closing)│
│    ├── interview/evaluator.py  (scorecard)           │
│    ├── audio/stt_local.py      (faster-whisper)      │
│    └── audio/tts.py            (gTTS → subtitle chunks)
│                                                       │
│  /chat            legacy general voice chat (server) │
│  /audio/{file}    serves generated MP3               │
│  /health /healthz /metrics                           │
└──────────────────────────────────────────────────────┘
```

**Component breakdown:**

| Component | File | Responsibility |
|---|---|---|
| Web UI | `static/index.html` | Setup, interview, results; avatar + waveform + subtitles |
| API Server | `app.py` | Routing, session management, cleanup |
| Interview Plan | `interview/plan.py` | Sections + questions from JD & resume |
| Interview Engine | `interview/engine.py` | Turn state machine: follow-ups, bridges, transitions, closing |
| Evaluator | `interview/evaluator.py` | Score + strengths / improvements / verdict |
| File Extractor | `interview/extractor.py` | PDF / DOCX / TXT text extraction |
| LLM OpenRouter | `brain/llm_openrouter.py` | OpenRouter free-tier chat + model fallbacks + retry |
| STT | `audio/stt_local.py` | faster-whisper transcription (local, all modes) |
| TTS | `audio/tts.py` | gTTS subtitle-synced sentence chunks |
| LLM Local | `brain/llm_local.py` | Ollama (offline fallback) |

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn | Async REST API |
| LLM | OpenRouter free-tier open-source models / Ollama | Auto-discovers $0 models with quality ordering + `openrouter/free` fallback, env-pinnable |
| Speech-to-Text | faster-whisper | Local (all modes) |
| Text-to-Speech | gTTS | Free, chunked per sentence for subtitle sync |
| Resume Parsing | pypdf, python-docx | PDF / DOCX / TXT |
| Frontend | Vanilla HTML / CSS / JS | Single file; SVG avatar + canvas waveform |
| Audio Capture | Web MediaRecorder API | Browser-native |
| Deployment | Render + UptimeRobot / Railway / Azure Container Apps | Auto-deploy from GitHub, always on via health check pings |

---

## Docker & CI/CD

The app ships as a containerized service with a full CI/CD pipeline:

- **`Dockerfile`** — multi-stage build (`python:3.13-slim`, non-root user), runs `uvicorn app:app`.
- **`docker-compose.yml`** — local development; app service plus an optional `ollama` service for fully-local inference mode.
- **`.github/workflows/ci.yml`** — on every push/PR: `ruff` lint → `pytest` → build & push the image to GitHub Container Registry → on `main`, deploy to **Azure Container Apps**.
- **Health & monitoring** — `/health` and `/healthz` liveness endpoints; `/metrics` exposes Prometheus metrics (request counter, latency histogram, process/GC stats).

### Local build

```bash
docker compose build
docker compose up            # http://localhost:8000
```

### Azure Container Apps — cost note

The Azure deployment runs on the **free tier** (180,000 vCPU-seconds / 360,000 GiB-seconds / 2M requests per month) with `min-replicas 0`, so it **scales to zero** when idle — it wakes on first request and costs nothing while unused. Secrets (`OPENROUTER_API_KEY`) are stored in Azure Container App secrets, never in the image or repo.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- OpenRouter API key — [openrouter.ai/keys](https://openrouter.ai/keys) *(free tier, no credit card)*
- [Ollama](https://ollama.com) installed *(local mode only)*
- [ffmpeg](https://ffmpeg.org/download.html) installed

### 1. Clone the repository

```bash
git clone https://github.com/MohammeddArsh/ank-interviewer
cd ank-interviewer
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure mode

`MODE` is read from the environment (defaults to `openrouter`):

```bash
MODE=openrouter  # "openrouter" = OpenRouter free-tier LLM + local faster-whisper (default)
                 # "local"      = Ollama + faster-whisper (fully offline, no API key needed)
```

### 5. Add your API key *(OpenRouter mode only)*

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your-openrouter-api-key
```

**Model auto-discovery:** free-tier OpenRouter models rotate constantly (models get retired to paid or replaced), so the app **discovers currently-free models at runtime** (`GET /models`, filtered to $0 pricing), orders them by quality, and tries each one in turn — advancing automatically past `404 "unavailable for free"` and rate-limit errors. `openrouter/free` is always appended as the ultimate fallback. To **pin** a specific list and skip discovery:

```
# OPENROUTER_MODELS=["qwen/qwen3-next-80b-a3b-instruct:free","openrouter/free"]
```

Free-tier rate limits are ~20 requests/minute and ~50 requests/day (raised to 1,000/day after a one-time $10 credit purchase).

> **Troubleshooting:** if *every* free model returns a 404 like `No endpoints available matching your data policy`, it's your OpenRouter **Privacy settings**, not the app — disable **"Zero data retention endpoints only"** under [Settings → Privacy](https://openrouter.ai/settings/privacy) so free endpoints can serve requests.

The model that generated the current turn is shown as a **"brain: \<model\>" chip** in the interview screen and returned as `model` in each `/interview/*` response.

### 6. Configure ffmpeg *(local playback only)*

Open `audio/tts.py` and update the path:

```python
FFMPEG_DIR = r"C:\path\to\ffmpeg\bin"   # Windows
# FFMPEG_DIR = "/usr/local/bin"         # Mac / Linux
```

### 7. Pull the Llama model *(local mode only)*

```bash
ollama pull llama3.2
```

### 8. Run the server

```bash
uvicorn app:app --reload
```

### 9. Open in browser

```
http://localhost:8000
```

---

## Project Structure

```
ank-interviewer/
├── app.py                  # FastAPI server, endpoint wiring, temp file cleanup
├── config.py               # Mode switch, API keys, interview tuning
├── requirements.txt
├── render.yaml             # Render deployment config
├── static/
│   └── index.html          # Full web UI — single file (avatar + waveform)
├── interview/              # Mock-interview engine
│   ├── routes.py           # /interview/* endpoints
│   ├── engine.py           # Session state machine (follow-ups, transitions, closing)
│   ├── plan.py             # Section + question generation
│   ├── evaluator.py        # Scorecard generation
│   ├── extractor.py        # PDF / DOCX / TXT text extraction
│   └── prompts.py          # LLM prompt builders
├── audio/
│   ├── stt.py              # STT router (always faster-whisper)
│   ├── stt_local.py        # faster-whisper (local)
│   └── tts.py              # gTTS (incl. subtitle-synced chunks)
└── brain/
    ├── llm.py              # LLM router
    ├── llm_openrouter.py   # OpenRouter chat + model fallbacks + retry
    ├── llm_local.py        # Ollama
    ├── memory.py           # Conversation history (legacy /chat)
    └── logger.py           # Session logging, JSON/CSV export
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/interview/start` | Builds the interview and returns the opener; body: `job_description`, `resume_text`, `interviewer` (JSON), `duration` (`quick`/`standard`/`deep`) |
| `POST` | `/interview/prepare` | Builds the plan without speaking → `{plan, model}` (sections preview for the setup wizard) |
| `POST` | `/interview/begin` | Delivers the greeting + first question (after `prepare`) |
| `POST` | `/interview/answer` | Multipart audio answer → next interviewer turn (follow-up / transition / closing) |
| `POST` | `/interview/answer-text` | JSON `{transcript}` fast path — submits an already-transcribed answer, skipping server STT |
| `WS` | `/ws/transcribe` | Real-time streaming transcription: browser sends raw Int16 LE 16 kHz mono PCM frames + `{"type":"stop"}`; server replies with `start` / `partial` / `final` JSON events |
| `POST` | `/interview/upload` | Upload PDF/DOCX/TXT → extracted text |
| `POST` | `/interview/skip` | Skip the current question |
| `POST` | `/interview/end` | End early → returns immediately with `pending: true`; evaluation runs in the background |
| `GET` | `/interview/results` | Poll for the finished evaluation → `{ready, evaluation, evaluation_segments}` |
| `GET` | `/interview/state` | Current interview progress |
| `POST` | `/interview/reset` | Clear the active interview |
| `POST` | `/chat` | Legacy voice chat endpoint (server-only, kept for compatibility) |
| `GET` | `/audio/{file}` | Serves generated TTS audio |
| `GET` | `/health`, `/healthz` | Liveness probes (200 OK) |
| `GET` | `/metrics` | Prometheus metrics (counter, latency histogram, process/GC) |

---

## Privacy & Ethics

- A consent banner is shown at the start of every session explaining data usage clearly
- Voice audio and your interview answers are transcribed locally with faster-whisper; the job description, resume and transcript are sent to the OpenRouter API (free-tier open-source models) to run the interview
- No data is stored permanently on the server
- The local mode option (`MODE=local`, Ollama + faster-whisper) allows fully offline operation with zero data leaving the device
- Users can decline consent — no data is collected if declined

---

## Built by

**Mohammed Arsh** — [github.com/MohammeddArsh](https://github.com/MohammeddArsh)
