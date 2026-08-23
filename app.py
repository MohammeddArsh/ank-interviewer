import glob
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

sys.path.insert(0, os.path.dirname(__file__))

from audio.stream_stt import router as stream_router
from audio.stt import transcribe
from audio.tts import speak_to_file
from brain.llm import get_reply
from brain.logger import SessionLogger
from brain.memory import ConversationMemory
from config import TEMP_DIR
from interview.routes import router as interview_router

CHAT_REQUESTS = Counter("ank_chat_requests_total", "Total /chat requests")
CHAT_LATENCY = Histogram(
    "ank_chat_latency_seconds",
    "Chat request latency in seconds",
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)


def cleanup_temp_files():
    patterns = ["tts_*.mp3", "upload_*.wav", "rec_*.wav"]
    removed = 0
    for pattern in patterns:
        for f in glob.glob(os.path.join(TEMP_DIR, pattern)):
            try:
                os.remove(f)
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"Cleaned up {removed} leftover temp files.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_temp_files()
    yield
    cleanup_temp_files()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview_router)
app.include_router(stream_router)

memory = ConversationMemory()
logger = SessionLogger()


@app.get("/")
def index():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache"})


@app.post("/chat")
async def chat(audio: UploadFile = File(...)):  # noqa: B008 - FastAPI dependency injection
    audio_path = os.path.join(TEMP_DIR, "upload_" + uuid.uuid4().hex + ".wav")
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    user_text = transcribe(audio_path)
    try:
        os.remove(audio_path)
    except Exception:
        pass

    if not user_text:
        return JSONResponse({"error": "Could not transcribe audio"}, status_code=400)

    memory.add_user_message(user_text)
    memory.trim_if_needed()

    CHAT_REQUESTS.inc()
    start_time = time.time()
    reply, token_usage = get_reply(memory.get_messages())
    CHAT_LATENCY.observe(time.time() - start_time)
    response_time_ms = (time.time() - start_time) * 1000

    memory.add_assistant_message(reply)
    logger.log_turn(user_text, reply, response_time_ms, token_usage)

    audio_out = speak_to_file(reply)

    return JSONResponse({
        "user_text": user_text,
        "reply": reply,
        "audio_url": "/audio/" + os.path.basename(audio_out),
        "analytics": logger.get_analytics()
    })


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(TEMP_DIR, filename)
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/reset")
def reset():
    logger.reset()
    memory.__init__()
    cleanup_temp_files()
    return {"status": "reset"}


@app.get("/analytics")
def analytics():
    return logger.get_analytics()


@app.get("/export/json")
def export_json():
    data = logger.export_json()
    filename = f"ank_session_{logger.session_id}.json"
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/export/csv")
def export_csv():
    data = logger.export_csv()
    filename = f"ank_session_{logger.session_id}.csv"
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.head("/health")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Mount static LAST so it doesn't intercept API routes
app.mount("/static", StaticFiles(directory="static"), name="static")

