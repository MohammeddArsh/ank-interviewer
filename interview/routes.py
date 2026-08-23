"""HTTP routes for the AI mock interviewer."""

import os
import uuid

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from audio.stt import transcribe
from audio.tts import speak_to_chunks
from brain.llm import get_last_model
from config import DEFAULT_DURATION, MAX_UPLOAD_BYTES, TEMP_DIR
from interview import extractor
from interview.engine import InterviewSession

router = APIRouter(prefix="/interview")

_session: InterviewSession = None


def _reset_session():
    global _session
    _session = None


def _require_session():
    if _session is None:
        raise ValueError("No active interview. Start one first.")
    return _session


def _utterance_response(utterance: str, done: bool = False, evaluation: dict = None, transcript: str = None) -> dict:
    resp = {
        "utterance": utterance,
        "segments": speak_to_chunks(utterance),
        "state": _session.build_state(),
        "model": get_last_model(),
    }
    if transcript is not None:
        resp["transcript"] = transcript
    if done:
        resp["done"] = True
        resp["evaluation"] = evaluation or _session.evaluation
        resp["evaluation_segments"] = _evaluation_segments(resp["evaluation"])
    return resp


def _evaluation_segments(evaluation: dict) -> list:
    """Build spoken feedback chunks from an evaluation result."""
    parts = [evaluation.get("verdict") or ""]
    strengths = evaluation.get("strengths") or []
    improvements = evaluation.get("improvements") or []
    if strengths:
        parts.append("Your strengths: " + "; ".join(str(s) for s in strengths))
    if improvements:
        parts.append("Areas to work on: " + "; ".join(str(i) for i in improvements))
    text = " ".join(p for p in parts if p.strip())
    return speak_to_chunks(text)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):  # noqa: B008
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File too large (max 5 MB)."}, status_code=413)

    tmp_path = os.path.join(TEMP_DIR, "upload_" + uuid.uuid4().hex)
    with open(tmp_path, "wb") as f:
        f.write(contents)
    try:
        text = extractor.extract_text(tmp_path, file.filename)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not text:
        return JSONResponse({"error": "Could not extract any text from the file."}, status_code=400)
    return {"text": text, "filename": file.filename}


@router.post("/start")
async def start(job_description: str = Form(""), resume_text: str = Form(""),
                interviewer: str = Form("{}"),
                duration: str = Form(DEFAULT_DURATION)):  # noqa: B008
    global _session
    created = _build_session(job_description, resume_text, interviewer, duration)
    if isinstance(created, JSONResponse):
        return created
    _session = created
    try:
        result = _session.start()
    except Exception as e:
        _reset_session()
        return JSONResponse({"error": f"Could not build the interview: {e}"}, status_code=500)

    return _utterance_response(result["utterance"])


@router.post("/prepare")
async def prepare(job_description: str = Form(""), resume_text: str = Form(""),
                  interviewer: str = Form("{}"),
                  duration: str = Form(DEFAULT_DURATION)):  # noqa: B008
    global _session
    created = _build_session(job_description, resume_text, interviewer, duration)
    if isinstance(created, JSONResponse):
        return created
    _session = created
    try:
        result = _session.prepare()
    except Exception as e:
        _reset_session()
        return JSONResponse({"error": f"Could not build the interview: {e}"}, status_code=500)

    return {"plan": result, "model": get_last_model()}


@router.post("/begin")
async def begin():  # noqa: B008
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    try:
        result = session.begin()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return _utterance_response(result["utterance"])


def _build_session(job_description: str, resume_text: str, interviewer: str,
                   duration: str = DEFAULT_DURATION):
    """Validate inputs and construct a session, or return an error JSONResponse."""
    import json

    job_description = job_description.strip()
    resume_text = resume_text.strip()
    if not job_description or not resume_text:
        return JSONResponse({"error": "Both the job description and resume are required."}, status_code=400)

    try:
        interviewer_data = json.loads(interviewer or "{}")
    except json.JSONDecodeError:
        interviewer_data = {}
    return InterviewSession(job_description, resume_text, interviewer_data or None, duration)


def _audio_extension(content_type: str) -> str:
    """Map a browser MediaRecorder content type to a file extension Gemini can sniff."""
    ct = (content_type or "").lower()
    if "webm" in ct or "opus" in ct:
        return ".webm"
    if "ogg" in ct:
        return ".ogg"
    if "mp4" in ct or "m4a" in ct or "aac" in ct:
        return ".m4a"
    if "flac" in ct:
        return ".flac"
    return ".wav"


@router.post("/answer")
async def answer(audio: UploadFile = File(...)):  # noqa: B008
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    audio_path = os.path.join(TEMP_DIR, "upload_" + uuid.uuid4().hex + _audio_extension(audio.content_type))
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    try:
        user_text = transcribe(audio_path)
    except Exception:
        user_text = ""
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    if not user_text:
        return JSONResponse({"error": "Could not hear your answer. Please try again."}, status_code=400)

    result = session.handle_answer(user_text)
    return _utterance_response(
        result["utterance"],
        done=result.get("done", False),
        evaluation=result.get("evaluation"),
        transcript=user_text,
    )


class TranscriptIn(BaseModel):
    transcript: str


@router.post("/answer-text")
async def answer_text(payload: TranscriptIn):
    """Fast path: the client already holds a streamed transcript, so skip STT."""
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    user_text = payload.transcript.strip()
    if not user_text:
        return JSONResponse({"error": "Could not hear your answer. Please try again."}, status_code=400)

    result = session.handle_answer(user_text)
    return _utterance_response(
        result["utterance"],
        done=result.get("done", False),
        evaluation=result.get("evaluation"),
        transcript=user_text,
    )


@router.post("/skip")
async def skip():  # noqa: B008
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    result = session.handle_answer("", is_skip=True)
    return _utterance_response(
        result["utterance"],
        done=result.get("done", False),
        evaluation=result.get("evaluation"),
    )


@router.post("/end")
async def end():  # noqa: B008
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    result = session.end_early()
    return {"done": True, "pending": True, "state": result["state"]}


@router.get("/results")
async def results():  # noqa: B008
    try:
        session = _require_session()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if session.evaluation is None:
        return {"ready": False}
    return {
        "ready": True,
        "evaluation": session.evaluation,
        "evaluation_segments": _evaluation_segments(session.evaluation),
    }


@router.get("/state")
async def state():  # noqa: B008
    try:
        session = _require_session()
    except ValueError:
        return {"session": None, "model": get_last_model()}
    return {"session": session.build_state(), "model": get_last_model()}


@router.post("/reset")
async def reset():  # noqa: B008
    _reset_session()
    return {"status": "reset"}
