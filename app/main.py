import asyncio
import json
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
LANGUAGE = os.environ.get("LANGUAGE", "ru")
CHUNK_THRESHOLD_SEC = 45 * 60  # длиннее — включаем чанкование ASR автоматически
CHUNK_TARGET_SEC = 600
CHUNK_MAX_SEC = 720


def job_dir(file_id: str) -> Path:
    return DATA_DIR / file_id


def set_status(file_id: str, **kwargs) -> dict:
    path = job_dir(file_id) / "status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(kwargs)
    path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
    return current


def load_status(file_id: str) -> dict | None:
    path = job_dir(file_id) / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def worker(app: FastAPI) -> None:
    while True:
        file_id = await app.state.queue.get()
        try:
            await process_job(app, file_id)
        except Exception as e:
            set_status(file_id, status="error", stage=None, error=f"{e}\n{traceback.format_exc()[-2000:]}")
        finally:
            app.state.queue.task_done()


def _run_asr(app: FastAPI, wav_path: Path) -> dict:
    pipeline.move_model(app.state.whisper_model, app.state.device)
    try:
        return pipeline.transcribe_with_model(app.state.whisper_model, wav_path, LANGUAGE)
    finally:
        pipeline.move_model(app.state.whisper_model, "cpu")


def _run_asr_chunked(app: FastAPI, chunks) -> dict:
    pipeline.move_model(app.state.whisper_model, app.state.device)
    try:
        return pipeline.transcribe_chunked(app.state.whisper_model, chunks, LANGUAGE)
    finally:
        pipeline.move_model(app.state.whisper_model, "cpu")


async def process_job(app: FastAPI, file_id: str) -> None:
    d = job_dir(file_id)
    original = next(d.glob("original.*"))

    set_status(file_id, status="processing", stage="preprocess")
    t0 = time.time()
    wav_path = await asyncio.to_thread(pipeline.preprocess_audio, original, d)
    duration = await asyncio.to_thread(pipeline.get_audio_duration, wav_path)
    preprocess_time = time.time() - t0

    set_status(file_id, status="processing", stage="asr")
    t0 = time.time()
    if duration > CHUNK_THRESHOLD_SEC:
        chunks = await asyncio.to_thread(
            pipeline.split_audio_on_silence, wav_path, d, CHUNK_TARGET_SEC, CHUNK_MAX_SEC
        )
        result = await asyncio.to_thread(_run_asr_chunked, app, chunks)
    else:
        result = await asyncio.to_thread(_run_asr, app, wav_path)
    asr_time = time.time() - t0

    set_status(file_id, status="processing", stage="diarization")
    t0 = time.time()
    diarization = await asyncio.to_thread(pipeline.diarize_nemo, wav_path, d, app.state.device)
    diarization_time = time.time() - t0

    set_status(file_id, status="processing", stage="postprocess")
    t0 = time.time()
    result = pipeline.assign_word_speakers(result, diarization)
    turns = pipeline.group_into_turns(result)
    features = pipeline.extract_role_features(turns)
    role_map = pipeline.assign_roles(features)
    pipeline.apply_roles(turns, role_map)
    pipeline.export_json(turns, d / "transcript.json", audio_file=original.name, language=LANGUAGE)
    pipeline.export_text(turns, d / "transcript.txt")
    pipeline.export_srt(turns, d / "transcript.srt")
    postprocess_time = time.time() - t0

    total_time = preprocess_time + asr_time + diarization_time + postprocess_time

    def rtf(t: float) -> float:
        return round(t / duration, 4) if duration else 0.0

    metrics = {
        "audio_duration_sec": round(duration, 2),
        "preprocess_time_sec": round(preprocess_time, 2), "preprocess_rtf": rtf(preprocess_time),
        "asr_time_sec": round(asr_time, 2), "asr_rtf": rtf(asr_time),
        "diarization_time_sec": round(diarization_time, 2), "diarization_rtf": rtf(diarization_time),
        "postprocess_time_sec": round(postprocess_time, 2), "postprocess_rtf": rtf(postprocess_time),
        "total_time_sec": round(total_time, 2), "total_rtf": rtf(total_time),
    }
    set_status(file_id, status="done", stage=None, metrics=metrics)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.state.ready = False
    app.state.device = pipeline.detect_device()
    app.state.queue = asyncio.Queue()
    print(f"[startup] device: {app.state.device}")

    print("[startup] loading Whisper...")
    app.state.whisper_model = await asyncio.to_thread(pipeline.load_whisper_model, WHISPER_MODEL, "cpu")

    print("[startup] warming up NeMo (форсируем скачивание весов сразу, не на первом запросе)...")
    warmup_dir = DATA_DIR / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    silence_wav = await asyncio.to_thread(pipeline.generate_silence_wav, warmup_dir / "silence.wav")
    await asyncio.to_thread(pipeline.diarize_nemo, silence_wav, warmup_dir, app.state.device)

    app.state.ready = True
    print("[startup] ready")

    worker_task = asyncio.create_task(worker(app))
    yield
    worker_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"ready": getattr(app.state, "ready", False)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    d = job_dir(file_id)
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio").suffix or ".bin"
    dest = d / f"original{ext}"
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    set_status(file_id, status="uploaded")
    return {"file_id": file_id}


@app.post("/api/transcribe/{file_id}")
async def transcribe(file_id: str):
    if not job_dir(file_id).exists():
        raise HTTPException(404, "file_id не найден")
    if not app.state.ready:
        raise HTTPException(503, "модели ещё прогреваются, попробуйте через немного")
    set_status(file_id, status="queued", stage=None, error=None)
    await app.state.queue.put(file_id)
    return {"status": "queued"}


@app.get("/api/status/{file_id}")
async def status(file_id: str):
    st = load_status(file_id)
    if st is None:
        raise HTTPException(404, "file_id не найден")
    return st


@app.get("/api/transcript/{file_id}")
async def transcript(file_id: str):
    path = job_dir(file_id) / "transcript.json"
    if not path.exists():
        raise HTTPException(404, "транскрипт ещё не готов")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/audio/{file_id}")
async def audio(file_id: str):
    d = job_dir(file_id)
    candidates = list(d.glob("original.*")) if d.exists() else []
    if not candidates:
        raise HTTPException(404, "аудио не найдено")
    return FileResponse(candidates[0])


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
