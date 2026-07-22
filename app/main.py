import asyncio
import json
import multiprocessing
import os
import queue
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
WATCHDOG_INTERVAL_SEC = 3
WORKER_READY_POLL_SEC = 5

MP_CTX = multiprocessing.get_context("spawn")  # CUDA-контексты не fork-safe


def job_dir(file_id: str) -> Path:
    return DATA_DIR / file_id


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_status(file_id: str, **kwargs) -> dict:
    path = job_dir(file_id) / "status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.setdefault("file_id", file_id)
    current.setdefault("created_at", now_iso())
    current.update(kwargs)
    current["updated_at"] = now_iso()
    path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
    return current


def list_jobs() -> list[dict]:
    jobs = []
    for status_path in DATA_DIR.glob("*/status.json"):
        try:
            jobs.append(json.loads(status_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


def reconcile_interrupted_jobs(reason: str) -> None:
    """Джобы, застрявшие в queued/processing (сервер/воркер умер посреди работы) — иначе
    UI будет вечно показывать 'в процессе', хотя их больше никто не считает."""
    for job in list_jobs():
        if job.get("status") in ("queued", "processing"):
            set_status(job["file_id"], status="error", stage=None, error=reason)


def load_status(file_id: str) -> dict | None:
    path = job_dir(file_id) / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- Воркер-процесс: инференс, никогда не касается FastAPI ------------------
#
# NeMo/Whisper — нативный CUDA-код. Жёсткий крах там (segfault, зависший драйвер,
# OOM-kill от cgroup) try/except в Python не ловит, потому что это гибель процесса,
# а не исключение. Поэтому весь инференс живёт в отдельном OS-процессе: крах там
# роняет только его, API и UI продолжают отвечать (см. watchdog ниже).


def _run_asr(model, device: str, wav_path: Path) -> dict:
    pipeline.move_model(model, device)
    try:
        return pipeline.transcribe_with_model(model, wav_path, LANGUAGE)
    finally:
        pipeline.move_model(model, "cpu")


def _run_asr_chunked(model, device: str, chunks) -> dict:
    pipeline.move_model(model, device)
    try:
        return pipeline.transcribe_chunked(model, chunks, LANGUAGE)
    finally:
        pipeline.move_model(model, "cpu")


def process_job(model, device: str, file_id: str) -> None:
    """Синхронная версия — выполняется целиком внутри воркер-процесса."""
    d = job_dir(file_id)
    original = next(d.glob("original.*"))

    set_status(file_id, status="processing", stage="preprocess")
    t0 = time.time()
    wav_path = pipeline.preprocess_audio(original, d)
    duration = pipeline.get_audio_duration(wav_path)
    preprocess_time = time.time() - t0

    set_status(file_id, status="processing", stage="asr")
    t0 = time.time()
    if duration > CHUNK_THRESHOLD_SEC:
        chunks = pipeline.split_audio_on_silence(wav_path, d, CHUNK_TARGET_SEC, CHUNK_MAX_SEC)
        result = _run_asr_chunked(model, device, chunks)
    else:
        result = _run_asr(model, device, wav_path)
    asr_time = time.time() - t0

    set_status(file_id, status="processing", stage="diarization")
    t0 = time.time()
    diarization = pipeline.diarize_nemo(wav_path, d, device)
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


def worker_main(job_queue: multiprocessing.Queue, ready_queue: multiprocessing.Queue, device: str) -> None:
    """Точка входа дочернего процесса: грузит модели один раз, держит резидентно, обрабатывает
    задачи по одной. Ошибки внутри одной задачи ловятся и не убивают цикл — а вот жёсткий крах
    нативного кода (то, что try/except не поймает) убьёт только этот процесс, см. watchdog."""
    print(f"[worker] device: {device}", flush=True)
    print("[worker] loading Whisper...", flush=True)
    model = pipeline.load_whisper_model(WHISPER_MODEL, "cpu")

    print("[worker] warming up NeMo (форсируем скачивание весов сразу)...", flush=True)
    warmup_dir = DATA_DIR / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    silence_wav = pipeline.generate_silence_wav(warmup_dir / "silence.wav")
    pipeline.diarize_nemo(silence_wav, warmup_dir, device)

    print("[worker] ready", flush=True)
    ready_queue.put("ready")

    while True:
        file_id = job_queue.get()
        if file_id is None:  # сигнал остановки от API-процесса
            break
        try:
            process_job(model, device, file_id)
        except Exception as e:
            set_status(file_id, status="error", stage=None, error=f"{e}\n{traceback.format_exc()[-2000:]}")


# --- API-процесс: только оркестрация, CUDA/Whisper/NeMo не касается ---------


def spawn_worker(app: FastAPI) -> None:
    """Поднимает воркер-процесс и блокирует (в вызывающем to_thread) до его готовности —
    с таймаутом на каждый опрос, чтобы не зависнуть навечно, если процесс упал при старте.

    daemon=False намеренно: NeMo сам порождает дочерние процессы (torch DataLoader с
    num_workers>0 для VAD), а Python запрещает daemon-процессам иметь своих детей
    (AssertionError: daemonic processes are not allowed to have children). Явный
    shutdown (job_queue.put(None) → join → terminate) уже есть в lifespan ниже и не
    зависит от флага daemon — при падении контейнера процесс всё равно убьёт вместе
    со всей cgroup, а не оставит висеть на хосте."""
    ready_queue = MP_CTX.Queue()
    process = MP_CTX.Process(
        target=worker_main, args=(app.state.job_queue, ready_queue, app.state.device), daemon=False,
    )
    process.start()
    while True:
        if not process.is_alive():
            raise RuntimeError("воркер-процесс упал во время старта (загрузка моделей) — см. логи контейнера")
        try:
            ready_queue.get(timeout=WORKER_READY_POLL_SEC)
            break
        except queue.Empty:
            continue
    app.state.worker_process = process
    app.state.worker_started_at = now_iso()


async def watchdog(app: FastAPI) -> None:
    """Если воркер упал жёстко (то, что try/except в worker_main поймать не может) —
    помечает зависшую в processing задачу ошибкой и поднимает воркера заново, чтобы очередь
    не встала намертво, а API/UI не пострадали от краша в ASR/диаризации."""
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_SEC)
        if app.state.worker_process.is_alive():
            continue

        print("[watchdog] воркер-процесс упал, перезапускаю...", flush=True)
        reconcile_interrupted_jobs("воркер-процесс упал (краш ASR/диаризации)")

        app.state.ready = False
        app.state.worker_restart_count += 1
        await asyncio.to_thread(spawn_worker, app)
        app.state.ready = True
        print("[watchdog] воркер перезапущен", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.state.ready = False
    app.state.device = pipeline.detect_device()
    app.state.job_queue = MP_CTX.Queue()
    app.state.worker_restart_count = 0
    print(f"[startup] device: {app.state.device}")

    reconcile_interrupted_jobs("сервер перезапустился во время обработки этой задачи")
    await asyncio.to_thread(spawn_worker, app)
    app.state.ready = True
    print("[startup] ready")

    watchdog_task = asyncio.create_task(watchdog(app))
    yield

    watchdog_task.cancel()
    app.state.job_queue.put(None)
    app.state.worker_process.join(timeout=5)
    if app.state.worker_process.is_alive():
        app.state.worker_process.terminate()


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"ready": getattr(app.state, "ready", False)}


# /api/worker и /api/jobs намеренно не публичные HTTP-роуты: в открытом для всех сервисе
# это была бы утечка чужих file_id/статусов/ошибок кому угодно. list_jobs() продолжает
# использоваться внутри — для reconcile_interrupted_jobs() при старте/после краша воркера.


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
    app.state.job_queue.put(file_id)
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
