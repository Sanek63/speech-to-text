#!/usr/bin/env python3
"""
Воркер: держит Whisper/NeMo резидентно в памяти, забирает file_id из Redis-очереди по
одной штуке, обрабатывает, пишет статус и результат в Postgres. Отдельный от API контейнер
— падение здесь (нативный CUDA-краш, зависший драйвер) не задевает API/UI. Перезапуск —
через restart policy самого Docker (`restart: unless-stopped` в docker-compose.yml), не
кастомный supervisor: раз общение с API идёт только через Redis/Postgres, а не in-memory
объекты, Docker сам делает то, что раньше делал наш watchdog.
"""
import os
import time
import traceback
from pathlib import Path

import db
import jobqueue
import pipeline

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
LANGUAGE = os.environ.get("LANGUAGE", "ru")
CHUNK_THRESHOLD_SEC = 45 * 60  # длиннее — включаем чанкование ASR автоматически
CHUNK_TARGET_SEC = 600
CHUNK_MAX_SEC = 720


def job_dir(file_id: str) -> Path:
    return DATA_DIR / file_id


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


def process_job(conn, model, device: str, file_id: str) -> None:
    d = job_dir(file_id)
    original = next(d.glob("original.*"))

    db.sync_set_status(conn, file_id, status="processing", stage="preprocess")
    t0 = time.time()
    wav_path = pipeline.preprocess_audio(original, d)
    duration = pipeline.get_audio_duration(wav_path)
    preprocess_time = time.time() - t0

    db.sync_set_status(conn, file_id, status="processing", stage="asr")
    t0 = time.time()
    if duration > CHUNK_THRESHOLD_SEC:
        chunks = pipeline.split_audio_on_silence(wav_path, d, CHUNK_TARGET_SEC, CHUNK_MAX_SEC)
        result = _run_asr_chunked(model, device, chunks)
    else:
        result = _run_asr(model, device, wav_path)
    asr_time = time.time() - t0

    db.sync_set_status(conn, file_id, status="processing", stage="diarization")
    t0 = time.time()
    diarization = pipeline.diarize_nemo(wav_path, d, device)
    diarization_time = time.time() - t0

    db.sync_set_status(conn, file_id, status="processing", stage="postprocess")
    t0 = time.time()
    result = pipeline.assign_word_speakers(result, diarization)
    turns = pipeline.group_into_turns(result)
    features = pipeline.extract_role_features(turns)
    role_map = pipeline.assign_roles(features)
    pipeline.apply_roles(turns, role_map)
    transcript_dict = pipeline.turns_to_dict(turns, audio_file=original.name, language=LANGUAGE)
    # текстовые форматы — на диск, для тех, кто хочет скачать файлом, а не только через API
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
    db.sync_set_status(conn, file_id, status="done", stage=None, metrics=metrics, transcript=transcript_dict)


def main() -> None:
    device = pipeline.detect_device()
    print(f"[worker] device: {device}", flush=True)

    db.sync_init_schema()
    conn = db.sync_connect()
    db.sync_reconcile_interrupted(conn, "воркер перезапустился во время обработки этой задачи")

    print("[worker] loading Whisper...", flush=True)
    model = pipeline.load_whisper_model(WHISPER_MODEL, "cpu")

    print("[worker] warming up NeMo (форсируем скачивание весов сразу)...", flush=True)
    warmup_dir = DATA_DIR / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    silence_wav = pipeline.generate_silence_wav(warmup_dir / "silence.wav")
    pipeline.diarize_nemo(silence_wav, warmup_dir, device)

    print("[worker] ready", flush=True)
    redis_client = jobqueue.sync_client()

    while True:
        file_id = jobqueue.sync_pop_job(redis_client)
        if file_id is None:
            continue  # обычный таймаут BLPOP — новых задач нет, просто снова ждём
        try:
            process_job(conn, model, device, file_id)
        except Exception as e:
            db.sync_set_status(
                conn, file_id, status="error", stage=None,
                error=f"{e}\n{traceback.format_exc()[-2000:]}",
            )


if __name__ == "__main__":
    main()
