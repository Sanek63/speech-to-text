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

import redis
import torch

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

    print(f"[worker] [{file_id}] препроцессинг аудио...", flush=True)
    db.sync_set_status(conn, file_id, status="processing", stage="preprocess", progress=None)
    t0 = time.time()

    def _report_preprocess_progress(pct: float) -> None:
        # прогресс -- не более чем удобство для UI, не должен уронить обработку задачи
        try:
            db.sync_set_status(conn, file_id, stage="preprocess", progress=round(pct, 1))
        except Exception as e:
            print(f"[worker] [{file_id}] не удалось записать прогресс препроцессинга: {e}", flush=True)

    wav_path = pipeline.preprocess_audio(original, d, on_progress=_report_preprocess_progress)
    duration = pipeline.get_audio_duration(wav_path)
    preprocess_time = time.time() - t0
    print(f"[worker] [{file_id}] длительность аудио {duration:.0f}с, препроцессинг за {preprocess_time:.1f}с", flush=True)

    print(f"[worker] [{file_id}] ASR (Whisper {WHISPER_MODEL})...", flush=True)
    db.sync_set_status(conn, file_id, status="processing", stage="asr", progress=None)
    t0 = time.time()
    if duration > CHUNK_THRESHOLD_SEC:
        chunks = pipeline.split_audio_on_silence(wav_path, d, CHUNK_TARGET_SEC, CHUNK_MAX_SEC)
        result = _run_asr_chunked(model, device, chunks)
    else:
        result = _run_asr(model, device, wav_path)
    asr_time = time.time() - t0
    print(f"[worker] [{file_id}] ASR готов за {asr_time:.1f}с", flush=True)

    print(f"[worker] [{file_id}] диаризация (NeMo MSDD)...", flush=True)
    db.sync_set_status(conn, file_id, status="processing", stage="diarization")
    t0 = time.time()
    diarization = pipeline.diarize_nemo(wav_path, d, device)
    diarization_time = time.time() - t0
    print(f"[worker] [{file_id}] диаризация готова за {diarization_time:.1f}с", flush=True)

    print(f"[worker] [{file_id}] роли и экспорт...", flush=True)
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
    print(f"[worker] [{file_id}] готово за {total_time:.1f}с, total_rtf={metrics['total_rtf']}", flush=True)
    db.sync_set_status(conn, file_id, status="done", stage=None, metrics=metrics, transcript=transcript_dict)


def _log_gpu_memory() -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    print(f"[worker] GPU память: {free_gb:.2f}/{total_gb:.2f} ГБ свободно", flush=True)
    if free_gb < 8:
        print(
            f"[worker] ВНИМАНИЕ: свободно всего {free_gb:.2f}ГБ, а Whisper large-v3 обычно "
            "требует ~14ГБ. Похоже, GPU-память ещё занята другим процессом (например, не до "
            "конца остановленным старым контейнером воркера после рестарта). Проверьте на "
            "хосте `nvidia-smi` (PID держащих память) и `docker compose ps` (нет ли лишних "
            "контейнеров) до того, как загрузка модели упадёт в OOM.",
            flush=True,
        )


def main() -> None:
    device = pipeline.detect_device()
    print(f"[worker] device: {device}", flush=True)
    if device == "cuda":
        _log_gpu_memory()

    db.sync_init_schema()
    conn = db.sync_connect()

    print("[worker] loading Whisper...", flush=True)
    model = pipeline.load_whisper_model(WHISPER_MODEL, "cpu")

    print("[worker] warming up NeMo (форсируем скачивание весов сразу)...", flush=True)
    warmup_dir = DATA_DIR / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    silence_wav = pipeline.generate_silence_wav(warmup_dir / "silence.wav")
    try:
        pipeline.diarize_nemo(silence_wav, warmup_dir, device)
    except ValueError as e:
        # NeMo сам детектирует VAD-ом полное отсутствие речи в тишине и осознанно
        # прерывает .diarize() — но веса к этому моменту уже скачаны и загружены
        # (это происходит раньше, при конструировании NeuralDiarizer), так что для
        # цели warm-up это ожидаемо и не ошибка.
        if "silence" not in str(e).lower():
            raise
        print(f"[worker] warm-up diarize пропущен (тишина, ожидаемо): {e}", flush=True)

    redis_client = jobqueue.sync_client()

    # Восстанавливаем задачи, застрявшие в processing-списке Redis с прошлого (упавшего или
    # убитого) запуска воркера — раньше, чем reconcile пометит их ошибкой в Postgres, чтобы
    # успевшие восстановиться не помечались error и пересчитались с нуля вместо этого.
    recovered = jobqueue.sync_recover_stuck_jobs(redis_client)
    if recovered:
        print(f"[worker] после рестарта нашёл {len(recovered)} незавершённых задач(и), верну в работу: {recovered}", flush=True)
    db.sync_reconcile_interrupted(conn, "воркер перезапустился во время обработки этой задачи", keep_file_ids=recovered)

    print("[worker] ready", flush=True)

    while True:
        try:
            file_id = jobqueue.sync_pop_job(redis_client)
        except redis.exceptions.RedisError as e:
            print(f"[worker] Redis временно недоступен ({e}), пробую снова через 2с", flush=True)
            time.sleep(2)
            continue

        if file_id is None:
            continue  # обычный таймаут BLMOVE — новых задач нет, просто снова ждём

        print(f"[worker] взял задачу {file_id}", flush=True)
        try:
            process_job(conn, model, device, file_id)
        except Exception as e:
            print(f"[worker] задача {file_id} упала: {e}", flush=True)
            db.sync_set_status(
                conn, file_id, status="error", stage=None,
                error=f"{e}\n{traceback.format_exc()[-2000:]}",
            )
        finally:
            jobqueue.sync_ack_job(redis_client, file_id)


if __name__ == "__main__":
    main()
