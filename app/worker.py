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
from datetime import datetime
from pathlib import Path

import redis
import torch

import db
import jobqueue
import pipeline


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
LANGUAGE = os.environ.get("LANGUAGE", "ru")
CHUNK_THRESHOLD_SEC = 45 * 60  # длиннее — включаем чанкование ASR автоматически
CHUNK_TARGET_SEC = 600
CHUNK_MAX_SEC = 720


def job_dir(file_id: str) -> Path:
    return DATA_DIR / file_id


def _run_asr(model, device: str, wav_path: Path, on_progress=None) -> dict:
    pipeline.move_model(model, device)
    try:
        return pipeline.transcribe_with_model(model, wav_path, LANGUAGE, on_progress=on_progress)
    finally:
        pipeline.move_model(model, "cpu")


def _run_asr_chunked(model, device: str, chunks, on_progress=None) -> dict:
    pipeline.move_model(model, device)
    try:
        return pipeline.transcribe_chunked(model, chunks, LANGUAGE, on_progress=on_progress)
    finally:
        pipeline.move_model(model, "cpu")


def process_job(conn, model, device: str, file_id: str) -> None:
    d = job_dir(file_id)
    original = next(d.glob("original.*"))

    log(f"[worker] [{file_id}] препроцессинг аудио...")
    db.sync_set_status(conn, file_id, status="processing", stage="preprocess", progress=None)
    t0 = time.time()

    def _report_preprocess_progress(pct: float) -> None:
        # прогресс -- не более чем удобство для UI, не должен уронить обработку задачи
        try:
            db.sync_set_status(conn, file_id, stage="preprocess", progress=round(pct, 1))
        except Exception as e:
            log(f"[worker] [{file_id}] не удалось записать прогресс препроцессинга: {e}")

    wav_path = pipeline.preprocess_audio(original, d, on_progress=_report_preprocess_progress)
    duration = pipeline.get_audio_duration(wav_path)
    preprocess_time = time.time() - t0
    log(f"[worker] [{file_id}] длительность аудио {duration:.0f}с, препроцессинг за {preprocess_time:.1f}с")

    log(f"[worker] [{file_id}] ASR (Whisper {WHISPER_MODEL})...")
    db.sync_set_status(conn, file_id, status="processing", stage="asr", progress=None)
    t0 = time.time()

    def _report_asr_progress(pct: float) -> None:
        try:
            db.sync_set_status(conn, file_id, stage="asr", progress=round(pct, 1))
        except Exception as e:
            log(f"[worker] [{file_id}] не удалось записать прогресс ASR: {e}")

    if duration > CHUNK_THRESHOLD_SEC:
        chunks = pipeline.split_audio_on_silence(wav_path, d, CHUNK_TARGET_SEC, CHUNK_MAX_SEC)
        result = _run_asr_chunked(model, device, chunks, on_progress=_report_asr_progress)
    else:
        result = _run_asr(model, device, wav_path, on_progress=_report_asr_progress)
    asr_time = time.time() - t0
    log(f"[worker] [{file_id}] ASR готов за {asr_time:.1f}с")

    log(f"[worker] [{file_id}] диаризация (NeMo MSDD)...")
    db.sync_set_status(conn, file_id, status="processing", stage="diarization", progress=None)
    t0 = time.time()
    diarization = pipeline.diarize_nemo(wav_path, d, device)
    diarization_time = time.time() - t0
    log(f"[worker] [{file_id}] диаризация готова за {diarization_time:.1f}с")

    log(f"[worker] [{file_id}] роли и экспорт...")
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
    log(f"[worker] [{file_id}] готово за {total_time:.1f}с, total_rtf={metrics['total_rtf']}")
    db.sync_set_status(conn, file_id, status="done", stage=None, metrics=metrics, transcript=transcript_dict)


def _log_gpu_memory() -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    log(f"[worker] GPU память: {free_gb:.2f}/{total_gb:.2f} ГБ свободно")
    if free_gb < 8:
        log(
            f"[worker] ВНИМАНИЕ: свободно всего {free_gb:.2f}ГБ, а Whisper large-v3 обычно "
            "требует ~14ГБ. Похоже, GPU-память ещё занята другим процессом (например, не до "
            "конца остановленным старым контейнером воркера после рестарта). Проверьте на "
            "хосте `nvidia-smi` (PID держащих память) и `docker compose ps` (нет ли лишних "
            "контейнеров) до того, как загрузка модели упадёт в OOM."
        )


def main() -> None:
    device = pipeline.detect_device()
    log(f"[worker] device: {device}")
    if device == "cuda":
        _log_gpu_memory()

    db.sync_init_schema()
    conn = db.sync_connect()

    log("[worker] loading Whisper...")
    model = pipeline.load_whisper_model(WHISPER_MODEL, "cpu")

    log("[worker] warming up NeMo (форсируем скачивание весов сразу)...")
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
        log(f"[worker] warm-up diarize пропущен (тишина, ожидаемо): {e}")

    redis_client = jobqueue.sync_client()

    # Восстанавливаем задачи, застрявшие в processing-списке Redis с прошлого (упавшего или
    # убитого) запуска воркера — раньше, чем reconcile пометит их ошибкой в Postgres, чтобы
    # успевшие восстановиться не помечались error и пересчитались с нуля вместо этого.
    recovered = jobqueue.sync_recover_stuck_jobs(redis_client)
    if recovered:
        log(f"[worker] после рестарта нашёл {len(recovered)} незавершённых задач(и), верну в работу: {recovered}")
    db.sync_reconcile_interrupted(conn, "воркер перезапустился во время обработки этой задачи", keep_file_ids=recovered)

    log("[worker] ready")

    while True:
        try:
            file_id = jobqueue.sync_pop_job(redis_client)
        except redis.exceptions.RedisError as e:
            log(f"[worker] Redis временно недоступен ({e}), переподключаюсь через 2с")
            time.sleep(2)
            try:
                redis_client.close()
            except Exception:
                pass
            redis_client = jobqueue.sync_client()
            continue

        if file_id is None:
            time.sleep(2)  # очередь пуста — LMOVE не блокирует сам, пауза на стороне воркера
            continue

        log(f"[worker] взял задачу {file_id}")
        try:
            process_job(conn, model, device, file_id)
        except Exception as e:
            log(f"[worker] задача {file_id} упала: {e}")
            db.sync_set_status(
                conn, file_id, status="error", stage=None,
                error=f"{e}\n{traceback.format_exc()[-2000:]}",
            )
        finally:
            # Если это тоже упадёт (та же протухшая коннекция) -- не даём убить весь процесс
            # из-за неснятой отметки processing: sync_recover_stuck_jobs подберёт её на
            # следующем рестарте воркера и пересчитает заново, это не потеря данных.
            try:
                jobqueue.sync_ack_job(redis_client, file_id)
            except redis.exceptions.RedisError as e:
                log(f"[worker] не удалось снять задачу {file_id} с processing-списка ({e}) — подберётся сама при следующем рестарте")


if __name__ == "__main__":
    main()
