from pathlib import Path
from typing import List, Tuple


def load_whisper_model(model_name: str, device: str):
    """
    Грузит faster-whisper (CTranslate2) из локально сконвертированных весов openai-whisper
    -- см. ct2_convert.py, конвертация полностью без HuggingFace (веса берутся из уже
    скачанного с Azure CDN чекпоинта openai-whisper). local_files_only=True — на случай если
    что-то в цепочке всё же попытается сходить в сеть, падаем громко, а не тихо виснем.

    Компромисс с прежним паттерном (openai-whisper, где модель — обычный nn.Module с
    .to(device)): CTranslate2 фиксирует "исходное" устройство модели в момент создания и не
    меняет его — unload_model()/load_model() переключают, ГДЕ лежат веса (GPU/CPU), а не
    пересоздают модель. Поэтому модель создаётся сразу на целевом устройстве (GPU, если оно
    есть) и тут же выгружается на CPU — тот же результат ("резидентна на CPU между
    задачами"), только другим API.
    """
    from faster_whisper import WhisperModel

    from . import ct2_convert

    ct2_dir = ct2_convert.ensure_ct2_model(model_name)
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(str(ct2_dir), device=device, compute_type=compute_type, local_files_only=True)
    if device == "cuda":
        model.model.unload_model(to_cpu=True)
    return model


def _segments_to_dict(segments, offset: float = 0.0) -> List[dict]:
    # float(...): faster-whisper отдаёт start/end/probability как numpy.float64 -- сериализуется
    # в JSON и сегодня (numpy.float64 -- подкласс float), но не полагаемся на эту деталь
    # реализации молча, приводим к обычному Python float явно.
    result = []
    for seg in segments:
        words = [
            {"word": w.word, "start": float(w.start) + offset, "end": float(w.end) + offset, "probability": float(w.probability)}
            for w in (seg.words or [])
        ]
        result.append({"start": float(seg.start) + offset, "end": float(seg.end) + offset, "text": seg.text, "words": words})
    return result


def transcribe_with_model(model, wav_path: Path, language: str, on_progress=None) -> dict:
    """
    ASR через faster-whisper. В отличие от openai-whisper (который отдаёт результат только
    целиком после завершения), сегменты здесь приходят генератором по мере распознавания —
    прогресс считается напрямую по позиции текущего сегмента в аудио, без хаков с перехватом
    tqdm (см. историю в git — так делали для openai-whisper, здесь это больше не нужно).
    """
    segments, info = model.transcribe(str(wav_path), language=language, word_timestamps=True)
    duration = info.duration
    out_segments = []
    for seg in segments:
        out_segments.extend(_segments_to_dict([seg]))
        if on_progress and duration:
            on_progress(min(100.0, seg.end / duration * 100))
    return {"segments": out_segments}


def transcribe_chunked(model, chunks: List[Tuple[Path, float]], language: str, on_progress=None) -> dict:
    """
    Транскрибирует по кускам уже загруженной моделью: initial_prompt для связности
    терминологии между кусками, таймкоды сдвигаются в абсолютные по offset куска.
    on_progress получает глобальный процент — прогресс внутри текущего куска масштабируется
    на долю "один кусок из N" (чанки и так режутся на примерно равные по длительности куски,
    см. split_audio_on_silence, поэтому равный вес на кусок — разумное приближение).
    """
    all_segments = []
    prev_tail = ""
    n = len(chunks)
    for i, (chunk_path, offset) in enumerate(chunks):
        segments, info = model.transcribe(
            str(chunk_path), language=language, word_timestamps=True,
            initial_prompt=prev_tail or None,
        )
        chunk_duration = info.duration
        chunk_text_tail = ""
        for seg in segments:
            all_segments.extend(_segments_to_dict([seg], offset=offset))
            chunk_text_tail += seg.text
            if on_progress and chunk_duration:
                frac = min(1.0, seg.end / chunk_duration)
                on_progress(min(100.0, (i + frac) / n * 100))
        prev_tail = chunk_text_tail[-200:]

    return {"segments": all_segments}


def move_model(model, device: str) -> None:
    """
    faster-whisper: не .to(device), как в PyTorch. unload_model(to_cpu=True) освобождает GPU
    (веса остаются в CPU-памяти, не удаляются полностью), load_model() возвращает модель на
    исходное устройство, с которым она была создана в load_whisper_model() — то есть
    move_model(model, "cuda") имеет смысл, только если модель изначально создавалась с
    device="cuda" там.
    """
    if device == "cpu":
        model.model.unload_model(to_cpu=True)
        free_gpu_memory()
    else:
        model.model.load_model()


def free_gpu_memory() -> None:
    """
    gc.collect() + torch.cuda.empty_cache() — подчищает PyTorch-аллокатор (NeMo работает на
    PyTorch) после того, как Whisper освободил GPU через move_model(model, "cpu").
    """
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
