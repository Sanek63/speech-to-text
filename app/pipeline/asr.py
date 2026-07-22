from pathlib import Path
from typing import List, Tuple


def load_whisper_model(model_name: str, device: str):
    """
    Грузит openai-whisper (веса с CDN OpenAI, не с HF) один раз — вызывающая сторона держит
    объект в памяти между запросами и переиспользует через transcribe_with_model(), вместо
    того чтобы грузить модель заново на каждый файл.
    """
    import whisper  # тяжёлая зависимость — импортируется только здесь

    return whisper.load_model(model_name, device=device)


def transcribe_with_model(model, wav_path: Path, language: str) -> dict:
    """
    ASR уже загруженной моделью, со встроенными word-level таймкодами — отдельная
    forced-alignment модель не нужна.
    """
    return model.transcribe(str(wav_path), language=language, word_timestamps=True)


def transcribe_chunked(model, chunks: List[Tuple[Path, float]], language: str) -> dict:
    """
    Транскрибирует по кускам уже загруженной моделью: initial_prompt для связности
    терминологии между кусками, таймкоды сдвигаются в абсолютные по offset куска.
    """
    all_segments = []
    prev_tail = ""
    for chunk_path, offset in chunks:
        result = model.transcribe(
            str(chunk_path), language=language, word_timestamps=True,
            initial_prompt=prev_tail or None,
        )
        for seg in result["segments"]:
            seg["start"] += offset
            seg["end"] += offset
            for w in seg.get("words", []):
                w["start"] += offset
                w["end"] += offset
        all_segments.extend(result["segments"])
        prev_tail = result["text"][-200:]

    return {"segments": all_segments}


def move_model(model, device: str) -> None:
    """
    Переносит модель между CPU/GPU. Whisper держится на CPU между запросами и переезжает на
    GPU только на время самого ASR-вызова — на одной GPU-карте Whisper large-v3 и три модели
    NeMo (VAD/эмбеддинги/MSDD) одновременно резидентными не помещаются.
    """
    model.to(device)
    if device == "cpu":
        free_gpu_memory()


def free_gpu_memory() -> None:
    """
    gc.collect() + torch.cuda.empty_cache() — вызывается после `del model` (CLI, одноразовый
    процесс) или после move_model(model, "cpu") (backend, модель живёт между запросами).
    """
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
