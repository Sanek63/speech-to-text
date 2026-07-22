import contextlib
from pathlib import Path
from typing import List, Tuple


@contextlib.contextmanager
def _hooked_progress(on_progress):
    """
    openai-whisper трекает свой прогресс через обычный `tqdm.tqdm(total=content_frames, ...)`
    внутри model.transcribe() — total считается в мел-фреймах, что прямо соответствует
    позиции по времени в аудио. Явного колбэка в публичном API нет, поэтому на время самого
    вызова подменяем класс tqdm на подкласс, перехватывающий .update(), и возвращаем
    оригинал в finally. Если в будущей версии whisper перестанет использовать tqdm именно
    так — хук просто не будет вызван, на саму транскрибацию это не влияет (graceful no-op,
    не падение).
    """
    if on_progress is None:
        yield
        return

    import tqdm as tqdm_module

    original_cls = tqdm_module.tqdm

    class _HookedTqdm(original_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hook_done = 0

        def update(self, n=1):
            super().update(n)
            self._hook_done += n
            if self.total:
                on_progress(min(100.0, self._hook_done / self.total * 100))

    tqdm_module.tqdm = _HookedTqdm
    try:
        yield
    finally:
        tqdm_module.tqdm = original_cls


def load_whisper_model(model_name: str, device: str):
    """
    Грузит openai-whisper (веса с CDN OpenAI, не с HF) один раз — вызывающая сторона держит
    объект в памяти между запросами и переиспользует через transcribe_with_model(), вместо
    того чтобы грузить модель заново на каждый файл.
    """
    import whisper  # тяжёлая зависимость — импортируется только здесь

    return whisper.load_model(model_name, device=device)


def transcribe_with_model(model, wav_path: Path, language: str, on_progress=None) -> dict:
    """
    ASR уже загруженной моделью, со встроенными word-level таймкодами — отдельная
    forced-alignment модель не нужна. on_progress(pct: 0..100), если передан, вызывается по
    ходу распознавания (см. _hooked_progress).
    """
    with _hooked_progress(on_progress):
        return model.transcribe(str(wav_path), language=language, word_timestamps=True)


def transcribe_chunked(model, chunks: List[Tuple[Path, float]], language: str, on_progress=None) -> dict:
    """
    Транскрибирует по кускам уже загруженной моделью: initial_prompt для связности
    терминологии между кусками, таймкоды сдвигаются в абсолютные по offset куска.
    on_progress получает глобальный процент — прогресс внутри текущего куска
    масштабируется на долю "один кусок из N", веса кусков считаем равными (чанки и так
    режутся на примерно равные по длительности куски, см. split_audio_on_silence).
    """
    all_segments = []
    prev_tail = ""
    n = len(chunks)
    for i, (chunk_path, offset) in enumerate(chunks):
        def _chunk_progress(pct: float, i: int = i) -> None:
            if on_progress:
                on_progress(min(100.0, (i + pct / 100) / n * 100))

        with _hooked_progress(_chunk_progress if on_progress else None):
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
