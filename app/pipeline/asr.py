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


def _fallback_word_spans(raw_words, seg_start: float, seg_end: float):
    """
    Изредка word-level DTW-выравнивание faster-whisper не справляется с длинным сегментом и
    отдаёт всем словам в нём ОДИНАКОВЫЙ таймкод (обычно = началу сегмента) -- сам текст при
    этом декодирован верно, страдает только точность таймкодов. Реплика с start==end потом
    никогда не подсвечивается при воспроизведении (t >= start и t < end одновременно
    невозможны). Границы самого СЕГМЕНТА -- более фундаментальная часть декодирования
    Whisper (спецтокены таймстемпов в основном цикле генерации), не зависящая от отдельного
    шага word-level DTW-выравнивания, поэтому остаются осмысленными даже в этом случае.
    Раскидываем слова по границам сегмента пропорционально длине слова -- грубое
    приближение, но лучше вырожденного нуля.
    """
    total_chars = sum(len(w.word) for w in raw_words) or 1
    span = max(seg_end - seg_start, 0.01)
    cursor = seg_start
    for w in raw_words:
        share = len(w.word) / total_chars * span
        yield w, cursor, cursor + share
        cursor += share


def _segments_to_dict(segments, offset: float = 0.0) -> List[dict]:
    # float(...): faster-whisper отдаёт start/end/probability как numpy.float64 -- сериализуется
    # в JSON и сегодня (numpy.float64 -- подкласс float), но не полагаемся на эту деталь
    # реализации молча, приводим к обычному Python float явно.
    result = []
    for seg in segments:
        raw_words = seg.words or []
        seg_start, seg_end = float(seg.start), float(seg.end)
        degenerate = len(raw_words) > 1 and all(
            float(w.start) == float(raw_words[0].start) and float(w.end) == float(raw_words[0].end)
            for w in raw_words
        )
        if degenerate:
            words = [
                {"word": w.word, "start": w_start + offset, "end": w_end + offset, "probability": float(w.probability)}
                for w, w_start, w_end in _fallback_word_spans(raw_words, seg_start, seg_end)
            ]
        else:
            words = [
                {"word": w.word, "start": float(w.start) + offset, "end": float(w.end) + offset, "probability": float(w.probability)}
                for w in raw_words
            ]
        result.append({"start": seg_start + offset, "end": seg_end + offset, "text": seg.text, "words": words})
    return result


def transcribe_with_model(model, wav_path: Path, language: str, on_progress=None) -> dict:
    """
    ASR через faster-whisper. В отличие от openai-whisper (который отдаёт результат только
    целиком после завершения), сегменты здесь приходят генератором по мере распознавания —
    прогресс считается напрямую по позиции текущего сегмента в аудио, без хаков с перехватом
    tqdm (см. историю в git — так делали для openai-whisper, здесь это больше не нужно).

    condition_on_previous_text=False: по умолчанию (True) текст каждого ~30-секундного окна
    декодирования попадает в промпт следующего окна -- стоит один раз получить правдоподобную,
    но галлюцинированную фразу, модель, затравленная своим же выводом, начинает сама себя
    воспроизводить до конца файла (наблюдали на реальной 30-минутной записи: с некоторого
    момента одно и то же предложение повторялось 15+ раз подряд до самого конца). Каждое
    повторение -- чистый, беглый сегмент, поэтому compression_ratio_threshold/log_prob_threshold
    его не ловят (они судят один сегмент, не повтор между сегментами).
    """
    segments, info = model.transcribe(
        str(wav_path), language=language, word_timestamps=True,
        condition_on_previous_text=False,
    )
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

    condition_on_previous_text=False (см. transcribe_with_model) -- не мешает initial_prompt:
    тот засеивает только самое первое окно каждого вызова transcribe() независимо от этого
    флага, который влияет лишь на окна 2+ внутри одного вызова. Кросс-чанковая связность
    терминологии сохраняется, а риск самоусиливающегося повтора внутри длинного куска
    (до CHUNK_MAX_SEC=720с, то есть ~24 окна) -- снят тем же способом, что и для обычного
    (нечанкованного) пути.
    """
    all_segments = []
    prev_tail = ""
    n = len(chunks)
    for i, (chunk_path, offset) in enumerate(chunks):
        segments, info = model.transcribe(
            str(chunk_path), language=language, word_timestamps=True,
            initial_prompt=prev_tail or None,
            condition_on_previous_text=False,
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
