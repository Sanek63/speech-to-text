import json
from pathlib import Path
from typing import List

from .models import Turn


def export_text(turns: List[Turn], path: Path) -> None:
    """
    Текст с ролями, напрямую сравнимый с эталонным script.pdf.
    """
    lines = [f"[{t.role}] {t.text.strip()}" for t in turns]
    path.write_text("\n\n".join(lines), encoding="utf-8")


def _srt_timestamp(seconds: float) -> str:
    ms_total = round(seconds * 1000)
    h, ms_total = divmod(ms_total, 3_600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_turn_for_srt(turn: Turn, max_words: int = 14) -> List[Turn]:
    """
    Режет длинную реплику на короткие субтитровые реплики по словам — иначе 20-минутный
    монолог докладчика превращается в один нечитаемый SRT-кадр.
    """
    if len(turn.words) <= max_words:
        return [turn]
    chunks = []
    for i in range(0, len(turn.words), max_words):
        chunk_words = turn.words[i : i + max_words]
        chunks.append(Turn(
            speaker=turn.speaker,
            role=turn.role,
            start=chunk_words[0].start,
            end=chunk_words[-1].end,
            text=" ".join(w.text for w in chunk_words),
            words=chunk_words,
        ))
    return chunks


def export_srt(turns: List[Turn], path: Path) -> None:
    """
    SRT-субтитры, каждый кадр помечен ролью спикера.
    """
    cues = [c for t in turns for c in _split_turn_for_srt(t)]
    lines = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}")
        lines.append(f"[{cue.role}] {cue.text.strip()}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


_SENTENCE_END_CHARS = (".", "!", "?", "…")
_MAX_SENTENCE_WORDS = 40  # предохранитель на случай куска текста совсем без пунктуации


def _split_into_sentences(words: List) -> List[List]:
    sentences = []
    current = []
    for w in words:
        current.append(w)
        ends_sentence = w.text.rstrip().endswith(_SENTENCE_END_CHARS)
        if ends_sentence or len(current) >= _MAX_SENTENCE_WORDS:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _split_turn_for_display(turn: Turn, sentences_per_chunk: int = 2) -> List[Turn]:
    """
    Режет длинную реплику (например, 10-минутный монолог студента на защите — с ролью это
    уже никак не связано, роль назначается раньше, на уровне непрерывной по спикеру реплики
    в roles.py) на кусочки по 1-2 предложения — иначе в UI это один нечитаемый блок без
    возможности перемотать на нужный момент внутри него. speaker/role наследуются как есть,
    это разбивка для отображения, а не переклассификация.
    """
    sentences = _split_into_sentences(turn.words)
    if len(sentences) <= 1:
        return [turn]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_words = [w for sentence in sentences[i : i + sentences_per_chunk] for w in sentence]
        chunks.append(Turn(
            speaker=turn.speaker, role=turn.role,
            start=chunk_words[0].start, end=chunk_words[-1].end,
            text=" ".join(w.text for w in chunk_words), words=chunk_words,
        ))
    return chunks


def turns_to_dict(turns: List[Turn], audio_file: str = "", language: str = "") -> dict:
    """
    Общее представление транскрипта (объект с метаданными + реплики с word-level таймкодами
    и confidence) — используется и export_json() для файла, и backend'ом для отдачи по API,
    без дублирования схемы в двух местах. Реплики здесь уже раздроблены на 1-2 предложения
    для отображения (см. _split_turn_for_display) — роль/спикер посчитаны раньше, на исходных
    непрерывных репликах, и не пересчитываются.
    """
    display_turns = [c for t in turns for c in _split_turn_for_display(t)]
    return {
        "audio_file": audio_file,
        "language": language,
        "duration": display_turns[-1].end if display_turns else 0.0,
        "turns": [
            {
                "id": i,
                "speaker": t.speaker,
                "role": t.role,
                "start": t.start,
                "end": t.end,
                "text": t.text.strip(),
                "words": [
                    {"word": w.text, "start": w.start, "end": w.end, "confidence": round(w.confidence, 3)}
                    for w in t.words
                ],
            }
            for i, t in enumerate(display_turns)
        ],
    }


def export_json(turns: List[Turn], path: Path, audio_file: str = "", language: str = "") -> None:
    """
    Структурированный JSON под плеер: объект с метаданными файла + список реплик со
    стабильными id и word-level таймкодами (перемотка аудио по клику на реплику/слово, live-
    подсветка текущего слова при воспроизведении по currentTime).
    """
    data = turns_to_dict(turns, audio_file=audio_file, language=language)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
