from typing import List

from .models import Turn, Word


def group_into_turns(result: dict) -> List[Turn]:
    """
    Разворачивает word-level вывод ASR в последовательные реплики по спикеру.
    """
    words: List[Word] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            if "start" not in w or "end" not in w:
                continue  # слово, для которого не нашлось таймкода — пропускаем
            words.append(Word(
                text=w["word"], start=w["start"], end=w["end"],
                speaker=w.get("speaker", "UNKNOWN"), confidence=w.get("probability", 1.0),
            ))

    turns: List[Turn] = []
    for w in words:
        if turns and turns[-1].speaker == w.speaker:
            turns[-1].words.append(w)
            turns[-1].end = w.end
            turns[-1].text += " " + w.text
        else:
            turns.append(Turn(speaker=w.speaker, role="", start=w.start, end=w.end, text=w.text, words=[w]))
    return turns
