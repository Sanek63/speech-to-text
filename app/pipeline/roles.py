from typing import Dict, List, Tuple

from .config import (
    MIN_SIGNIFICANT_DURATION_RATIO,
    MIN_SIGNIFICANT_TURNS,
    QUESTION_MARKERS,
    ROLE_WEIGHTS,
    STUDENT_MARKERS,
    TEACHER_MARKERS,
)
from .models import SpeakerFeatures, Turn


def extract_role_features(turns: List[Turn]) -> Dict[str, SpeakerFeatures]:
    """
    Признаки по каждому спикеру для ролевого скоринга (см. docstring assign_roles).
    """
    speakers = {t.speaker for t in turns}
    features: Dict[str, SpeakerFeatures] = {}
    for sp in speakers:
        sp_turns = [t for t in turns if t.speaker == sp]
        total_duration = sum(t.end - t.start for t in sp_turns)
        turn_count = len(sp_turns)
        word_count = sum(len(t.words) for t in sp_turns)
        text = " ".join(t.text for t in sp_turns).lower()
        question_hits = text.count("?") + sum(text.count(m) for m in QUESTION_MARKERS)
        teacher_hits = sum(text.count(m) for m in TEACHER_MARKERS)
        student_hits = sum(text.count(m) for m in STUDENT_MARKERS)
        features[sp] = SpeakerFeatures(
            speaker=sp,
            total_duration=total_duration,
            turn_count=turn_count,
            avg_turn_duration=(total_duration / turn_count) if turn_count else 0.0,
            is_first=False,
            is_last=False,
            question_density=(question_hits / word_count * 100) if word_count else 0.0,
            teacher_marker_hits=teacher_hits,
            student_marker_hits=student_hits,
            word_count=word_count,
        )
    if turns:
        features[turns[0].speaker].is_first = True
        features[turns[-1].speaker].is_last = True
    return features


def _normalize(values: Dict[str, float]) -> Dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _split_by_score_gap(ranked: List[Tuple[str, float]]) -> int:
    """
    Сколько спикеров сверху списка считать 'преподавателем' — по наибольшему разрыву в score.
    """
    scores = [s for _, s in ranked]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    return gaps.index(max(gaps)) + 1


def assign_roles(features: Dict[str, SpeakerFeatures]) -> Dict[str, str]:
    """
    Оценивает не объём речи, а паттерн поведения ('преподавательский' vs 'докладывающий').

    Суммарное время речи как единственный признак ломается на защитах/гостевых докладах,
    где докладчик (студент) говорит большую часть времени, а модератор/экзаменатор
    (преподаватель) — редко, но направляет ход разговора и задаёт вопросы. Поэтому здесь
    число реплик, вопросность и позиция в разговоре (кто открыл/закрыл) весят не меньше
    длительности речи.
    """
    total_duration = sum(f.total_duration for f in features.values()) or 1.0
    significant = {
        sp: f for sp, f in features.items()
        if f.total_duration / total_duration >= MIN_SIGNIFICANT_DURATION_RATIO
        and f.turn_count >= MIN_SIGNIFICANT_TURNS
    }
    roles = {sp: "Other" for sp in features}
    if not significant:
        return roles
    if len(significant) == 1:
        only = next(iter(significant))
        roles[only] = "Неизвестно (только один значимый спикер)"
        return roles

    inv_avg = {sp: 1.0 / (f.avg_turn_duration + 1e-6) for sp, f in significant.items()}
    marker = {sp: f.teacher_marker_hits - f.student_marker_hits for sp, f in significant.items()}
    normed = {
        "turn_count": _normalize({sp: f.turn_count for sp, f in significant.items()}),
        "inv_avg_turn_duration": _normalize(inv_avg),
        "question_density": _normalize({sp: f.question_density for sp, f in significant.items()}),
        "marker_score": _normalize(marker),
    }

    scores: Dict[str, float] = {}
    for sp, f in significant.items():
        scores[sp] = (
            ROLE_WEIGHTS["turn_count"] * normed["turn_count"][sp]
            + ROLE_WEIGHTS["inv_avg_turn_duration"] * normed["inv_avg_turn_duration"][sp]
            + ROLE_WEIGHTS["is_first"] * (1.0 if f.is_first else 0.0)
            + ROLE_WEIGHTS["is_last"] * (1.0 if f.is_last else 0.0)
            + ROLE_WEIGHTS["question_density"] * normed["question_density"][sp]
            + ROLE_WEIGHTS["marker_score"] * normed["marker_score"][sp]
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n_teachers = _split_by_score_gap(ranked)
    teachers, students = ranked[:n_teachers], ranked[n_teachers:]

    for i, (sp, _) in enumerate(teachers, start=1):
        roles[sp] = "Преподаватель" if len(teachers) == 1 else f"Преподаватель {i}"
    for i, (sp, _) in enumerate(students, start=1):
        roles[sp] = "Студент" if len(students) == 1 else f"Студент {i}"
    return roles


def apply_roles(turns: List[Turn], role_map: Dict[str, str]) -> None:
    for t in turns:
        t.role = role_map.get(t.speaker, "Other")
