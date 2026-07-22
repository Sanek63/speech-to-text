from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


@dataclass
class Turn:
    speaker: str
    role: str
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class SpeakerFeatures:
    speaker: str
    total_duration: float
    turn_count: int
    avg_turn_duration: float
    is_first: bool
    is_last: bool
    question_density: float  # вопросительных конструкций на 100 слов
    teacher_marker_hits: int
    student_marker_hits: int
    word_count: int
