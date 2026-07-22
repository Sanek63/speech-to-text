import difflib
import re
from pathlib import Path
from typing import List

from .models import Turn


def _wer(reference: List[str], hypothesis: List[str]) -> float:
    """
    Word error rate через расстояние Левенштейна на уровне слов.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if reference[i - 1] == hypothesis[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n


def evaluate_against_pdf(turns: List[Turn], pdf_path: Path) -> None:
    """
    Сравнивает вывод пайплайна с эталонным транскриптом: PDF с ролями в квадратных
    скобках, без таймкодов — поэтому сопоставление текстовое (difflib), а не по времени.
    """
    import fitz  # PyMuPDF — тяжёлая зависимость, импортируется только здесь

    text = "".join(page.get_text() for page in fitz.open(pdf_path))
    parts = re.split(r"\[(Преподаватель|Студент)\]", text)

    ref_words, ref_roles = [], []
    for role, chunk in zip(parts[1::2], parts[2::2]):
        for w in re.findall(r"\w+", chunk.lower()):
            ref_words.append(w)
            ref_roles.append(role)

    hyp_words, hyp_roles = [], []
    for t in turns:
        for w in t.words:
            for tok in re.findall(r"\w+", w.text.lower()):
                hyp_words.append(tok)
                hyp_roles.append(t.role)

    wer = _wer(ref_words, hyp_words)

    matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    matched = correct = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            ref_role, hyp_role = ref_roles[block.a + k], hyp_roles[block.b + k]
            matched += 1
            if ref_role in hyp_role or hyp_role in ref_role:  # "Преподаватель" ~ "Преподаватель 1"
                correct += 1

    print(f"WER: {wer:.1%}  ({len(ref_words)} эталонных слов)")
    if matched:
        print(f"Role accuracy: {correct / matched:.1%}  ({matched} выровненных слов)")
    else:
        print("Role accuracy: недостаточно совпадений текста для оценки")
