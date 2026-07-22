NEMO_DIAR_CONFIG_URL = (
    "https://raw.githubusercontent.com/NVIDIA/NeMo/v2.7.3/examples/speaker_tasks/"
    "diarization/conf/inference/diar_infer_telephonic.yaml"
)

ROLE_WEIGHTS = {
    "turn_count": 0.25,
    "inv_avg_turn_duration": 0.15,
    "is_first": 0.15,
    "is_last": 0.10,
    "question_density": 0.25,
    "marker_score": 0.10,
}
MIN_SIGNIFICANT_DURATION_RATIO = 0.03  # спикеры <3% от общего времени речи — фоновая болтовня, не голосуют за роль
MIN_SIGNIFICANT_TURNS = 1

QUESTION_MARKERS = [
    "почему", "как вы думаете", "поясните", "объясните", "согласны ли",
    "в чём", "что если", "каким образом", "расскажите",
]
TEACHER_MARKERS = [
    "давайте", "переходим к", "на этом закончим", "спасибо за доклад", "коллеги",
    "следующий вопрос", "оппонент", "комиссия", "прошу вопросы", "передаю слово",
    "представьтесь", "у нас есть время",
]
STUDENT_MARKERS = [
    "я исследовал", "в своей работе", "целью моей работы было", "хотел бы представить",
    "моя работа посвящена", "в результате исследования", "мой доклад", "я хочу рассказать",
]
