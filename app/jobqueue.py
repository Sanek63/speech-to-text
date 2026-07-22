import os

import redis
import redis.asyncio as aredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_KEY = "speech:jobs:queue"
PROCESSING_KEY = "speech:jobs:processing"

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024 * 1024)))  # 4GB — видео-лекции тяжелее чистого аудио, но одному файлу всё ещё не даём забить диск


# --- Синхронный клиент — для воркера -----------------------------------------


def sync_client() -> redis.Redis:
    # socket_keepalive: соединение может простаивать по 10-20+ минут между задачами (пока
    # идёт ASR/диаризация) — без TCP keepalive такое долгое молчание рискует быть тихо
    # оборвано промежуточной сетью. socket_timeout теперь можно смело ставить (раньше не
    # выставляли — конфликтовал с ожиданием BLMOVE/BLPOP на сервере), все команды здесь не
    # блокирующие и быстрые — если соединение всё-таки подвиснет молча, лучше поймать
    # TimeoutError и переподключиться, чем ждать бесконечно.
    return redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_keepalive=True, socket_timeout=10)


def sync_pop_job(client: redis.Redis) -> str | None:
    """Неблокирующий pop — LMOVE (не BLMOVE/BLPOP). Изначально был BLPOP, потом BLMOVE с
    таймаутом ожидания на сервере, но обе блокирующие версии на этой сети регулярно ловили
    TimeoutError даже сразу после переподключения свежим клиентом: похоже, что-то по пути
    (не наш код) рвёт соединение, которое подолгу молчит в ожидании данных на сервере
    (тот же класс нестабильности сети, что уже встречался при скачивании весов). LMOVE
    отвечает сразу (есть элемент или нет) — соединение никогда не "висит открытым молча",
    и повод для такого таймаута исчезает. Логика распределения по PROCESSING_KEY (крэш-
    safety) та же, что была у BLMOVE — просто пуллинг с паузой вместо ожидания на сервере,
    см. вызывающий код в worker.py."""
    return client.lmove(QUEUE_KEY, PROCESSING_KEY, "LEFT", "RIGHT")


def sync_ack_job(client: redis.Redis, file_id: str) -> None:
    """Задача дообработана (успешно или с ошибкой, но обработанной) — убираем из
    processing-списка, чтобы она не считалась "зависшей" при следующем рестарте воркера."""
    client.lrem(PROCESSING_KEY, 1, file_id)


def sync_recover_stuck_jobs(client: redis.Redis) -> list[str]:
    """При старте воркера — всё, что осталось в PROCESSING_KEY с прошлого (упавшего или
    убитого) запуска, возвращаем в начало очереди на повторную обработку."""
    recovered: list[str] = []
    while True:
        file_id = client.rpoplpush(PROCESSING_KEY, QUEUE_KEY)
        if file_id is None:
            break
        recovered.append(file_id)
    return recovered


# --- Асинхронный клиент — для API --------------------------------------------


def async_client() -> aredis.Redis:
    return aredis.Redis.from_url(REDIS_URL, decode_responses=True, socket_keepalive=True, socket_timeout=10)


async def async_push_job(client: aredis.Redis, file_id: str) -> None:
    await client.rpush(QUEUE_KEY, file_id)
