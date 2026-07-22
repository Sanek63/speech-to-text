import os

import redis
import redis.asyncio as aredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_KEY = "speech:jobs:queue"
PROCESSING_KEY = "speech:jobs:processing"

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))  # 500MB — не даём одному файлу забить диск


# --- Синхронный клиент — для воркера -----------------------------------------


def sync_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def sync_pop_job(client: redis.Redis, timeout_sec: int = 5) -> str | None:
    """Блокирующий pop с таймаутом. Используем BLMOVE (не BLPOP) — сервер атомарно
    перекладывает file_id в PROCESSING_KEY, а не просто удаляет из очереди. Если клиент
    (воркер) упадёт до того, как реально обработает задачу — Redis-соединение оборвётся
    (или сам процесс умрёт), но file_id останется в PROCESSING_KEY и будет подобран через
    sync_recover_stuck_jobs() на следующем старте, а не потеряется молча, как было бы с
    обычным BLPOP (тот удаляет элемент из списка сразу на сервере, и если клиент не успел
    прочитать ответ — например, из-за обрыва сокета/таймаута — элемент исчезает без следа)."""
    return client.blmove(QUEUE_KEY, PROCESSING_KEY, timeout_sec, "LEFT", "RIGHT")


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
    return aredis.Redis.from_url(REDIS_URL, decode_responses=True)


async def async_push_job(client: aredis.Redis, file_id: str) -> None:
    await client.rpush(QUEUE_KEY, file_id)
