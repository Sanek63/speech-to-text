import os

import redis
import redis.asyncio as aredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_KEY = "speech:jobs:queue"

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))  # 500MB — не даём одному файлу забить диск


# --- Синхронный клиент — для воркера -----------------------------------------


def sync_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def sync_pop_job(client: redis.Redis, timeout_sec: int = 5) -> str | None:
    """Блокирующий pop с таймаутом — воркер время от времени просыпается сам по себе,
    даже без новых задач, чтобы не держать соединение вечно заблокированным."""
    item = client.blpop(QUEUE_KEY, timeout=timeout_sec)
    return item[1] if item else None


# --- Асинхронный клиент — для API --------------------------------------------


def async_client() -> aredis.Redis:
    return aredis.Redis.from_url(REDIS_URL, decode_responses=True)


async def async_push_job(client: aredis.Redis, file_id: str) -> None:
    await client.lpush(QUEUE_KEY, file_id)
