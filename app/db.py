import json
import os
from datetime import datetime, timezone
from typing import Any

import psycopg
import psycopg.rows

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://speech:speech@postgres:5432/speech")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    file_id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'uploaded',
    stage TEXT,
    error TEXT,
    audio_filename TEXT,
    language TEXT,
    metrics JSONB,
    transcript JSONB,
    client_ip TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
"""


def _row_to_dict(row: dict) -> dict:
    """JSONB-колонки psycopg уже парсит в dict/list сам; timestamps -> ISO-строки для JSON-ответа."""
    out = dict(row)
    for key in ("created_at", "updated_at"):
        if isinstance(out.get(key), datetime):
            out[key] = out[key].isoformat()
    return out


# --- Синхронный клиент — для воркера (обычный процесс, без event loop) ------


def sync_connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=psycopg.rows.dict_row)


def sync_init_schema() -> None:
    with sync_connect() as conn:
        conn.execute(SCHEMA)


def sync_set_status(conn: psycopg.Connection, file_id: str, **fields: Any) -> None:
    if not fields:
        return
    columns = list(fields.keys())
    set_clause = ", ".join(f"{c} = %s" for c in columns) + ", updated_at = now()"
    values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE file_id = %s", (*values, file_id))


def sync_reconcile_interrupted(
    conn: psycopg.Connection, reason: str, keep_file_ids: list[str] | None = None
) -> None:
    """При старте воркера: джобы, застрявшие в processing (предыдущий воркер умер посреди
    работы), иначе UI будет вечно показывать прогресс, хотя их больше никто не считает.
    'queued' не трогаем — они целы в Redis (AOF-персистентность) и будут подхвачены заново.
    keep_file_ids — то, что jobqueue.sync_recover_stuck_jobs уже вернул в очередь на повтор:
    их не помечаем ошибкой, они будут пересчитаны с нуля."""
    if keep_file_ids:
        conn.execute(
            "UPDATE jobs SET status = 'error', stage = NULL, error = %s, updated_at = now() "
            "WHERE status = 'processing' AND file_id::text != ALL(%s)",
            (reason, keep_file_ids),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = 'error', stage = NULL, error = %s, updated_at = now() "
            "WHERE status = 'processing'",
            (reason,),
        )


# --- Асинхронный клиент — для API (FastAPI/uvicorn, event loop) -------------


async def async_connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        DATABASE_URL, autocommit=True, row_factory=psycopg.rows.dict_row
    )


async def async_init_schema(conn: psycopg.AsyncConnection) -> None:
    await conn.execute(SCHEMA)


async def async_create_job(
    conn: psycopg.AsyncConnection, file_id: str, client_ip: str, audio_filename: str
) -> None:
    await conn.execute(
        "INSERT INTO jobs (file_id, status, client_ip, audio_filename) VALUES (%s, 'uploaded', %s, %s)",
        (file_id, client_ip, audio_filename),
    )


async def async_set_status(conn: psycopg.AsyncConnection, file_id: str, **fields: Any) -> None:
    if not fields:
        return
    columns = list(fields.keys())
    set_clause = ", ".join(f"{c} = %s" for c in columns) + ", updated_at = now()"
    values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    await conn.execute(f"UPDATE jobs SET {set_clause} WHERE file_id = %s", (*values, file_id))


async def async_get_job(conn: psycopg.AsyncConnection, file_id: str) -> dict | None:
    cur = await conn.execute("SELECT * FROM jobs WHERE file_id = %s", (file_id,))
    row = await cur.fetchone()
    return _row_to_dict(row) if row else None


async def async_list_jobs(
    conn: psycopg.AsyncConnection, limit: int = 10, offset: int = 0
) -> tuple[list[dict], int]:
    """Список задач для истории в UI — без тяжёлого transcript и без client_ip (наружу не
    отдаём). Свежие сверху, вся история видна всем — сервис открытый, без авторизации."""
    cur = await conn.execute(
        "SELECT file_id, status, stage, error, audio_filename, language, metrics, "
        "created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
    )
    rows = await cur.fetchall()
    count_cur = await conn.execute("SELECT COUNT(*) AS n FROM jobs")
    total_row = await count_cur.fetchone()
    return [_row_to_dict(r) for r in rows], total_row["n"]
