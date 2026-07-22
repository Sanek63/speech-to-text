import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg.rows
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import AsyncConnectionPool

import db
import jobqueue

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
STATIC_DIR = BASE_DIR / "static"


def job_dir(file_id: str) -> Path:
    return DATA_DIR / file_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    app.state.db_pool = AsyncConnectionPool(
        db.DATABASE_URL, open=False,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row},
    )
    await app.state.db_pool.open()
    async with app.state.db_pool.connection() as conn:
        await db.async_init_schema(conn)

    app.state.redis = jobqueue.async_client()
    print("[startup] ready", flush=True)

    yield

    await app.state.db_pool.close()
    await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    try:
        async with app.state.db_pool.connection() as conn:
            await conn.execute("SELECT 1")
        await app.state.redis.ping()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Загрузка открыта для всех, без авторизации и без лимита на число загрузок — единственный
    предохранитель: потолок размера одного файла (обрыв стрима на превышении), чтобы один
    файл не занял весь диск."""
    client_ip = request.client.host if request.client else "unknown"

    file_id = str(uuid.uuid4())
    d = job_dir(file_id)
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio").suffix or ".bin"
    dest = d / f"original{ext}"

    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > jobqueue.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Файл больше {jobqueue.MAX_UPLOAD_BYTES // (1024 * 1024)}МБ")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        try:
            d.rmdir()
        except OSError:
            pass
        raise

    async with app.state.db_pool.connection() as conn:
        await db.async_create_job(conn, file_id, client_ip)

    return {"file_id": file_id}


@app.post("/api/transcribe/{file_id}")
async def transcribe(file_id: str):
    async with app.state.db_pool.connection() as conn:
        job = await db.async_get_job(conn, file_id)
        if job is None:
            raise HTTPException(404, "file_id не найден")
        await db.async_set_status(conn, file_id, status="queued", stage=None, error=None)
    await jobqueue.async_push_job(app.state.redis, file_id)
    return {"status": "queued"}


@app.get("/api/status/{file_id}")
async def status(file_id: str):
    async with app.state.db_pool.connection() as conn:
        job = await db.async_get_job(conn, file_id)
    if job is None:
        raise HTTPException(404, "file_id не найден")
    job.pop("transcript", None)  # лёгкий эндпоинт под частый опрос — транскрипт отдаётся отдельно
    return job


@app.get("/api/transcript/{file_id}")
async def transcript(file_id: str):
    async with app.state.db_pool.connection() as conn:
        job = await db.async_get_job(conn, file_id)
    if job is None or job.get("transcript") is None:
        raise HTTPException(404, "транскрипт ещё не готов")
    return job["transcript"]


@app.get("/api/audio/{file_id}")
async def audio(file_id: str):
    d = job_dir(file_id)
    candidates = list(d.glob("original.*")) if d.exists() else []
    if not candidates:
        raise HTTPException(404, "аудио не найдено")
    return FileResponse(candidates[0])


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
