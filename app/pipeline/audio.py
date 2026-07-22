import subprocess
from pathlib import Path
from typing import List, Tuple


def preprocess_audio(input_path: Path, cache_dir: Path) -> Path:
    """
    ffmpeg: любой вход -> 16kHz mono WAV с loudness-нормализацией.
    """
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден в PATH — установите его перед запуском.")
    if not Path(input_path).exists():
        raise RuntimeError(f"Входной аудиофайл не найден: {input_path}")
    out_path = cache_dir / "audio_16k_mono.wav"
    if out_path.exists():
        return out_path
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", "-af", "loudnorm",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg упал (код {proc.returncode}). stderr:\n{proc.stderr[-3000:]}")
    return out_path


def get_audio_duration(wav_path: Path) -> float:
    """
    Длительность аудио в секундах через ffprobe.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def split_audio_on_silence(
    wav_path: Path, cache_dir: Path, target_sec: float, max_sec: float,
    silence_db: float = -30, min_silence_sec: float = 0.6,
) -> List[Tuple[Path, float]]:
    """
    Режет длинное аудио на куски по паузам (ffmpeg silencedetect), не по фиксированному
    времени — иначе можно разорвать слово или реплику посередине. Если пауза не находится до
    max_sec — режет жёстко на границе (единственный оправданный случай: сплошной монолог
    длиннее лимита). Возвращает [(путь_к_куску, абсолютный_offset_сек), ...].
    """
    import re

    proc = subprocess.run(
        ["ffmpeg", "-i", str(wav_path), "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_sec}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    silences = list(zip(starts, ends))

    total_duration = get_audio_duration(wav_path)

    cut_points = [0.0]
    while cut_points[-1] < total_duration - target_sec:
        window_start = cut_points[-1] + target_sec * 0.5
        window_end = min(cut_points[-1] + max_sec, total_duration)
        candidates = [(s + e) / 2 for s, e in silences if window_start <= (s + e) / 2 <= window_end]
        cut = min(candidates, key=lambda c: abs(c - (cut_points[-1] + target_sec))) if candidates else window_end
        cut_points.append(cut)
    cut_points.append(total_duration)

    chunks_dir = cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i + 1]
        chunk_path = chunks_dir / f"chunk_{i:03d}.wav"
        if not chunk_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-ss", str(start), "-to", str(end), "-c", "copy", str(chunk_path)],
                check=True, capture_output=True,
            )
        chunks.append((chunk_path, start))
    return chunks


def generate_silence_wav(path: Path, duration_sec: float = 2.0) -> Path:
    """
    Генерирует короткий тишинный wav на лету (ffmpeg lavfi) — используется для warm-up
    диаризации при старте сервиса, чтобы форсировать скачивание весов NeMo заранее.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(duration_sec), str(path),
        ],
        check=True, capture_output=True,
    )
    return path
