import json
import urllib.request
from pathlib import Path
from typing import List, Tuple

from .config import NEMO_DIAR_CONFIG_URL


def diarize_nemo(wav_path: Path, cache_dir: Path, device: str) -> List[Tuple[float, float, str]]:
    """
    Диаризация через NVIDIA NeMo MSDD (веса тянутся с NGC, не с huggingface.co). Прогоняется
    на весь файл одним проходом (не чанкуется) — NeMo сама справляется с длинными записями
    через внутреннее скользящее окно, а глобальный проход даёт согласованные speaker-метки без
    риска перепутать спикеров между кусками (актуально было бы только при чанковании).

    NeMo, в отличие от pyannote, ожидает на входе manifest.json + YAML-конфиг, а не прямой
    вызов с аудио-массивом. Схема конфига диаризации крупная (вложенные `parameters` на
    VAD/эмбеддингах/кластеризации/MSDD) и меняется между релизами NeMo — URL в config.py
    зафиксирован и проверен под nemo_toolkit 2.7.3 (diarizer.msdd_model.parameters на месте).
    При смене версии nemo_toolkit в окружении может понадобиться другой тег. Файл качается с
    raw.githubusercontent.com (GitHub, не HF) и кэшируется локально.

    Пересобирается заново на каждую задачу, не держится "прогретым" между запросами:
    NeuralDiarizer — не обычный nn.Module с чистым .to(device), а обёртка со своим конфигом
    (manifest/out_dir меняются на каждый файл), и это самый безопасный, уже провалидированный
    паттерн после нескольких сюрпризов с NeMo API.
    """
    from nemo.collections.asr.models import NeuralDiarizer
    from omegaconf import OmegaConf

    diar_dir = cache_dir / "nemo_diarization"
    diar_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = diar_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "audio_filepath": str(wav_path.resolve()),
        "offset": 0, "duration": None, "label": "infer",
        "text": "-", "num_speakers": None, "rttm_filepath": None, "uem_filepath": None,
    }) + "\n", encoding="utf-8")

    config_path = cache_dir / "diar_infer_telephonic.yaml"
    if not config_path.exists():
        urllib.request.urlretrieve(NEMO_DIAR_CONFIG_URL, config_path)
    config = OmegaConf.load(config_path)

    config.diarizer.manifest_filepath = str(manifest_path)
    config.diarizer.out_dir = str(diar_dir)
    config.diarizer.oracle_vad = False
    config.diarizer.collar = 0.25
    config.diarizer.vad.model_path = "vad_multilingual_marblenet"
    config.diarizer.speaker_embeddings.model_path = "titanet_large"
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

    diarizer = NeuralDiarizer(cfg=config)
    diarizer.diarize()

    rttm_path = next((diar_dir / "pred_rttms").glob("*.rttm"))
    segments = []
    for line in rttm_path.read_text().splitlines():
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        start, dur, speaker = float(parts[3]), float(parts[4]), parts[7]
        segments.append((start, start + dur, f"SPEAKER_{speaker}"))
    return segments


def assign_word_speakers(result: dict, diarization: List[Tuple[float, float, str]]) -> dict:
    """
    Размечает каждое слово ASR спикером по максимальному перекрытию с диаризационным
    сегментом; если слово попало в паузу между сегментами — берёт ближайший по времени.
    """
    for segment in result["segments"]:
        for w in segment.get("words", []):
            w_start, w_end = w["start"], w["end"]
            best_speaker, best_overlap, best_gap, nearest_speaker = "UNKNOWN", 0.0, float("inf"), "UNKNOWN"
            for d_start, d_end, speaker in diarization:
                overlap = min(w_end, d_end) - max(w_start, d_start)
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, speaker
                gap = max(d_start - w_end, w_start - d_end, 0.0)
                if gap < best_gap:
                    best_gap, nearest_speaker = gap, speaker
            w["speaker"] = best_speaker if best_overlap > 0 else nearest_speaker
    return result
