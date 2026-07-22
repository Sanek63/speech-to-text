"""
Конвертирует уже скачанный (Azure CDN) чекпоинт openai-whisper в формат CTranslate2 для
faster-whisper — полностью локально, без единого обращения к huggingface.co. Официальный
путь (transformers.convert_openai_to_hf + ct2-transformers-converter) в одном месте тянет
generation_config с HF Hub — см. _openai_to_hf.py, где это заменено на локальное построение
той же конфигурации. Результат кешируется на диск (тот же model-cache volume, что уже
используется под веса Whisper/NeMo) — конвертация происходит один раз, не на каждый запуск
воркера.
"""
import os
from pathlib import Path


def _whisper_cache_dir() -> Path:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return Path(os.getenv("XDG_CACHE_HOME", default)) / "whisper"


def ct2_model_dir(model_name: str) -> Path:
    return _whisper_cache_dir() / f"ct2-{model_name}"


def ensure_ct2_model(model_name: str) -> Path:
    """
    Возвращает путь к готовой CTranslate2-модели, конвертируя её при первом вызове
    (кешируется — повторные вызовы с уже готовым результатом ничего не делают).
    """
    out_dir = ct2_model_dir(model_name)
    if (out_dir / "model.bin").exists():
        return out_dir

    checkpoint_path = _whisper_cache_dir() / f"{model_name}.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Не найден чекпоинт {checkpoint_path} — ожидается, что openai-whisper уже "
            "скачал его через load_whisper_model() до вызова ensure_ct2_model()."
        )

    import ctranslate2
    from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast

    from . import _openai_to_hf as conv

    import whisper as _whisper_pkg

    # _TOKENIZERS по умолчанию в _openai_to_hf.py указывает на raw.githubusercontent.com —
    # подменяем на файл, уже установленный локально вместе с пакетом openai-whisper, сеть не
    # трогаем вообще (см. docstring в _openai_to_hf.convert_tiktoken_to_hf).
    conv._TOKENIZERS["multilingual"] = os.path.join(
        os.path.dirname(_whisper_pkg.__file__), "assets", "multilingual.tiktoken"
    )

    hf_dir = _whisper_cache_dir() / f"hf-{model_name}"
    hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ct2_convert] openai-whisper -> HF формат ({checkpoint_path} -> {hf_dir})...", flush=True)
    model, is_multilingual, num_languages = conv.convert_openai_whisper_to_tfms(str(checkpoint_path), str(hf_dir))

    tokenizer = conv.convert_tiktoken_to_hf(is_multilingual, num_languages)
    feature_extractor = WhisperFeatureExtractor(feature_size=model.config.num_mel_bins)
    processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    processor.save_pretrained(str(hf_dir))

    # slow -> fast токенайзер, тоже локально (читает файлы, только что сохранённые выше)
    fast_tokenizer = WhisperTokenizerFast.from_pretrained(str(hf_dir))
    fast_tokenizer.save_pretrained(str(hf_dir), legacy_format=False)

    model.save_pretrained(str(hf_dir))

    print(f"[ct2_convert] HF -> CTranslate2 формат ({hf_dir} -> {out_dir})...", flush=True)
    converter = ctranslate2.converters.TransformersConverter(str(hf_dir))
    converter.convert(str(out_dir), quantization="float16", force=True)

    # На всякий случай: faster-whisper при отсутствии локального tokenizer.json в директории
    # модели тихо пытается скачать его с HF Hub (tokenizers.Tokenizer.from_pretrained(...)).
    # ct2-transformers-converter обычно копирует его сам, но не полагаемся на это молча.
    import shutil as _shutil

    for fname in ("tokenizer.json", "preprocessor_config.json"):
        src, dst = hf_dir / fname, out_dir / fname
        if src.exists() and not dst.exists():
            _shutil.copy(src, dst)

    if not (out_dir / "tokenizer.json").exists():
        raise RuntimeError(
            f"{out_dir}/tokenizer.json не появился после конвертации — faster-whisper в "
            "этом случае попытается скачать токенайзер с HuggingFace Hub, что здесь "
            "недоступно. Проверьте, что HF-конвертация выше действительно сохранила "
            f"tokenizer.json в {hf_dir}."
        )

    print(f"[ct2_convert] готово: {out_dir}", flush=True)
    return out_dir
