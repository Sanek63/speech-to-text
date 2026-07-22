#!/usr/bin/env python3
"""
CLI поверх пакета pipeline — поведение то же, что раньше было в pipeline_nemo.py.

    python -m pipeline.cli audio/audio.mp3 --reference audio/script.pdf
"""
import argparse
import json
from pathlib import Path

from .asr import free_gpu_memory, load_whisper_model, transcribe_chunked, transcribe_with_model
from .audio import preprocess_audio, split_audio_on_silence
from .device import detect_device
from .diarization import assign_word_speakers, diarize_nemo
from .evaluate import evaluate_against_pdf
from .export import export_json, export_srt, export_text
from .roles import apply_roles, assign_roles, extract_role_features
from .turns import group_into_turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Путь к аудио/видео файлу")
    parser.add_argument("--outdir", type=Path, default=Path("output_nemo"))
    parser.add_argument("--reference", type=Path, default=None, help="PDF с эталонным транскриптом для оценки качества")
    parser.add_argument("--model", default="large-v3", help="Название модели openai-whisper")
    parser.add_argument("--device", default=None, help="cuda/cpu, по умолчанию автоопределение (NeMo не поддерживает mps)")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--force", action="store_true", help="Пересчитать ASR+диаризацию, игнорируя кэш")
    parser.add_argument("--chunked", action="store_true",
                         help="Резать ASR на куски по паузам (для очень длинных записей; диаризация всё равно идёт одним проходом на весь файл)")
    parser.add_argument("--chunk-target-sec", type=float, default=600)
    parser.add_argument("--chunk-max-sec", type=float, default=720)
    args = parser.parse_args()

    device = args.device or detect_device()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"Препроцессинг аудио ({device})...")
    wav_path = preprocess_audio(args.input, cache_dir)

    transcript_cache = cache_dir / "transcript_nemo.json"
    if transcript_cache.exists() and not args.force:
        print(f"Использую кэш: {transcript_cache}")
        result = json.loads(transcript_cache.read_text(encoding="utf-8"))
    else:
        print("ASR (openai-whisper)...")
        model = load_whisper_model(args.model, device)
        if args.chunked:
            chunks = split_audio_on_silence(wav_path, cache_dir, args.chunk_target_sec, args.chunk_max_sec)
            print(f"Кусков: {len(chunks)}")
            result = transcribe_chunked(model, chunks, args.language)
        else:
            result = transcribe_with_model(model, wav_path, args.language)
        del model
        free_gpu_memory()
        print("Диаризация (NeMo MSDD)...")
        diarization = diarize_nemo(wav_path, cache_dir, device)
        result = assign_word_speakers(result, diarization)
        transcript_cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    turns = group_into_turns(result)
    features = extract_role_features(turns)
    role_map = assign_roles(features)
    apply_roles(turns, role_map)

    print("Роли по спикерам:")
    for sp, role in role_map.items():
        f = features[sp]
        print(f"  {sp}: {role}  (речь {f.total_duration:.0f}с, {f.turn_count} реплик)")

    export_text(turns, args.outdir / "transcript.txt")
    export_srt(turns, args.outdir / "transcript.srt")
    export_json(turns, args.outdir / "transcript.json", audio_file=args.input.name, language=args.language)
    print(f"Готово: {args.outdir}/transcript.{{txt,srt,json}}")

    if args.reference:
        evaluate_against_pdf(turns, args.reference)


if __name__ == "__main__":
    main()
