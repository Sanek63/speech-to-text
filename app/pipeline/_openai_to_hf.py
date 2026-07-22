#!/usr/bin/env python
"""Converts a Whisper model in OpenAI format to Hugging Face format."""
# Copyright 2022 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --- Локально изменённая копия ---------------------------------------------------------
# Источник: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/convert_openai_to_hf.py
# (не входит в pip-пакет transformers, только в его GitHub-репозиторий — скопировано сюда
# вместо обращения к GitHub на каждый запуск).
#
# Единственное содержательное изменение — _get_generation_config(): оригинал вызывает
# GenerationConfig.from_pretrained(f"openai/whisper-{version}") с HuggingFace Hub, что
# недоступно в сети, для которой собран весь этот проект (huggingface.co заблокирован
# целиком). Заменено на локальное построение той же конфигурации теми же формулами, которые
# openai-whisper использует у себя по умолчанию (whisper/decoding.py::_get_suppress_tokens),
# плюс захардкоженные alignment_heads для large-v3 (тот же список, что зашит в исходниках
# самого ctranslate2 — см. ctranslate2/converters/transformers.py::_WHISPER_ALIGNMENT_HEADS
# — то есть тоже не тянется по сети, просто взят оттуда как константа). Остальной код —
# дословная копия оригинала: перенос весов из чекпоинта openai-whisper и конвертация
# tiktoken-токенайзера в формат HF, уже полностью локальные операции.
# ------------------------------------------------------------------------------------------

import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any

import torch
from huggingface_hub.utils import insecure_hashlib
from tiktoken.load import load_tiktoken_bpe
from torch import nn
from tqdm import tqdm

from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.models.whisper.tokenization_whisper import LANGUAGES

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

_TOKENIZERS = {
    "multilingual": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "english": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
}

# ctranslate2/converters/transformers.py::_WHISPER_ALIGNMENT_HEADS["openai/whisper-large-v3"]
_LARGE_V3_ALIGNMENT_HEADS = [
    (7, 0), (10, 17), (12, 18), (13, 12), (16, 1),
    (17, 14), (19, 11), (21, 4), (24, 1), (25, 6),
]


def _get_generation_config(
    is_multilingual: bool,
    num_languages: int = 100,
    openai_version: str | None = None,
) -> GenerationConfig:
    """Строит generation config локально (см. пояснение в начале файла) вместо
    GenerationConfig.from_pretrained(repo) с HuggingFace Hub."""
    import whisper.tokenizer as whisper_tokenizer

    tokenizer = whisper_tokenizer.get_tokenizer(
        multilingual=is_multilingual, num_languages=num_languages, task="transcribe"
    )
    # та же формула, что whisper/decoding.py::_get_suppress_tokens строит по умолчанию
    # (suppress_tokens="-1") -- сохраняем поведение уже провалидированного пайплайна
    suppress = list(tokenizer.non_speech_tokens)
    suppress.extend([tokenizer.transcribe, tokenizer.translate, tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm])
    if tokenizer.no_speech is not None:
        suppress.append(tokenizer.no_speech)

    gen_cfg = GenerationConfig()
    gen_cfg.suppress_tokens = sorted(set(suppress))
    if openai_version == "large-v3":
        gen_cfg.alignment_heads = _LARGE_V3_ALIGNMENT_HEADS
    return gen_cfg


def remove_ignore_keys_(state_dict):
    ignore_keys = ["layers", "blocks"]
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)
        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _download(url: str, root: str) -> Any:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if insecure_hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return torch.load(io.BytesIO(model_bytes), weights_only=True)
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if insecure_hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not match. Please retry loading the model."
        )

    return torch.load(io.BytesIO(model_bytes), weights_only=True)


def convert_openai_whisper_to_tfms(
    checkpoint_path, pytorch_dump_folder_path
) -> "tuple[WhisperForConditionalGeneration, bool, int]":
    if ".pt" not in checkpoint_path:
        root = os.path.dirname(pytorch_dump_folder_path) or "."
        original_checkpoint = _download(_MODELS[checkpoint_path], root)
        openai_version = checkpoint_path
    else:
        original_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        openai_version = None

    dimensions = original_checkpoint["dims"]
    state_dict = original_checkpoint["model_state_dict"]
    proj_out_weights = state_dict["decoder.token_embedding.weight"]
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    tie_embeds = True
    ffn_dim = state_dict["decoder.layers.0.fc1.weight"].shape[0]

    # a hacky way to properly set up the bos/eos/pad token ids in the model
    endoftext_id = 50257 if dimensions["n_vocab"] > 51865 else 50256

    config = WhisperConfig(
        vocab_size=dimensions["n_vocab"],
        encoder_ffn_dim=ffn_dim,
        decoder_ffn_dim=ffn_dim,
        num_mel_bins=dimensions["n_mels"],
        d_model=dimensions["n_audio_state"],
        max_target_positions=dimensions["n_text_ctx"],
        encoder_layers=dimensions["n_audio_layer"],
        encoder_attention_heads=dimensions["n_audio_head"],
        decoder_layers=dimensions["n_text_layer"],
        decoder_attention_heads=dimensions["n_text_head"],
        max_source_positions=dimensions["n_audio_ctx"],
        eos_token_id=endoftext_id,
        bos_token_id=endoftext_id,
        pad_token_id=endoftext_id,
        decoder_start_token_id=endoftext_id + 1,
    )

    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    if tie_embeds:
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.proj_out.weight.data = proj_out_weights

    # determine those parameters from a model checkpoint as Whisper repo does
    is_multilingual = model.config.vocab_size >= 51865
    num_languages = model.config.vocab_size - 51765 - int(is_multilingual)

    model.generation_config = _get_generation_config(
        is_multilingual,
        num_languages,
        openai_version,
    )

    return model, is_multilingual, num_languages


# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def _bpe(mergeable_ranks, token: bytes, max_rank=None) -> "list[bytes]":
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def convert_tiktoken_bpe_to_hf(tiktoken_path: str):
    bpe_ranks = load_tiktoken_bpe(tiktoken_path)
    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(b):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    merges = []
    vocab = {}
    for token, rank in bpe_ranks.items():
        vocab[token_bytes_to_string(token)] = rank
        if len(token) == 1:
            continue
        merged = tuple(_bpe(bpe_ranks, token, max_rank=rank))
        if len(merged) == 2:  # account for empty token
            merges.append(" ".join(map(token_bytes_to_string, merged)))
    return vocab, merges


def convert_tiktoken_to_hf(
    multilingual: bool = True, num_languages: int = 100, time_precision=0.02
) -> WhisperTokenizer:
    # _TOKENIZERS по умолчанию указывает на raw.githubusercontent.com — ct2_convert.py
    # подменяет это на локальный путь к файлу, уже установленному вместе с пакетом
    # openai-whisper (whisper/assets/*.tiktoken), так что по факту сеть здесь не трогается.
    tiktoken_tokenizer_path = _TOKENIZERS["multilingual" if multilingual else "english"]
    start_of_transcript = ["<|endoftext|>", "<|startoftranscript|>"]
    control_tokens = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    # these are special tokens, not normalized
    language_tokens = [f"<|{k}|>" for k in list(LANGUAGES)[:num_languages]]
    # These are not special but normalized
    timestamp_tokens = [("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)]

    vocab, merges = convert_tiktoken_bpe_to_hf(tiktoken_tokenizer_path)

    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_file = f"{tmpdirname}/vocab.json"
        merge_file = f"{tmpdirname}/merges.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            writer.writelines(bpe_tokens + "\n" for bpe_tokens in merges)

        hf_tokenizer = WhisperTokenizer(vocab_file, merge_file)

    hf_tokenizer.add_tokens(start_of_transcript + language_tokens + control_tokens, special_tokens=True)
    hf_tokenizer.add_tokens(timestamp_tokens, special_tokens=False)
    return hf_tokenizer
