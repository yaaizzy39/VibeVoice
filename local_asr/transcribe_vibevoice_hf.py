#!/usr/bin/env python
"""Local VibeVoice-ASR-HF transcription runner.

This script uses the Hugging Face Transformers-compatible VibeVoice ASR model.
The audio is processed locally after the model files have been downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR-HF"


def configure_hf_cache() -> Path:
    """Use a repo-local Hugging Face cache unless the user already set one."""
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".hf-cache"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    return cache_dir


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _safe_stem(path: Path) -> str:
    stem = path.stem or "audio"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "audio"


def _format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""

    lines: list[str] = []
    index = 1
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get("Start", segment.get("start", 0))
        end = segment.get("End", segment.get("end", start))
        content = segment.get("Content", segment.get("content", segment.get("text", "")))
        speaker = segment.get("Speaker", segment.get("speaker"))
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        text = str(content).strip()
        if speaker is not None:
            text = f"[Speaker {speaker}] {text}"
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_format_srt_time(start_f)} --> {_format_srt_time(end_f)}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(lines)


def _select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        _fail("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
    return requested


def _select_dtype(torch: Any, requested: str, device: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _read_prompt(args: argparse.Namespace) -> str | None:
    parts: list[str] = []
    if args.prompt:
        parts.append(args.prompt.strip())
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            _fail(f"Prompt file not found: {prompt_path}")
        parts.append(prompt_path.read_text(encoding="utf-8").strip())
    prompt = "\n".join(part for part in parts if part)
    return prompt or None


def _decode(processor: Any, generated_ids: Any, return_format: str | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if return_format:
        kwargs["return_format"] = return_format
    value = processor.decode(generated_ids, **kwargs)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def transcribe_one(
    *,
    processor: Any,
    model: Any,
    torch: Any,
    audio_path: Path,
    prompt: str | None,
    dtype: Any,
    max_new_tokens: int,
    tokenizer_chunk_size: int | None,
) -> dict[str, Any]:
    if not audio_path.exists():
        _fail(f"Audio file not found: {audio_path}")

    model_device = next(model.parameters()).device
    print(f"Processing: {audio_path}")

    inputs = processor.apply_transcription_request(
        audio=str(audio_path),
        prompt=prompt,
    ).to(model_device, dtype)

    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if tokenizer_chunk_size:
        generate_kwargs["tokenizer_chunk_size"] = tokenizer_chunk_size

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    raw_text = _decode(processor, generated_ids)
    try:
        text_only = _decode(processor, generated_ids, return_format="transcription_only")
    except Exception:
        text_only = raw_text
    try:
        parsed = _decode(processor, generated_ids, return_format="parsed")
    except Exception:
        parsed = None

    return {
        "audio": str(audio_path),
        "prompt": prompt,
        "raw_text": raw_text,
        "transcription": text_only,
        "segments": parsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio locally with microsoft/VibeVoice-ASR-HF."
    )
    parser.add_argument("audio", nargs="+", help="Audio/video file path(s) to transcribe.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model ID or local model directory.")
    parser.add_argument("--output-dir", default="transcripts", help="Directory for .txt/.json outputs.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype. Use auto unless a specific PC needs an override.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument(
        "--tokenizer-chunk-size",
        type=int,
        default=None,
        help="Lower this if audio tokenization runs out of memory. Example: 64000.",
    )
    parser.add_argument("--prompt", default=None, help="Optional context/hotwords.")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 text file with context/hotwords.")
    parser.add_argument("--offline", action="store_true", help="Use only already-downloaded model files.")
    parser.add_argument("--write-srt", action="store_true", help="Also write .srt when timestamps parse cleanly.")
    args = parser.parse_args()

    cache_dir = configure_hf_cache()

    try:
        import torch
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    except ImportError as exc:
        _fail(
            "Required packages are missing. Run scripts/setup_vibevoice_asr_hf.ps1 first. "
            f"Import error: {exc}"
        )

    device = _select_device(torch, args.device)
    dtype = _select_dtype(torch, args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _read_prompt(args)

    print(f"Model: {args.model}")
    print(f"Hugging Face cache: {cache_dir}")
    print(f"Device: {device}")
    print(f"Dtype: {str(dtype).replace('torch.', '')}")
    if args.offline:
        print("Offline mode: enabled")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.offline,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.offline)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    written: list[Path] = []
    for audio in args.audio:
        result = transcribe_one(
            processor=processor,
            model=model,
            torch=torch,
            audio_path=Path(audio),
            prompt=prompt,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            tokenizer_chunk_size=args.tokenizer_chunk_size,
        )
        base = output_dir / _safe_stem(Path(audio))
        txt_path = base.with_suffix(".txt")
        json_path = base.with_suffix(".json")

        txt_path.write_text(str(result["transcription"]).strip() + "\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.extend([txt_path, json_path])

        if args.write_srt:
            srt_text = _segments_to_srt(result.get("segments"))
            if srt_text:
                srt_path = base.with_suffix(".srt")
                srt_path.write_text(srt_text, encoding="utf-8")
                written.append(srt_path)

    print("Written files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
