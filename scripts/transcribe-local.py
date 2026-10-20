#!/usr/bin/env python3
"""Transcribe a video/audio file locally with Whisper -> word-level JSON.

Uses the same Whisper model OmniVoice uses for its own ASR
(openai/whisper-large-v3-turbo) through Hugging Face transformers, so the
voice venv from `npm run setup:voice` covers both voice and transcription.
Nothing is uploaded; the model downloads once into the Hugging Face cache.

Usage:
  npm run transcribe -- <input> [options]
  .venv-voice/bin/python scripts/transcribe-local.py <input> [options]

Defaults:
  --output    <input-without-ext>.json next to the source
  --model     openai/whisper-large-v3-turbo
  --language  en (use "auto" to let Whisper detect it)
  --device    auto (cuda, then mps, then cpu)

Output matches the ElevenLabs Scribe shape the cutting tools read:
  { language_code, audio_duration_secs, text, words: [{ text, start, end, type }] }
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16000
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser(description="Local Whisper word-level transcription for the editing tools.")
    p.add_argument("input", help="video or audio file")
    p.add_argument("-o", "--output", help="output JSON path (default: <input>.json)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    p.add_argument("--language", "--lang", default="en", help='language code, or "auto" to detect')
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps, or cpu")
    p.add_argument("--batch-size", type=int, default=8, help="30-second chunks decoded per batch")
    return p.parse_args()


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def probe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def load_audio(path):
    """Decode any ffmpeg-readable input to 16 kHz mono float32."""
    import numpy as np

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        die(f"ffmpeg failed:\n{r.stderr.decode(errors='replace')}", r.returncode)
    return np.frombuffer(r.stdout, dtype=np.float32)


def to_words(chunks, duration):
    """Normalize pipeline word chunks to the kit's transcript words."""
    words = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        start, end = chunk.get("timestamp", (None, None))
        if not text or start is None:
            continue
        # The last word of a file can come back without an end time.
        if end is None or end < start:
            end = duration if duration is not None else start
        words.append({"text": text, "start": round(float(start), 3), "end": round(float(end), 3), "type": "word"})
    return words


def main():
    opts = parse_args()
    input_path = Path(opts.input).resolve()
    if not input_path.exists():
        die(f"input not found: {input_path}")
    output_path = Path(opts.output).resolve() if opts.output else input_path.with_suffix(".json")

    try:
        import torch
        from transformers import pipeline
    except ImportError:
        die("torch/transformers missing. Run `npm run setup:voice` and use the .venv-voice Python.")

    device = pick_device(opts.device)
    dtype = torch.float16 if device.startswith(("cuda", "xpu")) else torch.float32
    duration = probe_duration(input_path)

    print(f"decoding audio from {input_path.name} ...")
    audio = load_audio(input_path)
    if duration is None:
        duration = len(audio) / SAMPLE_RATE

    print(f"loading {opts.model} on {device} ...")
    asr = pipeline("automatic-speech-recognition", model=opts.model, dtype=dtype, device=device)

    generate_kwargs = {"task": "transcribe"}
    if opts.language != "auto":
        generate_kwargs["language"] = opts.language

    t0 = time.time()
    result = asr(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        chunk_length_s=30,
        batch_size=opts.batch_size,
        return_timestamps="word",
        generate_kwargs=generate_kwargs,
    )
    dt = time.time() - t0

    words = to_words(result.get("chunks", []), duration)
    transcript = {
        "language_code": opts.language,
        "audio_duration_secs": round(duration, 3),
        "text": result.get("text", "").strip(),
        "words": words,
        "provider": "local-whisper",
        "model": opts.model,
    }
    output_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok {dt:.1f}s | {len(words)} words | {duration:.1f}s audio | {output_path}")


if __name__ == "__main__":
    main()
