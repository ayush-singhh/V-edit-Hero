#!/usr/bin/env python3
"""Generate voiceover with OmniVoice (k2-fsa/OmniVoice, Apache-2.0).

Three voice modes, matching OmniVoice's own API:
  clone   --ref-audio ref.wav [--ref-text "..."]   (3-10 s clip of the speaker)
  saved   --voice my-voice.pt                      (from --save-voice)
  design  --instruct "female, low pitch, british accent"
  auto    none of the above; the model picks a voice

A script file is split on blank lines. Each paragraph is generated separately,
then joined with --pause seconds of silence. A <output>.segments.json file
records each paragraph's start and end so beats can be timed against it.

Usage:
  npm run voice -- --script vo.txt --ref-audio me.wav --output vo.wav
  npm run voice -- --text "Hello there." --instruct "male, american accent"
  npm run voice -- --ref-audio me.wav --ref-text "..." --save-voice me.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "k2-fsa/OmniVoice"


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser(description="OmniVoice voiceover for video projects.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="text to speak")
    src.add_argument("--script", help="UTF-8 text file; blank lines separate segments")
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--ref-audio", help="reference clip for voice cloning (3-10 s)")
    voice.add_argument("--voice", help="saved voice prompt (.pt) from --save-voice")
    voice.add_argument("--instruct", help='voice design attributes, e.g. "female, british accent"')
    p.add_argument("--ref-text", help="transcript of --ref-audio (Whisper fills it in when omitted)")
    p.add_argument("--save-voice", help="write the cloned voice prompt to this .pt path")
    p.add_argument("-o", "--output", default="voiceover.wav", help="output WAV path")
    p.add_argument("--language", help='language name or code, e.g. "English" or "en"')
    p.add_argument("--speed", type=float, help="speaking rate (>1 faster)")
    p.add_argument("--duration", type=float, help="fixed duration in seconds (single segment only)")
    p.add_argument("--num-step", type=int, default=32, help="diffusion steps; 16 is faster")
    p.add_argument("--pause", type=float, default=0.35, help="silence between script segments, seconds")
    p.add_argument("--normalize-text", action="store_true", help='read numbers as words (needs omnivoice[tn])')
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps, xpu, or cpu")
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


def read_segments(opts):
    if opts.text:
        text = opts.text
    elif opts.script:
        text = Path(opts.script).read_text(encoding="utf-8")
    else:
        return []
    segments = [" ".join(block.split()) for block in text.replace("\r\n", "\n").split("\n\n")]
    return [s for s in segments if s]


def main():
    opts = parse_args()
    segments = read_segments(opts)
    if not segments and not opts.save_voice:
        die("nothing to do: pass --text or --script (or --ref-audio with --save-voice)")
    if opts.save_voice and not opts.ref_audio:
        die("--save-voice needs --ref-audio")
    if opts.ref_text and not opts.ref_audio:
        die("--ref-text needs --ref-audio")
    if opts.duration and len(segments) > 1:
        die("--duration applies to a single segment; use --speed for scripts")

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice, VoiceClonePrompt
    except ImportError:
        die("omnivoice missing. Run `npm run setup:voice` and use the .venv-voice Python.")

    device = pick_device(opts.device)
    dtype = torch.float16 if device.startswith(("cuda", "xpu")) else torch.float32
    print(f"loading {opts.model} on {device} ...")
    # Whisper is only needed to transcribe a reference clip given without --ref-text.
    load_asr = bool(opts.ref_audio and not opts.ref_text)
    model = OmniVoice.from_pretrained(opts.model, device_map=device, dtype=dtype, load_asr=load_asr)
    sr = model.sampling_rate

    prompt = None
    if opts.voice:
        prompt = VoiceClonePrompt.load(opts.voice)
    elif opts.ref_audio:
        # Encode the reference once and reuse it for every segment.
        prompt = model.create_voice_clone_prompt(ref_audio=opts.ref_audio, ref_text=opts.ref_text)
        if opts.save_voice:
            Path(opts.save_voice).parent.mkdir(parents=True, exist_ok=True)
            prompt.save(opts.save_voice)
            print(f"saved voice prompt -> {opts.save_voice}")
    if not segments:
        return

    kwargs = {"num_step": opts.num_step, "normalize_text": opts.normalize_text}
    if prompt is not None:
        kwargs["voice_clone_prompt"] = prompt
    if opts.instruct:
        kwargs["instruct"] = opts.instruct
    if opts.language:
        kwargs["language"] = opts.language
    if opts.speed:
        kwargs["speed"] = opts.speed
    if opts.duration:
        kwargs["duration"] = opts.duration

    gap = np.zeros(int(round(opts.pause * sr)), dtype=np.float32)
    pieces, timeline, cursor = [], [], 0.0
    t0 = time.time()
    for i, text in enumerate(segments):
        print(f"[{i + 1}/{len(segments)}] {text[:70]}{'...' if len(text) > 70 else ''}")
        audio = np.asarray(model.generate(text=text, **kwargs)[0], dtype=np.float32)
        if i:
            pieces.append(gap)
            cursor += len(gap) / sr
        pieces.append(audio)
        length = len(audio) / sr
        timeline.append({"index": i, "text": text, "start": round(cursor, 3), "end": round(cursor + length, 3)})
        cursor += length

    out = Path(opts.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.concatenate(pieces), sr)
    meta = {
        "audio": out.name,
        "sample_rate": sr,
        "duration_secs": round(cursor, 3),
        "model": opts.model,
        "mode": "clone" if prompt is not None else "design" if opts.instruct else "auto",
        "segments": timeline,
    }
    out.with_suffix(".segments.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok {time.time() - t0:.1f}s | {len(segments)} segments | {cursor:.1f}s audio | {out}")


if __name__ == "__main__":
    main()
