---
name: omnivoice
description: Local voice and transcription for this workspace. Generates voiceover with OmniVoice (k2-fsa/OmniVoice, Apache-2.0) by voice cloning, voice design, or an automatic voice, and transcribes recordings to word-level JSON with local Whisper large-v3-turbo for the cutting tools. Use when asked to transcribe footage, make a voiceover or narration, clone the user's voice, design a voice, re-record a line, or caption generated speech. This is the workspace default for both; ElevenLabs is an opt-in alternative.
---

# OmniVoice: voice and transcription

Both run locally from one Python environment, `.venv-voice/`. Nothing is uploaded.
Model weights download once from Hugging Face into its cache (several GB in all).

| Job | Command | Output |
| --- | --- | --- |
| Transcribe a recording | `npm run transcribe -- <video-or-audio>` | `<input>.json`, word timestamps |
| Generate voiceover | `npm run voice -- --script vo.txt [voice options] -o vo.wav` | `vo.wav` + `vo.segments.json` |
| Save a cloned voice | `npm run voice -- --ref-audio me.wav --ref-text "..." --save-voice me.pt` | reusable `me.pt` |

OmniVoice itself is text-to-speech only. Transcription uses Whisper large-v3-turbo,
the ASR model OmniVoice already ships with, through Hugging Face transformers.

## Setup

Check `.venv-voice/` exists. If not, run `npm run setup:voice` (add `--download`
to prefetch weights, `--cpu` to force CPU PyTorch). It needs Python 3.10+ and
detects NVIDIA (CUDA 12.8 wheels), macOS (MPS), or CPU. Expect a few GB of disk.
A GPU is strongly preferred: voice generation and Whisper are slow on CPU, so tell
the user the expected wait before transcribing long footage on CPU.

## Transcription

```bash
npm run transcribe -- video-projects/<slug>/assets/raw.mp4             # -> raw.json
npm run transcribe -- raw.mp4 --language auto --output raw.json       # detect language
```

The JSON uses the ElevenLabs Scribe shape the kit expects
(`words: [{ text, start, end, type: "word" }]`, `audio_duration_secs` from ffprobe),
so it feeds `cut-silences`, `cut-mistakes`, captions, and beat anchors unchanged.

Whisper limits to account for:

- **It tidies speech.** Whisper often drops fillers ("um"), stutters, and repeated
  words. `cut-mistakes` relies on seeing them, so listen around suspected retakes
  and false starts rather than trusting the transcript alone.
- **It can invent text over silence or music.** Spot-check words that fall inside
  long pauses before cutting on them.
- **Word end times are approximate** (cross-attention alignment). Keep the
  cutting tools' padding defaults; do not tighten them for Whisper transcripts.

Reuse an existing verified transcript instead of re-running on the same take.
If the user asks for ElevenLabs Scribe, use `npm run transcribe:elevenlabs -- <input>`.

## Voiceover

Write the script as plain text, one paragraph per breath or beat, blank lines
between them. Each paragraph is generated separately and joined with `--pause`
seconds (default 0.35) of silence. `vo.segments.json` gives each paragraph's start
and end; for word-level captions run `npm run transcribe -- vo.wav` on the result.

Pick one voice mode:

- **Clone** (`--ref-audio clip.wav [--ref-text "exact words"]`): the most stable
  mode. Use a clean 3-10 second clip in the target language. Without `--ref-text`
  Whisper transcribes the clip. Save the result with `--save-voice` and reuse it
  with `--voice name.pt`.
- **Design** (`--instruct "female, low pitch, british accent"`): gender, age, pitch,
  whisper style, English accent, Chinese dialect. Trained on English and Chinese.
- **Auto**: no voice option; the model picks a voice.

Useful options: `--speed 1.1`, `--duration 8` (single segment only), `--num-step 16`
(faster draft), `--language English`, `--normalize-text` (needs `omnivoice[tn]`).
Inline tags work in the text: `[laughter]`, `[sigh]`, `[surprise-oh]`, and CMU
pronunciation overrides such as `[B EY1 S]`. See the OmniVoice README for the list.

Put generated audio in `video-projects/<slug>/assets/` and wire it into the
composition as a sibling audio element, as with any other track. Listen to every
segment before rendering: check names, numbers, and pacing against the script.

## Consent

Clone only the user's own voice, or a voice whose owner has given permission for
this use. Ask before cloning any other recognizable person, and never use a clone
to impersonate someone or to put words in a real person's mouth they did not
approve. Voice prompts (`*.pt`) and reference clips are personal biometric data:
keep them inside `video-projects/` or another gitignored path.
