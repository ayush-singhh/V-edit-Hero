#!/usr/bin/env node
// Create .venv-voice with OmniVoice (voiceover) and Whisper (transcription).
//
// Usage:
//   npm run setup:voice                 install into .venv-voice
//   npm run setup:voice -- --download   also prefetch the model weights (several GB)
//   npm run setup:voice -- --cpu        force the CPU build of PyTorch
//
// PyTorch build: NVIDIA (nvidia-smi found) -> CUDA 12.8 wheels; macOS -> PyPI
// (Apple Silicon uses MPS); otherwise -> CPU wheels. Falls back to PyPI.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { venvDir, venvPython } from './voice-python.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const OMNIVOICE = 'omnivoice==0.2.1';
const TORCH = ['torch==2.8.0', 'torchaudio==2.8.0'];
const MODELS = ['k2-fsa/OmniVoice', 'eustlb/higgs-audio-v2-tokenizer', 'openai/whisper-large-v3-turbo'];

const run = (command, argv) => spawnSync(command, argv, { cwd: root, stdio: 'inherit', windowsHide: true }).status === 0;
const die = message => { console.error(message); process.exit(1); };

function findPython() {
  for (const command of process.platform === 'win32' ? ['py', 'python'] : ['python3', 'python']) {
    const r = spawnSync(command, ['-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], { encoding: 'utf8', windowsHide: true });
    if (r.status !== 0) continue;
    const [major, minor] = r.stdout.trim().split('.').map(Number);
    if (major === 3 && minor >= 10) return command;
  }
  die('Python 3.10+ not found. Install it, then rerun npm run setup:voice.');
}

function torchIndex() {
  if (args.includes('--cpu')) return 'https://download.pytorch.org/whl/cpu';
  if (process.platform === 'darwin') return null;
  const gpu = spawnSync('nvidia-smi', ['-L'], { encoding: 'utf8', windowsHide: true });
  return gpu.status === 0 ? 'https://download.pytorch.org/whl/cu128' : 'https://download.pytorch.org/whl/cpu';
}

if (!existsSync(venvPython)) {
  console.log(`creating ${venvDir}`);
  if (!run(findPython(), ['-m', 'venv', venvDir])) die('venv creation failed');
}
const pip = (...argv) => run(venvPython, ['-m', 'pip', 'install', ...argv]);
pip('--upgrade', 'pip');

const index = torchIndex();
console.log(`installing PyTorch from ${index ?? 'PyPI'}`);
if (!(index && pip(...TORCH, '--index-url', index)) && !pip(...TORCH)) die('PyTorch install failed');
if (!pip(OMNIVOICE)) die('OmniVoice install failed');

if (args.includes('--download')) {
  const code = `from huggingface_hub import snapshot_download\nfor m in ${JSON.stringify(MODELS)}: print(snapshot_download(m))`;
  if (!run(venvPython, ['-c', code])) die('model download failed (is huggingface.co reachable?)');
}

run(venvPython, ['-c', 'import torch, omnivoice; print("torch", torch.__version__, "| cuda", torch.cuda.is_available()); print("omnivoice ok")']);
console.log('Next: npm run transcribe -- <video>  |  npm run voice -- --text "Hello" --output vo.wav');
