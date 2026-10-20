#!/usr/bin/env node
// Run a Python script with the .venv-voice interpreter from npm scripts.
//   node scripts/voice-python.mjs scripts/transcribe-local.py <args...>
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const venvDir = join(root, '.venv-voice');
export const venvPython = process.platform === 'win32'
  ? join(venvDir, 'Scripts', 'python.exe')
  : join(venvDir, 'bin', 'python');

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [script, ...rest] = process.argv.slice(2);
  if (!script) { console.error('usage: node scripts/voice-python.mjs <script.py> [args...]'); process.exit(1); }
  if (!existsSync(venvPython)) { console.error('.venv-voice not found. Run: npm run setup:voice'); process.exit(1); }
  // npm runs scripts from the package root; keep the caller's directory for relative paths.
  const cwd = process.env.INIT_CWD ?? process.cwd();
  const r = spawnSync(venvPython, [resolve(root, script), ...rest], { cwd, stdio: 'inherit', windowsHide: true });
  process.exit(r.status ?? 1);
}
