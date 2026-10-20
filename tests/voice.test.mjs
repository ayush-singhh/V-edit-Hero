import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';
const root = resolve(import.meta.dirname, '..');
// These checks need only system Python, not the model environment.
const python = ['python3', 'python'].find(p => spawnSync(p, ['--version']).status === 0);
const skip = python ? false : 'python not installed';
function py(code) { return spawnSync(python, ['-I', '-c', code], { encoding: 'utf8', cwd: root }); }

test('local Whisper chunks become kit transcript words', { skip }, () => {
  const out = mkdtempSync(join(tmpdir(), 'voice-test-'));
  const chunks = [
    { text: ' We', timestamp: [0.6, 0.85] },
    { text: ' ', timestamp: [0.85, 0.9] },
    { text: ' build', timestamp: [0.9, 1.2] },
    { text: ' light.', timestamp: [7.1, null] },
  ];
  writeFileSync(join(out, 'chunks.json'), JSON.stringify(chunks));
  const r = py(`
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("t", "scripts/transcribe-local.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
chunks = [dict(c, timestamp=tuple(c["timestamp"])) for c in json.load(open(${JSON.stringify(join(out, 'chunks.json'))}))]
print(json.dumps(m.to_words(chunks, 7.5)))`);
  assert.equal(r.status, 0, r.stderr);
  const words = JSON.parse(r.stdout);
  assert.deepEqual(words.map(w => w.text), ['We', 'build', 'light.']);
  assert.ok(words.every(w => w.type === 'word' && w.end >= w.start));
  assert.equal(words.at(-1).end, 7.5);
});

test('voice script rejects inconsistent options before loading models', { skip }, () => {
  const run = args => spawnSync(python, ['-I', join(root, 'scripts/voice-omnivoice.py'), ...args], { encoding: 'utf8' });
  assert.match(run(['--save-voice', 'x.pt']).stderr, /--save-voice needs --ref-audio/);
  assert.match(run(['--text', 'hi', '--ref-text', 'hi']).stderr, /--ref-text needs --ref-audio/);
  const script = join(mkdtempSync(join(tmpdir(), 'voice-test-')), 'vo.txt');
  writeFileSync(script, 'One.\n\nTwo.\n');
  assert.match(run(['--script', script, '--duration', '3']).stderr, /single segment/);
  assert.notEqual(run(['--voice', 'a.pt', '--instruct', 'male']).status, 0);
});
