import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const linuxOnly = process.platform !== 'linux' && 'Linux ONNX Runtime build cache';
const cpu = 'onnxruntime-cpu-1.24.1';
const legacy = 'onnxruntime-1.24.1';
const archiveName = 'onnxruntime-linux-x64-1.24.1';
const hash = value => createHash('sha256').update(value).digest('hex');
const files = {
  'include/onnxruntime_c_api.h': 'official C API header',
  'include/onnxruntime_ep_c_api.h': 'official execution-provider C API header',
  'lib/libonnxruntime.so.1.24.1': 'official CPU core',
  'licenses/ONNXRUNTIME-LICENSE': 'official CPU license',
  'licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt': 'official CPU notices',
};
function put(directory, name, value, mode) {
  const target = path.join(directory, name);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, value, mode ? { mode } : undefined);
  return target;
}
function fixture(t, { local = false, includeCpu = true } = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'deckard-runtime-cache-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const commands = path.join(directory, 'commands');
  const cache = path.join(directory, local ? 'cache/native-build' : 'external');
  for (const name of ['json/include/nlohmann/json.hpp', 'licenses/NLOHMANN-LICENSE',
    'licenses/GRADIENT-MODEL-CARD.md', 'cargo/registry/fixture']) put(cache, name, name);
  for (const [name, body] of Object.entries({
    uname: 'echo x86_64', 'pkg-config': 'exit 0', cmake: 'exit 0', objcopy: 'exit 0', unzip: 'exit 0',
    'c++': 'exit 0', curl: 'echo "Unexpected download" >&2; exit 99',
  })) put(commands, name, `#!/bin/sh\n${body}\n`, 0o755);
  put(cache, 'rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/rustc',
    '#!/bin/sh\necho "rustc 1.90.0 (fixture)"\n', 0o755);
  put(cache, 'rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/cargo', '#!/bin/sh\nexit 0\n', 0o755);
  put(cache, 'cargo/bin/cargo', '#!/bin/sh\nexit 0\n', 0o755);
  // The old iGPU branch owns this namespace and the old common notices.
  put(cache, `${legacy}/include/onnxruntime_c_api.h`, 'GPU header');
  put(cache, `${legacy}/lib/libonnxruntime.so.1.24.1`, 'GPU core');
  put(cache, 'licenses/ONNXRUNTIME-LICENSE', 'GPU license');
  put(cache, 'licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt', 'GPU notices');
  if (includeCpu) for (const [name, value] of Object.entries(files)) put(cache, `${cpu}/${name}`, value);
  put(directory, 'native-cli/onnxruntime-cpu.sha256', Object.entries(files)
    .map(([name, value]) => `${hash(value)}  ${name}\n`).join(''));
  let script = fs.readFileSync(path.join(root, 'native-cli/bootstrap-linux.sh'), 'utf8');
  for (const [pin, name] of [
    ['9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6', 'json/include/nlohmann/json.hpp'],
    ['86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14', 'licenses/NLOHMANN-LICENSE'],
    ['a147869ab24ad59172bcdc7cc69cf716c646ddf0745a0e1fbb12515a2c6e752a', 'licenses/GRADIENT-MODEL-CARD.md'],
  ]) script = script.replace(pin, hash(fs.readFileSync(path.join(cache, name))));
  if (local) {
    const bundle = path.join(directory, 'bundle');
    for (const [name, value] of Object.entries(files)) {
      const member = name === 'licenses/ONNXRUNTIME-LICENSE' ? 'LICENSE'
        : name === 'licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt' ? 'ThirdPartyNotices.txt' : name;
      put(bundle, `${archiveName}/${member}`, value);
    }
    const archive = path.join(cache, 'downloads', `${archiveName}.tgz`);
    fs.mkdirSync(path.dirname(archive), { recursive: true });
    const packed = spawnSync('tar', ['-czf', archive, '-C', bundle, archiveName], { encoding: 'utf8' });
    assert.equal(packed.status, 0, packed.stderr);
    script = script.replace('9142552248b735920f9390027e4512a2cacf8946a1ffcbe9071a5c210531026f',
      hash(fs.readFileSync(archive)));
  }
  const scriptFile = put(directory, 'native-cli/bootstrap-linux.sh', script);
  const run = () => spawnSync('/bin/sh', [scriptFile], { encoding: 'utf8', timeout: 10000,
    env: { ...process.env, PATH: `${commands}:/usr/bin:/bin`, NATIVE_CACHE: cache, CXX: 'c++' } });
  return { directory, cache, run };
}

test('read-only CPU cache is distribution-specific, pinned and never modified', { skip: linuxOnly }, t => {
  const f = fixture(t);
  const before = fs.readdirSync(f.cache, { recursive: true });
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /read-only native dependencies/);
  assert.deepEqual(fs.readdirSync(f.cache, { recursive: true }), before);
  assert.equal(fs.readFileSync(path.join(f.cache, `${legacy}/lib/libonnxruntime.so.1.24.1`), 'utf8'), 'GPU core');
});

test('an old GPU-populated cache cannot masquerade as the official CPU cache', { skip: linuxOnly }, t => {
  const f = fixture(t, { includeCpu: false });
  const result = f.run();
  assert.notEqual(result.status, 0, result.stdout);
  assert.match(result.stderr, /incomplete: onnxruntime-cpu-1\.24\.1/);
});

test('renamed GPU libraries, drifted headers and mismatched notices are refused read-only', { skip: linuxOnly }, t => {
  const f = fixture(t);
  for (const [name, value] of Object.entries(files)) {
    const target = path.join(f.cache, cpu, name);
    fs.writeFileSync(target, 'different distribution');
    const result = f.run();
    assert.notEqual(result.status, 0, name);
    assert.match(result.stderr, /official CPU runtime.*checksum/i, result.stderr);
    assert.equal(fs.readFileSync(target, 'utf8'), 'different distribution', 'read-only cache is left alone');
    fs.writeFileSync(target, value);
  }
});

test('cached bootstrap repairs only the CPU namespace and preserves GPU libraries and notices', { skip: linuxOnly }, t => {
  const f = fixture(t, { local: true, includeCpu: false });
  for (let attempt = 0; attempt < 2; attempt++) {
    const result = f.run();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stderr, '', 'a normal cache miss or repair is not a checksum failure');
    for (const [name, value] of Object.entries(files))
      assert.equal(fs.readFileSync(path.join(f.cache, cpu, name), 'utf8'), value);
    assert.equal(fs.readFileSync(path.join(f.cache, legacy, 'lib/libonnxruntime.so.1.24.1'), 'utf8'), 'GPU core');
    assert.equal(fs.readFileSync(path.join(f.cache, 'licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt'), 'utf8'), 'GPU notices');
    fs.writeFileSync(path.join(f.cache, cpu, 'lib/libonnxruntime.so.1.24.1'), 'drifted CPU core');
  }
});

test('a valid cached CPU distribution is reused without extracting over it', { skip: linuxOnly }, t => {
  const f = fixture(t, { local: true });
  const target = path.join(f.cache, cpu, 'lib/libonnxruntime.so.1.24.1');
  const previous = new Date('2000-01-01T00:00:00Z');
  fs.utimesSync(target, previous, previous);
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.statSync(target).mtimeMs, previous.getTime());
});

test('CMake independently checks CPU pins before embedding a runtime', { skip: linuxOnly }, t => {
  const f = fixture(t);
  const cmake = spawnSync('cmake', ['--version'], { encoding: 'utf8' });
  if (cmake.error?.code === 'ENOENT') return t.skip('requires CMake for configuration validation');
  const run = () => spawnSync('cmake', [`-DONNXRUNTIME=${path.join(f.cache, cpu)}`,
    `-DONNXRUNTIME_PINS=${path.join(f.directory, 'native-cli/onnxruntime-cpu.sha256')}`,
    '-P', path.join(root, 'native-cli/verify-onnxruntime.cmake')], { encoding: 'utf8', timeout: 10000 });
  assert.equal(run().status, 0);
  for (const [name, value] of Object.entries(files)) {
    const target = path.join(f.cache, cpu, name);
    fs.writeFileSync(target, 'different distribution');
    const wrong = run();
    assert.notEqual(wrong.status, 0, name);
    assert.match(wrong.stderr, /official CPU runtime.*checksum/is);
    fs.writeFileSync(target, value);
  }
  fs.writeFileSync(path.join(f.directory, 'native-cli/onnxruntime-cpu.sha256'), '');
  const empty = run();
  assert.notEqual(empty.status, 0, 'an empty pin file must not disable verification');
  assert.match(empty.stderr, /CPU runtime checksum pins are missing/);
  const config = fs.readFileSync(path.join(root, 'native-cli/linux.cmake'), 'utf8');
  assert.match(config, /onnxruntime-cpu-\$\{ONNXRUNTIME_VERSION\}/);
  assert.match(config, /include\([^\n]*verify-onnxruntime\.cmake/);
});

test('packaging takes selected CPU notices even when common notices have identical timestamps', { skip: linuxOnly }, t => {
  const f = fixture(t);
  if (spawnSync('cmake', ['--version']).error?.code === 'ENOENT') return t.skip('requires CMake');
  const destination = path.join(f.directory, 'packaged notices');
  for (const [name, value] of Object.entries(files)) {
    if (!name.startsWith('licenses/')) continue;
    const common = put(f.cache, name, 'x'.repeat(value.length));
    const pinned = path.join(f.cache, cpu, name);
    const date = new Date('2000-01-01T00:00:00Z');
    fs.utimesSync(common, date, date);
    fs.utimesSync(pinned, date, date);
  }
  const rust = path.join(f.directory, 'rust');
  put(rust, 'share/doc/rust/COPYRIGHT-library.html', 'Rust notices');
  put(rust, 'share/doc/rust/licenses/NOTICE', 'Rust license');
  const tokenizer = path.join(f.directory, 'tokenizer');
  put(tokenizer, 'Cargo.lock', 'lock fixture');
  const result = spawnSync('cmake', [`-DNATIVE_CACHE=${f.cache}`, `-DRUST_TOOLCHAIN=${rust}`,
    `-DTOKENIZER_DIR=${tokenizer}`, `-DDESTINATION=${destination}`,
    `-DPLATFORM_NOTICE_DIRECTORY=${path.join(f.cache, cpu, 'licenses')}`,
    '-P', path.join(root, 'native-cli/collect-licenses.cmake')], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  for (const [name, value] of Object.entries(files)) if (name.startsWith('licenses/'))
    assert.equal(fs.readFileSync(path.join(destination, path.basename(name)), 'utf8'), value);
});
