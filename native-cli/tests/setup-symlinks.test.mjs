// Manifest paths may use trusted symlinks; every link in a chain is checked.
// These preflight checks need no model or native-host registration.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const binary = process.env.DECKARD_BIN || fileURLToPath(new URL("../build/dist/bin/deckard", import.meta.url));
function fixture(t) {
  const build = fileURLToPath(new URL("../build/", import.meta.url));
  fs.mkdirSync(build, { recursive: true });
  const root = fs.mkdtempSync(path.join(build, "setup-links-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const user = path.join(root, "user");
  fs.mkdirSync(user);
  const run = (command, manifest) => spawnSync(binary, [command, "--home", path.join(user, "deckard"),
    "--manifest-dir", manifest, ...(command === "install" ? ["--no-extension", "--shell", "none",
      "--model-dir", path.join(root, "empty model")] : [])], {
    encoding: "utf8", timeout: 10000, env: { ...process.env, HOME: user },
  });
  return { root, user, run };
}

test("manifest preflight resolves relative symlink chains and a missing suffix", t => {
  const f = fixture(t);
  const real = path.join(f.root, "real profiles");
  fs.mkdirSync(real);
  fs.mkdirSync(path.join(f.root, "links"));
  fs.symlinkSync("../real profiles", path.join(f.root, "links/second"));
  fs.symlinkSync("links/second", path.join(f.root, "first"));
  const result = f.run("uninstall", path.join(f.root, "first/NativeMessagingHosts"));
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!fs.existsSync(path.join(real, "NativeMessagingHosts")));
  assert.ok(fs.lstatSync(path.join(f.root, "first")).isSymbolicLink());
});

test("relative targets resolve symlinks before processing parent components", t => {
  const f = fixture(t);
  fs.mkdirSync(path.join(f.root, "real/nested"), { recursive: true });
  fs.mkdirSync(path.join(f.root, "real/profiles"));
  fs.symlinkSync("real/nested", path.join(f.root, "jump"));
  fs.symlinkSync("jump/../profiles", path.join(f.root, "first"));
  const result = f.run("uninstall", path.join(f.root, "first/NativeMessagingHosts"));
  assert.equal(result.status, 0, result.stderr);
});

test("a symlink loop fails as a setup conflict without hanging", t => {
  const f = fixture(t);
  fs.symlinkSync("second", path.join(f.root, "first"));
  fs.symlinkSync("first", path.join(f.root, "second"));
  const result = f.run("uninstall", path.join(f.root, "first/NativeMessagingHosts"));
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /setup_conflict.*Too many symlinks/);
});

test("dangling targets and non-directory target components are refused", t => {
  const f = fixture(t);
  fs.writeFileSync(path.join(f.root, "ordinary file"), "preserve");
  for (const [name, destination] of [["dangling", "missing"], ["invalid", "ordinary file/../profiles"]]) {
    fs.symlinkSync(destination, path.join(f.root, name));
    const result = f.run("uninstall", path.join(f.root, name, "NativeMessagingHosts"));
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /filesystem error/);
  }
  assert.equal(fs.readFileSync(path.join(f.root, "ordinary file"), "utf8"), "preserve");
});

test("install and uninstall reject an untrusted link reached inside a trusted target", t => {
  // /proc/self is owned by a different, unmapped uid in some user namespaces.
  // On ordinary hosts it is root-owned (trusted), so no foreign-owner fixture
  // can be made without privileges and this ownership-specific case is skipped.
  const indirect = "/proc/self";
  let info;
  try { info = fs.lstatSync(indirect); } catch { return t.skip("needs a foreign-owned /proc/self symlink"); }
  if (!info.isSymbolicLink() || info.uid === 0 || info.uid === process.getuid())
    return t.skip("needs a foreign-owned /proc/self symlink");
  const f = fixture(t);
  const real = path.join(f.root, "profiles");
  fs.mkdirSync(real);
  const first = path.join(f.root, "first");
  fs.symlinkSync(path.join(indirect, "root", real), first);
  assert.equal(fs.lstatSync(first).uid, process.getuid());
  for (const command of ["install", "uninstall"]) {
    const result = f.run(command, path.join(first, "NativeMessagingHosts"));
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /setup_conflict.*symlinks owned by you or root/);
    assert.ok(result.stderr.includes(indirect), result.stderr);
    assert.ok(!fs.existsSync(path.join(real, "NativeMessagingHosts")));
  }
});
