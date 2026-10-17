// The opt-in Intel GPU path (OpenVINO) and deckard check-system on Linux. The
// assertions hold with or without an Intel GPU: they compare deckard's choices
// with what check-system reports for this host.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { assets, modelAvailable } from "./model-fixture.mjs";

const linux = process.platform === "linux";
const binary = process.env.DECKARD_BIN || fileURLToPath(new URL("../build/dist/bin/deckard", import.meta.url));
const model = process.env.DECKARD_MODEL_DIR;
const onnxModel = !!model && fs.existsSync(path.join(model, "model.onnx"));
const run = (args, env = {}) => spawnSync(binary, args, { encoding: "utf8", timeout: 600000,
  env: { ...process.env, ...env } });

function scratch(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "deckard-gpu-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

function checkSystem(t, env = {}) {
  const result = run(["check-system"], { HOME: scratch(t), ...env });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

test("check-system reports models, the Intel GPU chain and every accelerator", { skip: !linux && "Linux only" }, t => {
  const report = checkSystem(t);
  assert.equal(report.command, "check-system");
  assert.ok(["onnx", "candle"].includes(report.runtime));
  for (const key of ["present", "opencl_loader", "opencl_driver", "usable"])
    assert.equal(typeof report.intel_gpu[key], "boolean", key);
  assert.ok(Array.isArray(report.intel_gpu.devices));
  assert.deepEqual(report.accelerators.map(a => a.name), ["openvino-gpu"]);
  assert.equal(report.models.candle, "missing", "an empty HOME has no model cache");
  assert.equal(report.problems.length, report.help.length);
});

test("without an Intel OpenCL driver the GPU and OpenVINO are unusable", { skip: !linux && "Linux only" }, t => {
  const empty = scratch(t);
  const report = checkSystem(t, { OCL_ICD_VENDORS: empty });
  assert.equal(report.intel_gpu.opencl_driver, false);
  assert.equal(report.intel_gpu.usable, false);
  assert.equal(report.accelerators[0].usable, false);
  if (report.intel_gpu.present)
    assert.ok(report.problems.includes("Intel GPU present but no Intel OpenCL driver is installed"));
});

test("--gpu needs ONNX Runtime", { skip: !linux && "Linux only" }, () => {
  const result = run(["scan", "--gpu", "--candle", "--file", "-"]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /--gpu needs ONNX Runtime/);
});

test("install --gpu without model.onnx names the export script",
  { skip: (!linux && "Linux only") || (!modelAvailable(model) && "needs DECKARD_MODEL_DIR") }, t => {
    const home = scratch(t);
    const models = path.join(home, "models");
    fs.mkdirSync(models);
    for (const name of Object.keys(assets.files)) fs.linkSync(path.join(model, name), path.join(models, name));
    const result = run(["install", "--home", path.join(home, "deckard"), "--manifest-dir", path.join(home, "m"),
      "--shell", "none", "--no-extension", "--model-dir", models, "--gpu"], { HOME: home });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /export it first \(scripts\/export-onnx\.sh\)/);
  });

test("verify --gpu uses the GPU exactly when check-system says it can",
  { skip: (!linux && "Linux only") || (!onnxModel && "needs DECKARD_MODEL_DIR with model.onnx") }, t => {
    const root = scratch(t);
    const directory = path.join(root, "model");
    fs.mkdirSync(directory);
    for (const name of [...Object.keys(assets.files), "model.onnx"])
      fs.linkSync(path.join(model, name), path.join(directory, name));
    const fixtures = path.join(root, "fixtures.json");
    fs.writeFileSync(fixtures, JSON.stringify({ cases: [{ name: "len3", logit: 0,
      feed: { input_ids: [[1, 5, 2]], attention_mask: [[1, 1, 1]] } }] }));
    const gpu = checkSystem(t).accelerators[0].usable;
    const verify = flags => {
      const result = run(["verify", "--model", directory, "--fixtures", fixtures, "--cache-dir", path.join(root, "cache"),
        ...flags]);
      return JSON.parse(result.stdout);
    };
    assert.equal(verify(["--onnx", "--gpu"]).device, gpu ? "gpu" : "cpu");
    assert.equal(verify(["--onnx"]).device, "cpu", "the GPU is opt-in");
  });
