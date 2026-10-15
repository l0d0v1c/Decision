// Golden harness for the browser oracle, run in Node with the same packages the page loads.
//   npm install && npm test        (cwd: repository root)
// 1. JS tokenization + sequence building must equal laya's Python output token for token.
// 2. The int8 ONNX graph on onnxruntime-web (WASM) must give the PyTorch reference answer,
//    with every probability within PROB_TOL.
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web";
import { buildInput, decide, loadModelBytes } from "../js/oracle.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PROB_TOL = 0.05;
// Near-tie in the fp16 original (Oui 0.38 / Non 0.35) that also flips with PyTorch int8.
const KNOWN_FLIPS = new Set(["Dois-je investir toutes mes économies"]);

const json = async (p) => JSON.parse(await readFile(join(ROOT, p), "utf8"));
const cfg = await json("model/config.json");
const tok = new Tokenizer(await json("model/tokenizer/tokenizer.json"), await json("model/tokenizer/tokenizer_config.json"));
const refs = await json("tests/golden_reference.json");
const pyInputs = await json("tests/golden_inputs.json");

let tokFail = 0;
const inputs = refs.map((c, i) => {
  const input = buildInput(tok, cfg, c);
  const same = JSON.stringify(input.ids) === JSON.stringify(pyInputs[i].input_ids)
    && JSON.stringify(input.markers) === JSON.stringify(pyInputs[i].marker_pos);
  if (!same) {
    tokFail++;
    const at = input.ids.findIndex((v, j) => v !== pyInputs[i].input_ids[j]);
    console.log(`TOKENS FAIL case ${i} at position ${at}: js=${input.ids.slice(at, at + 5)} py=${pyInputs[i].input_ids.slice(at, at + 5)}`);
  }
  return input;
});
console.log(`tokenization: ${refs.length - tokFail}/${refs.length} identical to Python`);

const manifest = await json("model/MANIFEST.json");
let t = Date.now();
const bytes = await loadModelBytes(
  manifest,
  async (f) => (await readFile(join(ROOT, "model", f))).buffer,
  async (buf) => createHash("sha256").update(buf).digest("hex"),
);
const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
console.log(`model: ${(manifest.bytes / 1e6).toFixed(1)} MB, loaded in ${Date.now() - t} ms (ort ${ort.env.versions.web})`);

let fail = 0, known = 0, worst = 0;
for (const [i, c] of refs.entries()) {
  t = Date.now();
  const r = await decide(ort, session, cfg, inputs[i]);
  const ms = Date.now() - t;
  const diff = Math.max(...r.ranking.map((x) => Math.abs(x.probability - c.probabilities[x.answer])));
  worst = Math.max(worst, diff);
  const ok = r.answer === c.expected && diff <= PROB_TOL;
  const isKnown = !ok && [...KNOWN_FLIPS].some((k) => c.question.startsWith(k));
  if (!ok && !isKnown) fail++;
  if (isKnown) known++;
  console.log(`${ok ? "PASS" : isKnown ? "KNOWN" : "FAIL"} ${String(ms).padStart(5)} ms dP=${diff.toFixed(4)} ${JSON.stringify(r.answer).padEnd(30)} <- ${c.question.slice(0, 50)}`);
}
console.log(`web int8: ${refs.length - fail - known}/${refs.length} pass, ${known} known flip, ${fail} fail, max |dP| = ${worst.toFixed(4)}`);
process.exit(tokFail || fail ? 1 : 0);
