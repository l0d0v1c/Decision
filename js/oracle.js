// Laya oracle in the browser: request validation, the exact input laya builds for one `choice`
// question, and the forward pass. Shared by index.html and tests/golden_web.mjs.

export class OracleError extends Error {}

// Python's json.dumps(..., ensure_ascii=False): same escapes as JSON.stringify, but ": " and ", "
// as separators. The state is a single-key object, so only the key separator matters.
function serializeState(text) {
  return '{"question": ' + JSON.stringify(text) + "}";
}

export function normalizeRequest(cfg, { question, context, options }) {
  question = (question || "").trim();
  context = (context || "").trim();
  options = (options || [])
    .map((o) => (typeof o === "string" ? { label: o } : o))
    .map((o) => ({ label: (o.label || "").trim(), meaning: (o.meaning || "").trim() }))
    .filter((o) => o.label);
  const L = cfg.limits;
  if (!question) throw new OracleError("La question est vide.");
  if (question.length > L.question) throw new OracleError(`La question dépasse ${L.question} caractères.`);
  if (context.length > L.context) throw new OracleError(`Le contexte dépasse ${L.context} caractères.`);
  if (options.length < 2 || options.length > L.options) throw new OracleError(`Il faut entre 2 et ${L.options} cas.`);
  if (options.some((o) => o.label.length > L.option)) throw new OracleError(`Un cas dépasse ${L.option} caractères.`);
  if (options.some((o) => o.meaning.length > L.meaning)) throw new OracleError(`Une signification dépasse ${L.meaning} caractères.`);
  if (new Set(options.map((o) => o.label.toLowerCase())).size !== options.length) {
    throw new OracleError("Deux cas portent le même nom : chacun doit être différent.");
  }
  return { question, context, options };
}

// Mirrors laya.common.build_head / build_sequence with the oracle's request (src/oracle/core.py):
// neutral keys A, B, ... whose criterion is "label: meaning", context above the question.
export function buildInput(tok, cfg, request) {
  const { question, context, options } = normalizeRequest(cfg, request);
  const mask = cfg.mask_token;
  const clean = (s) => s.split(mask).join(" ");
  const encode = (s) => tok.encode(s, { add_special_tokens: false }).ids;

  const stateText = context ? `${context}\n\n${question}` : question;
  const stateIds = encode(clean(serializeState(stateText)));

  const keys = options.map((_, i) => String.fromCharCode(65 + i));
  const rendered = options.map((o, i) => `${keys[i]}: ${o.meaning ? `${o.label}: ${o.meaning}` : o.label}`);
  let headIds = encode(`choice question: ${clean(cfg.instructions)}`);
  let optIds = rendered.map((r) => [cfg.mask_id, ...encode(" " + clean(r)).slice(0, cfg.option_max_tokens)]);
  const total = () => optIds.reduce((n, o) => n + o.length, 0);
  let budget = cfg.head_max_len - total();
  if (budget < 16) {
    const per = Math.max(4, Math.floor((cfg.head_max_len - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    budget = cfg.head_max_len - total();
  }
  headIds = headIds.slice(0, Math.max(8, budget));

  let ids = [cfg.cls_id, ...headIds, cfg.sep_id];
  let markers = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids.push(...o);
  }
  ids.push(cfg.sep_id);
  const room = Math.max(0, cfg.max_len - ids.length - 1);
  const truncated = stateIds.length > room;
  ids = [...ids, ...stateIds.slice(0, room), cfg.sep_id].slice(0, cfg.max_len);
  markers = markers.filter((m) => m < cfg.max_len);
  if (markers.length !== options.length) throw new OracleError("Trop de cas pour la taille du modèle : retire-en un.");
  return { ids, markers, labels: options.map((o) => o.label), truncated };
}

export async function decide(ort, session, cfg, input) {
  const T = input.ids.length, K = input.markers.length;
  const i64 = (arr) => BigInt64Array.from(arr, (v) => BigInt(v));
  const feeds = {
    input_ids: new ort.Tensor("int64", i64(input.ids), [1, T]),
    attention_mask: new ort.Tensor("int64", i64(new Array(T).fill(1)), [1, T]),
    marker_pos: new ort.Tensor("int64", i64(input.markers), [1, K]),
    marker_mask: new ort.Tensor("bool", new Uint8Array(K).fill(1), [1, K]),
    qtype: new ort.Tensor("int64", i64([cfg.qtype_choice]), [1]),
  };
  const { logits } = await session.run(feeds);
  const z = Array.from(logits.data);
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  const ranking = input.labels
    .map((answer, i) => ({ answer, probability: e[i] / sum }))
    .sort((a, b) => b.probability - a.probability);
  return { answer: ranking[0].answer, ranking, truncated: input.truncated };
}

// Fetch the parts listed in MANIFEST.json, check each sha256, and return the whole graph.
// `fetchPart(file)` returns an ArrayBuffer; `onProgress(loadedBytes, totalBytes)` is optional.
export async function loadModelBytes(manifest, fetchPart, sha256, onProgress) {
  const out = new Uint8Array(manifest.bytes);
  let offset = 0;
  for (const part of manifest.parts) {
    const buf = new Uint8Array(await fetchPart(part.file));
    if (buf.length !== part.bytes || (await sha256(buf)) !== part.sha256) {
      throw new OracleError(`Fichier du modèle corrompu : ${part.file}. Recharge la page.`);
    }
    out.set(buf, offset);
    offset += buf.length;
    if (onProgress) onProgress(offset, manifest.bytes);
  }
  return out;
}
