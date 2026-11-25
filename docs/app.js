import { OdlarTokenizer } from "./tokenizer.js";
import { generate } from "./generate.js";

const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

const $ = id => document.getElementById(id);
const EXAMPLES = ["Xəzər dənizi", "Qədim zamanlarda", "Azərbaycan mətbəxi", "Futbol üzrə", "Azərbaycanın paytaxtı", "Bakı şəhəri"];
let session = null, tokenizer = null, busy = false, stopRequested = false;

for (const [input, out] of [["len", "lenV"], ["temp", "tempV"], ["topk", "topkV"], ["rep", "repV"]]) {
  $(input).addEventListener("input", () => { $(out).textContent = $(input).value; });
}
for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = ex;
  b.onclick = () => { $("prompt").value = ex; run(); };
  $("chips").appendChild(b);
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(got, total);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function load() {
  try {
    const [tokJson, model] = await Promise.all([
      fetch("tok.json").then(r => r.json()),
      fetchWithProgress("odlar1.onnx", (got, total) => {
        const mb = x => (x / 1e6).toFixed(1);
        $("statusText").textContent = total ? `Model yüklənir: ${mb(got)} / ${mb(total)} MB` : `Model yüklənir: ${mb(got)} MB`;
        if (total) $("barFill").style.width = `${(got / total) * 100}%`;
      }),
    ]);
    tokenizer = new OdlarTokenizer(tokJson);
    session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    $("status").classList.add("ready");
    $("statusText").textContent = "Hazırdır. Odlar sənin brauzerində işləyir.";
    $("go").disabled = false;
  } catch (e) {
    $("statusText").textContent = "Model yüklənmədi: " + e.message + ". Səhifəni yenilə.";
    console.error(e);
  }
}

async function run() {
  if (busy || !session) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("output").innerHTML = '<span class="hint">Bir neçə söz yaz, Odlar davam etdirsin.</span>'; return; }
  busy = true; stopRequested = false;
  $("go").disabled = true; $("stop").disabled = false;
  const out = $("output");
  out.textContent = "";
  const head = document.createElement("b"); head.textContent = prompt;
  const tail = document.createElement("span");
  out.append(head, tail);
  const t0 = performance.now(); let n = 0;
  try {
    await generate(ort, session, tokenizer, prompt, {
      maxNewTokens: Number($("len").value),
      temperature: Number($("temp").value),
      topK: Number($("topk").value),
      repetitionPenalty: Number($("rep").value),
      minNewTokens: 20,
    }, text => { tail.textContent = text; n++; }, () => stopRequested);
    const s = (performance.now() - t0) / 1000;
    $("meta").textContent = `${n} token, ${s.toFixed(1)} saniyə`;
  } catch (e) {
    $("meta").textContent = "Xəta: " + e.message;
    console.error(e);
  }
  busy = false;
  $("go").disabled = false; $("stop").disabled = true;
}

$("go").onclick = run;
$("stop").onclick = () => { stopRequested = true; };
$("prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
load();
