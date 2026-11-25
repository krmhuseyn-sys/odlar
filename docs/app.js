import { OdlarTokenizer } from "./tokenizer.js";
import { generate } from "./generate.js";

const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

const MODELS = {
  "odlar-1.5": {
    name: "Odlar-1.5", onnx: "odlar15.onnx", tok: "tok15.json", blockSize: 512,
    facts: [["40M", "parametr"], ["345M", "token: Vikipediya + veb mətni"], ["2 saat", "təlim, bir T4 GPU"], ["49 MB", "brauzerdə model ölçüsü"]],
    about: "Azərbaycan Vikipediyası və FineWeb-2 veb mətnləri (xəbər, bloq, sayt) üzərində sıfırdan öyrədilib. Odlar-1-dən 3 dəfə böyükdür, daha uzun və əlaqəli mətn yazır.",
  },
  "odlar-1": {
    name: "Odlar-1", onnx: "odlar1.onnx", tok: "tok.json", blockSize: 256,
    facts: [["13.8M", "parametr"], ["84.6M", "token Vikipediya mətni"], ["22 dəq", "təlim, bir T4 GPU"], ["17 MB", "brauzerdə model ölçüsü"]],
    about: "Yalnız Azərbaycan Vikipediyası üzərində sıfırdan öyrədilib. Ailənin ilk, ən balaca üzvüdür: ensiklopedik üslubda yazır.",
  },
};

const $ = id => document.getElementById(id);
const EXAMPLES = ["Qədim zamanlarda", "Xəzər dənizi", "Azərbaycan mətbəxi", "Qış gecəsi idi", "Kür çayı", "Azərbaycanın paytaxtı"];
const loaded = {};
let current = null, busy = false, stopRequested = false;

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

function showInfo(id) {
  const m = MODELS[id];
  $("aboutTitle").textContent = `${m.name} nədir?`;
  $("aboutText").textContent = m.about;
  $("facts").innerHTML = "";
  for (const [big, small] of m.facts) {
    const d = document.createElement("div");
    const s = document.createElement("strong"); s.textContent = big;
    const sp = document.createElement("span"); sp.textContent = small;
    d.append(s, sp); $("facts").appendChild(d);
  }
  $("brand").textContent = m.name.replace("Odlar", "");
}

async function select(id) {
  current = id; showInfo(id);
  $("go").disabled = true;
  if (loaded[id]) { ready(); return; }
  const m = MODELS[id];
  $("status").classList.remove("ready");
  try {
    const [tokJson, model] = await Promise.all([
      fetch(m.tok).then(r => r.json()),
      fetchWithProgress(m.onnx, (got, total) => {
        const mb = x => (x / 1e6).toFixed(1);
        $("statusText").textContent = total ? `${m.name} yüklənir: ${mb(got)} / ${mb(total)} MB` : `${m.name} yüklənir: ${mb(got)} MB`;
        if (total) $("barFill").style.width = `${(got / total) * 100}%`;
      }),
    ]);
    loaded[id] = {
      tokenizer: new OdlarTokenizer(tokJson),
      session: await ort.InferenceSession.create(model, { executionProviders: ["wasm"] }),
    };
    if (current === id) ready();
  } catch (e) {
    $("statusText").textContent = "Model yüklənmədi: " + e.message + ". Səhifəni yenilə.";
    console.error(e);
  }
}

function ready() {
  $("status").classList.add("ready");
  $("statusText").textContent = `${MODELS[current].name} hazırdır. Sənin brauzerində işləyir.`;
  $("go").disabled = busy;
}

async function run() {
  const st = loaded[current];
  if (busy || !st) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("output").innerHTML = '<span class="hint">Bir neçə söz yaz, Odlar davam etdirsin.</span>'; return; }
  busy = true; stopRequested = false;
  $("go").disabled = true; $("stop").disabled = false; $("model").disabled = true;
  const out = $("output");
  out.textContent = "";
  const head = document.createElement("b"); head.textContent = prompt;
  const tail = document.createElement("span");
  out.append(head, tail);
  const t0 = performance.now(); let n = 0;
  try {
    await generate(ort, st.session, st.tokenizer, prompt, {
      maxNewTokens: Number($("len").value),
      temperature: Number($("temp").value),
      topK: Number($("topk").value),
      repetitionPenalty: Number($("rep").value),
      minNewTokens: 20,
      blockSize: MODELS[current].blockSize,
    }, text => { tail.textContent = text; n++; }, () => stopRequested);
    $("meta").textContent = `${MODELS[current].name} · ${n} token, ${((performance.now() - t0) / 1000).toFixed(1)} saniyə`;
  } catch (e) {
    $("meta").textContent = "Xəta: " + e.message;
    console.error(e);
  }
  busy = false;
  $("go").disabled = false; $("stop").disabled = true; $("model").disabled = false;
}

$("go").onclick = run;
$("stop").onclick = () => { stopRequested = true; };
$("model").addEventListener("change", e => select(e.target.value));
$("prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
select($("model").value);
