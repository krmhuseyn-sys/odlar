import { OdlarTokenizer } from "./tokenizer.js";
import { generate } from "./generate.js";
import { MODELS, DEFAULT_MODEL } from "./models.js";

const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

const $ = id => document.getElementById(id);
const byId = id => MODELS.find(m => m.id === id) || MODELS[0];
const STARTERS = ["Qış gecəsi idi", "Qədim zamanlarda", "Xəzər dənizi", "Azərbaycan mətbəxi", "Kür çayı", "Kəndimizin kənarında"];
const KEY = { chats: "odlar.chats.v1", model: "odlar.model", settings: "odlar.settings", theme: "odlar.theme" };
const MAX_CHATS = 60;

// ---------------------------------------------------------------- yaddaş (brauzerdə)
const store = {
  get(k, fallback) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* yaddaş dolu və ya bağlıdır */ } },
};

let chats = store.get(KEY.chats, []).filter(c => c && Array.isArray(c.turns));
let activeId = null;
let modelId = byId(store.get(KEY.model, DEFAULT_MODEL)).id;
const settings = Object.assign({ len: 120, temp: 0.8, rep: 1.3, topk: 40 }, store.get(KEY.settings, {}));

const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const saveChats = () => store.set(KEY.chats, chats.slice(0, MAX_CHATS));
const activeChat = () => chats.find(c => c.id === activeId) || null;

// ---------------------------------------------------------------- modelin yüklənməsi
const loaded = {};
const loading = {};

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} tapılmadı (${res.status})`);
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

function loadModel(id) {
  if (loaded[id]) return Promise.resolve(loaded[id]);
  if (loading[id]) return loading[id];
  const m = byId(id);
  showLoad(`${m.name} yüklənir`, 0);
  loading[id] = (async () => {
    const [tokJson, bytes] = await Promise.all([
      fetch(m.tok).then(r => { if (!r.ok) throw new Error(`${m.tok} tapılmadı`); return r.json(); }),
      fetchWithProgress(m.onnx, (got, total) => {
        const mb = x => (x / 1e6).toFixed(0);
        if (id === modelId) showLoad(total ? `${m.name} yüklənir: ${mb(got)} / ${mb(total)} MB` : `${m.name} yüklənir: ${mb(got)} MB`,
          total ? got / total : 0);
      }),
    ]);
    if (id === modelId) showLoad(`${m.name} hazırlanır`, 1);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    loaded[id] = { tokenizer: new OdlarTokenizer(tokJson), session };
    return loaded[id];
  })();
  loading[id].then(() => { if (id === modelId) hideLoad(); updateSend(); },
    e => {
      delete loading[id];
      console.error(e);
      if (id === modelId) showLoad(`${m.name} yüklənmədi: ${e.message}. Səhifəni yenilə.`, 0, true);
    });
  return loading[id];
}

function showLoad(text, frac, failed = false) {
  $("load").classList.remove("done");
  $("load").classList.toggle("failed", failed);
  $("loadText").textContent = text;
  $("loadFill").style.width = `${Math.round(frac * 100)}%`;
}
function hideLoad() {
  $("load").classList.add("done");
  $("loadText").textContent = `${byId(modelId).name} hazırdır`;
}

// ---------------------------------------------------------------- model seçimi
function renderPicker() {
  $("pickerName").textContent = byId(modelId).name;
  const menu = $("pickerMenu");
  menu.innerHTML = "";
  for (const m of MODELS) {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.dataset.id = m.id;
    li.setAttribute("aria-selected", String(m.id === modelId));
    li.innerHTML = `<b></b><small></small><svg class="tick" viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 10.5l3.5 3.5 7.5-8"/></svg>`;
    li.querySelector("b").textContent = m.name;
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = loaded[m.id] ? "yüklənib" : m.size;
    li.querySelector("b").appendChild(size);
    li.querySelector("small").textContent = m.summary;
    li.addEventListener("click", () => { chooseModel(m.id); closePicker(); });
    menu.appendChild(li);
  }
}
function chooseModel(id) {
  if (busy || id === modelId) return;
  modelId = id;
  store.set(KEY.model, id);
  renderPicker();
  if (loaded[id]) hideLoad();
  loadModel(id);
  updateSend();
}
let pickerFocus = -1;
function openPicker() {
  renderPicker();
  $("pickerMenu").hidden = false;
  $("pickerBtn").setAttribute("aria-expanded", "true");
  pickerFocus = MODELS.findIndex(m => m.id === modelId);
  markPickerFocus();
}
function closePicker() {
  $("pickerMenu").hidden = true;
  $("pickerBtn").setAttribute("aria-expanded", "false");
}
function markPickerFocus() {
  [...$("pickerMenu").children].forEach((li, i) => li.classList.toggle("focus", i === pickerFocus));
}
$("pickerBtn").addEventListener("click", () => ($("pickerMenu").hidden ? openPicker() : closePicker()));
$("pickerBtn").addEventListener("keydown", e => {
  const menu = $("pickerMenu");
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (menu.hidden) openPicker();
    pickerFocus = (pickerFocus + (e.key === "ArrowDown" ? 1 : -1) + MODELS.length) % MODELS.length;
    markPickerFocus();
  } else if ((e.key === "Enter" || e.key === " ") && !menu.hidden && pickerFocus >= 0) {
    e.preventDefault();
    chooseModel(MODELS[pickerFocus].id);
    closePicker();
  } else if (e.key === "Escape") closePicker();
});
document.addEventListener("click", e => { if (!e.target.closest(".picker")) closePicker(); });

// ---------------------------------------------------------------- tarixçə
function renderHistory() {
  const nav = $("history");
  nav.innerHTML = "";
  if (!chats.length) {
    const p = document.createElement("p");
    p.className = "history-empty";
    p.textContent = "Yazdığın mətnlər burada saxlanacaq.";
    nav.appendChild(p);
    return;
  }
  const h = document.createElement("h2");
  h.textContent = "Əvvəlki mətnlər";
  nav.appendChild(h);
  for (const c of chats) {
    const row = document.createElement("div");
    row.className = "h-item" + (c.id === activeId ? " active" : "");
    const a = document.createElement("a");
    a.href = `#${c.id}`;
    a.textContent = c.title || "Yeni mətn";
    if (c.id === activeId) a.setAttribute("aria-current", "page");
    const del = document.createElement("button");
    del.className = "h-del";
    del.setAttribute("aria-label", `Sil: ${c.title}`);
    del.innerHTML = `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg>`;
    del.addEventListener("click", () => {
      if (busy && c.id === activeId) return;
      if (!del.classList.contains("armed")) {      // iki addımlı silmə: əvvəl təsdiq
        del.classList.add("armed");
        del.textContent = "Sil";
        del.style.opacity = 1; del.style.fontSize = "13px"; del.style.color = "var(--nar)";
        setTimeout(() => { if (del.isConnected) renderHistory(); }, 3000);
        return;
      }
      chats = chats.filter(x => x.id !== c.id);
      saveChats();
      if (c.id === activeId) openChat(null); else renderHistory();
    });
    row.append(a, del);
    nav.appendChild(row);
  }
}

function openChat(id, { push = true } = {}) {
  if (busy) return;
  activeId = id && chats.some(c => c.id === id) ? id : null;
  if (push) history.replaceState(null, "", activeId ? `#${activeId}` : location.pathname + location.search);
  renderHistory();
  renderTurns();
  closeSide();
  $("prompt").focus({ preventScroll: true });
}
window.addEventListener("hashchange", () => openChat(location.hash.slice(1), { push: false }));

// ---------------------------------------------------------------- mətnlərin göstərilməsi
const turnEls = new Map();

function renderTurns() {
  const box = $("turns");
  box.innerHTML = "";
  turnEls.clear();
  const chat = activeChat();
  document.body.classList.toggle("has-turns", !!(chat && chat.turns.length));
  if (!chat) return;
  for (const t of chat.turns) box.appendChild(turnEl(t));
  $("feed").scrollTop = $("feed").scrollHeight;
}

function icon(path) {
  return `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="${path}"/></svg>`;
}

function turnEl(t) {
  const art = document.createElement("article");
  art.className = "turn";
  const text = document.createElement("p");
  text.className = "text";
  const p = document.createElement("span"); p.className = "p"; p.textContent = t.prompt;
  const seam = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  seam.setAttribute("class", "buta seam");
  seam.setAttribute("aria-hidden", "true");
  seam.innerHTML = `<use href="#buta"/>`;
  const c = document.createElement("span"); c.className = "c"; c.textContent = t.output;
  text.append(p, seam, c);

  const foot = document.createElement("div");
  foot.className = "turn-foot";
  const who = document.createElement("span"); who.className = "who"; who.textContent = byId(t.model).name;
  const meta = document.createElement("span");
  const actions = document.createElement("div"); actions.className = "turn-actions";
  const mk = (label, path, fn) => {
    const b = document.createElement("button");
    b.className = "act"; b.type = "button";
    b.innerHTML = icon(path) + `<span>${label}</span>`;
    b.addEventListener("click", fn);
    actions.appendChild(b);
    return b;
  };
  const bCont = mk("Davam etdir", "M4 10h10M10 5l5 5-5 5", () => runTurn(t, "continue"));
  const bAgain = mk("Yenidən yaz", "M4.5 9A5.5 5.5 0 0115 6.5M15.5 3.5v3h-3M15.5 11A5.5 5.5 0 015 13.5M4.5 16.5v-3h3", () => runTurn(t, "again"));
  const bCopy = mk("Kopyala", "M7 7h8.5v9.5H7zM4.5 13V4h8.5", async () => {
    try {
      await navigator.clipboard.writeText(t.prompt + t.output);
      bCopy.querySelector("span").textContent = "Kopyalandı";
    } catch { bCopy.querySelector("span").textContent = "Kopyalanmadı"; }
    setTimeout(() => { bCopy.querySelector("span").textContent = "Kopyala"; }, 1600);
  });
  foot.append(who, meta, actions);
  art.append(text, foot);
  const el = { art, text, c, meta, who, buttons: [bCont, bAgain, bCopy] };
  turnEls.set(t.id, el);
  updateTurnMeta(t);
  return art;
}

function updateTurnMeta(t, streaming = false) {
  const el = turnEls.get(t.id);
  if (!el) return;
  el.who.textContent = byId(t.model).name;
  el.meta.textContent = streaming ? "yazır" : t.error ? "" : `${t.tokens} token, ${(t.ms / 1000).toFixed(1)} saniyə`;
  for (const b of el.buttons) b.disabled = streaming || busy;
}

// ---------------------------------------------------------------- yazdırmaq
let busy = false, stopRequested = false;

function updateSend() {
  const send = $("send");
  send.classList.toggle("stop", busy);
  send.setAttribute("aria-label", busy ? "Dayandır" : "Yaz");
  send.disabled = busy ? false : !loaded[modelId] || !$("prompt").value.trim();
  $("pickerBtn").disabled = busy;
  for (const [, el] of turnEls) for (const b of el.buttons) b.disabled = busy;
}

function nearBottom() {
  const f = $("feed");
  return f.scrollHeight - f.scrollTop - f.clientHeight < 120;
}

async function runTurn(turn, mode) {
  if (busy) return;
  const st = loaded[modelId];
  if (!st) { loadModel(modelId); return; }
  busy = true; stopRequested = false;
  const prevOutput = turn.output;
  if (mode === "again") turn.output = "";
  turn.model = modelId;
  turn.error = null;
  const el = turnEls.get(turn.id);
  const caret = document.createElement("span");
  caret.className = "caret";
  el.text.appendChild(caret);
  el.c.textContent = turn.output;
  updateSend();
  updateTurnMeta(turn, true);
  if (mode !== "continue") el.art.scrollIntoView({ block: "nearest", behavior: "smooth" });

  const base = turn.output;
  const t0 = performance.now(); let n = 0;
  try {
    await generate(ort, st.session, st.tokenizer, turn.prompt + base, {
      maxNewTokens: settings.len, temperature: settings.temp, topK: settings.topk,
      repetitionPenalty: settings.rep, minNewTokens: mode === "continue" ? 5 : 20,
      blockSize: byId(modelId).blockSize,
    }, text => {
      const stick = nearBottom();
      turn.output = base + text; n++;
      el.c.textContent = turn.output;
      if (stick) $("feed").scrollTop = $("feed").scrollHeight;
    }, () => stopRequested);
    turn.tokens = (mode === "continue" ? turn.tokens || 0 : 0) + n;
    turn.ms = (mode === "continue" ? turn.ms || 0 : 0) + (performance.now() - t0);
  } catch (e) {
    console.error(e);
    if (mode === "again" && !turn.output) turn.output = prevOutput;
    turn.error = e.message;
    el.c.textContent = turn.output;
    const err = document.createElement("span");
    err.className = "err";
    err.textContent = `Yazmaq alınmadı: ${e.message}`;
    el.text.appendChild(err);
  }
  caret.remove();
  busy = false;
  saveChats();
  updateTurnMeta(turn);
  updateSend();
  $("live").textContent = `${byId(turn.model).name} yazdı: ${n} token`;
}

async function submit() {
  if (busy) { stopRequested = true; return; }
  const prompt = $("prompt").value.trim();
  if (!prompt || !loaded[modelId]) return;
  let chat = activeChat();
  if (!chat) {
    chat = { id: uid(), title: prompt.slice(0, 48), created: Date.now(), turns: [] };
    chats.unshift(chat);
    activeId = chat.id;
    history.replaceState(null, "", `#${chat.id}`);
  } else {
    chats = [chat, ...chats.filter(c => c !== chat)];   // son istifadə olunan yuxarıda
  }
  const turn = { id: uid(), prompt, output: "", model: modelId, tokens: 0, ms: 0 };
  chat.turns.push(turn);
  $("prompt").value = "";
  autosize();
  renderHistory();
  document.body.classList.add("has-turns");
  $("turns").appendChild(turnEl(turn));
  await runTurn(turn, "new");
}

// ---------------------------------------------------------------- yazı sahəsi
function autosize() {
  const ta = $("prompt");
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, window.innerHeight * 0.3) + "px";
}
$("prompt").addEventListener("input", () => { autosize(); updateSend(); });
$("prompt").addEventListener("keydown", e => {
  const coarse = matchMedia("(pointer: coarse)").matches;   // telefonda Enter yeni sətirdir
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && (!coarse || e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submit();
  }
});
$("composer").addEventListener("submit", e => { e.preventDefault(); submit(); });

for (const s of STARTERS) {
  const b = document.createElement("button");
  b.className = "starter"; b.type = "button"; b.textContent = s;
  b.addEventListener("click", () => {
    $("prompt").value = s;
    autosize();
    updateSend();
    if (loaded[modelId]) submit(); else $("prompt").focus();
  });
  $("starters").appendChild(b);
}

// ayarlar
for (const k of ["len", "temp", "rep", "topk"]) {
  const input = $(k), out = $(k + "V");
  input.value = settings[k];
  out.textContent = settings[k];
  input.addEventListener("input", () => {
    settings[k] = Number(input.value);
    out.textContent = input.value;
    store.set(KEY.settings, settings);
  });
}
$("settingsBtn").addEventListener("click", () => {
  const open = $("settings").hidden;
  $("settings").hidden = !open;
  $("settingsBtn").setAttribute("aria-expanded", String(open));
});

// ---------------------------------------------------------------- panel, tema, haqqında
function closeSide() { $("app").classList.remove("side-open"); $("scrim").hidden = true; }
$("openSide").addEventListener("click", () => { $("app").classList.add("side-open"); $("scrim").hidden = false; });
$("closeSide").addEventListener("click", closeSide);
$("scrim").addEventListener("click", closeSide);
$("newChat").addEventListener("click", () => openChat(null));

$("themeBtn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set(KEY.theme, root.dataset.theme);
});
try { const t = JSON.parse(localStorage.getItem(KEY.theme)); if (t) document.documentElement.dataset.theme = t; } catch { /* boş */ }

function renderAbout() {
  const box = $("aboutModels");
  box.innerHTML = "";
  for (const m of MODELS) {
    const d = document.createElement("div");
    d.className = "m-card";
    const h = document.createElement("h3"); h.textContent = m.name;
    const p = document.createElement("p"); p.textContent = m.about;
    const f = document.createElement("div"); f.className = "m-facts";
    for (const [big, small] of m.facts) {
      const x = document.createElement("div");
      const s = document.createElement("strong"); s.textContent = big;
      x.append(s, document.createTextNode(small));
      f.appendChild(x);
    }
    d.append(h, p, f);
    box.appendChild(d);
  }
}
$("openAbout").addEventListener("click", () => { renderAbout(); closeSide(); $("about").showModal(); });

// ---------------------------------------------------------------- başlanğıc
renderPicker();
openChat(location.hash.slice(1), { push: false });
autosize();
updateSend();
loadModel(modelId);
