// Növbəti tokenin seçilməsi: təkrar cəzası, temperatur, top-k, sonra təsadüfi seçim.
// Python-dakı model.generate ilə eyni məntiq.

export function pickNext(logits, context, { temperature = 0.8, topK = 40, repetitionPenalty = 1.3 }, ban = -1, rand = Math.random) {
  const v = Float32Array.from(logits);
  if (ban >= 0) v[ban] = -Infinity;
  if (repetitionPenalty !== 1) {
    for (const id of new Set(context)) {
      v[id] = v[id] > 0 ? v[id] / repetitionPenalty : v[id] * repetitionPenalty;
    }
  }
  const t = Math.max(temperature, 1e-5);
  const order = Array.from(v.keys()).sort((a, b) => v[b] - v[a]).slice(0, Math.max(1, topK));
  const max = v[order[0]];
  const w = order.map(i => Math.exp((v[i] - max) / t));
  const sum = w.reduce((a, b) => a + b, 0);
  let r = rand() * sum;
  for (let k = 0; k < order.length; k++) {
    r -= w[k];
    if (r <= 0) return order[k];
  }
  return order[order.length - 1];
}

// Brauzerə hər tokendən sonra ekranı yeniləmək və düymələrə cavab vermək imkanı verir.
const breathe = () => new Promise(r => setTimeout(r, 0));

// Modeli addım-addım işlədir; hər yeni token üçün onToken çağırılır.
// İki model formatı var:
//   keşli (Odlar-2-dən): girişlər input_ids + past, çıxışlar logits + present. Hər addımda yalnız
//     yeni token hesablanır; opts.kv = { layers, heads, headDim } lazımdır (models.js).
//   keşsiz (Odlar-1, 1.5): hər addımda bütün kontekst yenidən hesablanır.
export async function generate(ort, session, tokenizer, prompt, opts, onToken, shouldStop = () => false) {
  const block = opts.blockSize ?? 256;
  const ids = tokenizer.encode(prompt);
  if (!ids.length) ids.push(tokenizer.eot);
  const start = ids.length;
  const cached = session.inputNames.includes("past");
  const kv = opts.kv;
  if (cached && !kv) throw new Error("keşli model üçün models.js-də kv ölçüləri yazılmayıb");
  const emptyPast = () => new ort.Tensor("float32", new Float32Array(0), [kv.layers, 2, 1, kv.heads, 0, kv.headDim]);
  const tensor = a => new ort.Tensor("int64", BigInt64Array.from(a, BigInt), [1, a.length]);
  let past = null, pastLen = 0;
  for (let step = 0; step < opts.maxNewTokens; step++) {
    if (shouldStop()) break;
    let out;
    if (cached) {
      let feed;
      if (pastLen > 0 && pastLen < block) {
        feed = [ids[ids.length - 1]];                      // yalnız son token
      } else {
        // ilk addım və ya kontekst doldu: son hissəni keşsiz yenidən hesabla
        feed = ids.slice(-(pastLen > 0 ? Math.floor(block * 3 / 4) : block));
        past = emptyPast();
      }
      out = await session.run({ input_ids: tensor(feed), past });
      past = out.present;
      pastLen = past.dims[4];
    } else {
      out = await session.run({ input_ids: tensor(ids.slice(-block)) });
    }
    const ban = step < (opts.minNewTokens ?? 20) ? tokenizer.eot : -1;
    const next = pickNext(out.logits.data, ids.slice(-block), opts, ban);
    if (next === tokenizer.eot) break;
    ids.push(next);
    onToken(tokenizer.decode(ids.slice(start)));
    await breathe();
  }
  return tokenizer.decode(ids);
}
