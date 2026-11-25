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
export async function generate(ort, session, tokenizer, prompt, opts, onToken, shouldStop = () => false) {
  const block = opts.blockSize ?? 256;
  const ids = tokenizer.encode(prompt);
  const start = ids.length;
  for (let step = 0; step < opts.maxNewTokens; step++) {
    if (shouldStop()) break;
    const ctx = ids.slice(-block);
    const input = new ort.Tensor("int64", BigInt64Array.from(ctx, BigInt), [1, ctx.length]);
    const out = await session.run({ input_ids: input });
    const ban = step < (opts.minNewTokens ?? 20) ? tokenizer.eot : -1;
    const next = pickNext(out.logits.data, ctx, opts, ban);
    if (next === tokenizer.eot) break;
    ids.push(next);
    onToken(tokenizer.decode(ids.slice(start)));
    await breathe();
  }
  return tokenizer.decode(ids);
}
