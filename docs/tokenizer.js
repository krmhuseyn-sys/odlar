// Odlar tokenizer: byte-level BPE (GPT-2 üslubu), Python `tokenizers` kitabxanası ilə eyni nəticə verir.
// tok.json: { vocab: [id -> token], merges: ["a b", ...], digits?: true }
// digits: true olanda (Odlar-2) hər rəqəm ayrıca token olur, Python-dakı Digits(individual_digits) kimi.

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const enc = new Map(), dec = new Map();
  bs.forEach((b, i) => { const c = String.fromCharCode(cs[i]); enc.set(b, c); dec.set(c, b); });
  return { enc, dec };
}

const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const DIGIT = /(\p{N})/u;

// odlar/text.py normalize() ilə eyni: model təlimdə gördüyü yazılışı burada da görsün.
const CHAR_MAP = new Map([
  ["\u04d9", "\u0259"], ["\u04d8", "\u018f"],
  ["\u2018", "'"], ["\u2019", "'"], ["\u02bc", "'"], ["\u0060", "'"], ["\u00b4", "'"],
  ["\u201c", '"'], ["\u201d", '"'], ["\u201e", '"'], ["\u00ab", '"'], ["\u00bb", '"'],
  ["\u00a0", " "], ["\u2009", " "], ["\u202f", " "], ["\u2007", " "],
  ["\u200b", ""], ["\u200c", ""], ["\u200d", ""], ["\ufeff", ""],
]);
export function normalize(t) {
  t = t.normalize("NFC").replace(/\u00ad\s*/g, "");
  return Array.from(t, c => CHAR_MAP.has(c) ? CHAR_MAP.get(c) : c).join("");
}

export class OdlarTokenizer {
  constructor({ vocab, merges, digits = false }) {
    this.digits = digits;
    this.vocab = vocab;
    this.ids = new Map(vocab.map((t, i) => [t, i]));
    this.ranks = new Map(merges.map((m, i) => [m, i]));
    const { enc, dec } = bytesToUnicode();
    this.byteEnc = enc; this.byteDec = dec;
    this.utf8 = new TextEncoder();
    this.cache = new Map();
    this.eot = this.ids.get("<|endoftext|>");
  }

  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    this.cache.set(word, parts);
    return parts;
  }

  pieces(text) {
    if (!this.digits) return [text];
    return text.split(DIGIT).filter(p => p.length);
  }

  encode(text) {
    const out = [];
    text = normalize(text);
    for (const piece of this.pieces(text)) {
      for (const m of piece.matchAll(PRETOKEN)) {
        const mapped = Array.from(this.utf8.encode(m[0]), b => this.byteEnc.get(b)).join("");
        for (const p of this.bpe(mapped)) {
          const id = this.ids.get(p);
          if (id !== undefined) out.push(id);
        }
      }
    }
    return out;
  }

  decode(ids) {
    const chars = ids.filter(i => i !== this.eot).map(i => this.vocab[i]).join("");
    const bytes = Uint8Array.from(Array.from(chars), c => this.byteDec.get(c) ?? 32);
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
}
