"""Data hazırlığı: mənbələrdən mətn yığır, tokenizer öyrədir, train.npy / val.npy yazır.

    python -m odlar.data configs/odlar15.yaml

Mənbə növləri (config-də data.sources):
  - type: hf          Hugging Face dataseti
      path: wikimedia/wikipedia
      name: 20231101.az
      split: train
      text_field: text
      streaming: false
  - type: files       Öz faylların (.txt, .md, .pdf). Bədii kitablar üçün.
      glob: books/**/*.txt
Hər mənbədə əlavə olaraq:
      max_chars: 300000000   # bu mənbədən ən çox nə qədər simvol
      min_chars: 200         # bundan qısa sənədlər atılır
      repeat: 1              # 2 = bu mənbə təlimdə iki dəfə çox görünür (kiçik, dəyərli data üçün)
      label: vikipediya      # loqlarda görünən ad
Hazır data qovluqda varsa, yenidən hazırlanmır.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np


def _log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- mənbələr
def _iter_hf(src):
    from datasets import load_dataset
    kw = {"split": src.get("split", "train"), "streaming": bool(src.get("streaming", False))}
    if src.get("name"):
        kw["name"] = src["name"]
    ds = load_dataset(src["path"], **kw)
    field = src.get("text_field", "text")
    for row in ds:
        yield row[field]


def _read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("PDF oxumaq üçün: pip install pypdf")
    pages = []
    for page in PdfReader(path).pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _iter_files(src):
    paths = sorted(glob.glob(src["glob"], recursive=True))
    if not paths:
        _log(f"  diqqət: {src['glob']} üzrə fayl tapılmadı")
    for p in paths:
        if p.lower().endswith(".pdf"):
            yield _read_pdf(p)
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                yield f.read()


def collect(src):
    """Bir mənbədən təmizlənmiş sənədlərin siyahısı."""
    it = {"hf": _iter_hf, "files": _iter_files}[src["type"]](src)
    max_chars = src.get("max_chars") or float("inf")
    min_chars = src.get("min_chars", 200)
    docs, total, seen = [], 0, set()
    t0 = time.time()
    for t in it:
        t = (t or "").strip()
        if len(t) < min_chars:
            continue
        h = hashlib.md5(t.encode("utf-8", "replace")).digest()
        if h in seen:                       # eyni sənəd iki dəfə gəlməsin
            continue
        seen.add(h)
        docs.append(t)
        total += len(t)
        if len(docs) % 50_000 == 0:
            _log(f"    {src.get('label', src['type'])}: {total/1e6:.0f}M simvol...")
        if total >= max_chars:
            break
    _log(f"  {src.get('label', src['type'])}: {len(docs):,} sənəd, {total/1e6:.1f}M simvol "
         f"({(time.time()-t0)/60:.1f} dəq)")
    return docs, total


# ---------------------------------------------------------------- tokenizer
def train_tokenizer(corpora, vocab_size, max_chars, path):
    """Mənbələrdən növbə ilə götürür ki, tokenizer hamısını görsün (yalnız birincini yox)."""
    from tokenizers import ByteLevelBPETokenizer

    def it():
        n, iters = 0, [iter(c) for c in corpora]
        while iters and n < max_chars:
            for i in list(iters):
                try:
                    t = next(i)
                except StopIteration:
                    iters.remove(i)
                    continue
                yield t
                n += len(t)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(it(), vocab_size=vocab_size, min_frequency=2,
                            special_tokens=["<|endoftext|>"])
    tok.save(path)


def encode_docs(tok, docs, eot, dtype):
    parts = []
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]):
            parts.append(np.array(e.ids + [eot], dtype=dtype))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


# ---------------------------------------------------------------- əsas
def paths(cfg):
    d = cfg["data"]["dir"]
    return {k: os.path.join(d, f) for k, f in
            (("train", "train.npy"), ("val", "val.npy"), ("tok", "tokenizer.json"), ("meta", "meta.json"))}


def prepare(cfg, force=False):
    from tokenizers import Tokenizer
    dc = cfg["data"]
    P = paths(cfg)
    if not force and all(os.path.exists(P[k]) for k in ("train", "val", "tok")):
        _log(f"Hazır data tapıldı: {dc['dir']}")
        return P
    if not dc["sources"]:
        raise SystemExit("data.sources boşdur: config-də ən azı bir mənbə yaz")
    os.makedirs(dc["dir"], exist_ok=True)
    t0 = time.time()
    _log("1) Mətn yığılır")
    corpora = []
    for src in dc["sources"]:
        docs, chars = collect(src)
        corpora.append((src, docs, chars))

    _log("2) Tokenizer")
    if dc.get("tokenizer"):
        import shutil
        shutil.copy(dc["tokenizer"], P["tok"])
        _log(f"  hazır tokenizer götürüldü: {dc['tokenizer']}")
    else:
        train_tokenizer([c[1] for c in corpora], dc["vocab_size"], dc["tok_train_chars"], P["tok"])
        _log(f"  öyrədildi: {dc['vocab_size']} token ({(time.time()-t0)/60:.1f} dəq)")
    tok = Tokenizer.from_file(P["tok"])
    eot = tok.token_to_id("<|endoftext|>")
    dtype = np.uint16 if tok.get_vocab_size() < 65536 else np.uint32

    _log("3) Tokenlərə çevrilir")
    trains, vals, meta = [], [], {"sources": [], "vocab_size": tok.get_vocab_size()}
    for src, docs, chars in corpora:
        ids = encode_docs(tok, docs, eot, dtype)
        nv = int(len(ids) * dc["val_frac"])
        tr, va = ids[:len(ids) - nv], ids[len(ids) - nv:]
        rep = int(src.get("repeat", 1))
        trains += [tr] * rep
        vals.append(va)
        meta["sources"].append({"label": src.get("label", src["type"]), "docs": len(docs), "chars": chars,
                                "tokens": int(len(ids)), "repeat": rep,
                                "chars_per_token": round(chars / max(1, len(ids)), 3)})
        _log(f"  {src.get('label', src['type'])}: {len(ids)/1e6:.1f}M token (x{rep})")
        del docs
    train = np.concatenate(trains)
    val = np.concatenate(vals)
    np.save(P["train"], train)
    np.save(P["val"], val)
    meta.update(train_tokens=int(len(train)), val_tokens=int(len(val)),
                minutes=round((time.time() - t0) / 60, 1))
    with open(P["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    _log(f"Hazır: təlim {len(train)/1e6:.1f}M token, validasiya {len(val)/1e6:.2f}M token "
         f"({meta['minutes']} dəq)")
    return P


if __name__ == "__main__":
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Odlar data hazırlığı")
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="məs. data.vocab_size=16000")
    ap.add_argument("--force", action="store_true", help="hazır data olsa da yenidən hazırla")
    a = ap.parse_args()
    prepare(load_config(a.config, a.set), force=a.force)
    sys.exit(0)
