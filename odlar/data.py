"""Data hazırlığı: mənbələrdən mətn yığır, təmizləyir, dublikatları atır, tokenizer öyrədir, tokenlərə çevirir.

    python -m odlar.data configs/odlar2.yaml

Dörd mərhələ. Hər biri diskə yazılır və bitibsə təkrarlanmır, ona görə sessiya kəsilsə,
yenidən işə salanda qaldığı mərhələdən davam edir:

  1) yığım      mənbə -> normallaşdırma -> təmizləmə -> filtrlər -> text/<ad>/*.jsonl.gz
                (mətn RAM-da yığılmır, hissə-hissə diskə yazılır)
  2) dublikat   bütün mənbələr üzrə MinHash; config-də əvvəl yazılan mənbə üstünlük alır
  3) tokenizer  çəkili nümunədən byte-level BPE, rəqəmlər ayrı-ayrı, Simurq tokenləri ehtiyatda
  4) tokenlər   <ad>.train.bin, <ad>.val.bin (uint16), meta.json

Mənbə növləri (config-də data.sources):
  - type: hf          Hugging Face dataseti (streaming: true böyük datasetlər üçün)
      path: HuggingFaceFW/fineweb-2
      name: azj_Latn
      text_field: text
      url_field: url              # domen bölgüsü üçün
      meta_field: source          # (istəyə bağlı) routes.meta_regex və drop_meta_regex bu sütuna baxır
  - type: hf_sentences  Cümlə-cümlə saxlanmış kitablar: ardıcıl cümlələr kitab daxilində parçalara yığılır
      path: LocalDoc/books_dataset
      text_field: " Sentence"
      group_field: " Metadata"    # kitab məlumatı; routes.meta_regex buna baxır (məs. kateqoriya)
      doc_chars: 20000
  - type: files       Öz faylların (.txt, .md, .pdf, .epub). Bədii kitablar üçün.
      glob: books/**/*.txt
Hər mənbədə əlavə olaraq:
      label: vikipediya   # fayl və loq adı
      max_chars: 300000000
      min_chars: 200
      repeat: 1           # təlimdə neçə nüsxə (2 = iki dəfə; 1.5 da olar)
      filter: true        # keyfiyyət filtrləri (odlar.text.quality); false və ya qaydalar lüğəti
      wiki: false         # Vikipediya sonluq bölmələrini kəs (path-da "wikipedia" varsa avtomatik)
      pii: true           # e-poçt və telefonları maskala
      dedup: true         # MinHash dublikat yoxlamasına qatılsın
      tok_weight: 1       # tokenizer nümunəsində əlavə çəki (bədii, Vikipediya üçün > 1)
      keep_frac: 1.0      # sənədlərin bu payını bərabər seç (hash ilə); routes-da da ayrıca verilə bilər
      cyrillic: true      # Azərbaycan kirilindəki sənədləri latına çevir (K10)
      drop_meta_regex: "^(books|wikipedia)$"   # meta_field bu ifadəyə uyğundursa, sənəd atılır
      routes:             # ayrı səbətlərə yönəlt; ilk uyğun gələn seçilir
        - label: xeber    # domenə görə (xəbər saytları, payı max_chars ilə məhdud)
          domains: [oxu.az, report.az, apa.az]
          max_chars: 400000000
          repeat: 1
        - label: veb_uzun # uzunluğa görə (K9: uzun sənədlər əlavə çəki alır)
          min_chars: 8000
          repeat: 2
Köhnə formatda (train.npy/val.npy) hazırlanmış data da oxunur.
"""
import glob
import gzip
import hashlib
import itertools
import json
import os
import re
import sys
import time
import zlib
from multiprocessing import Pool
from urllib.parse import urlparse

import numpy as np

from . import text as T

SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"] + \
                 [f"<|reserved_{i}|>" for i in range(10)]
SHARD_DOCS = 20_000


def _log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- mənbələr
def _iter_hf(src):
    """(mətn, url, meta). meta_field verilibsə (məs. AzTC-də "source"), routes/drop onunla seçir."""
    from datasets import load_dataset
    kw = {"split": src.get("split", "train"), "streaming": bool(src.get("streaming", False))}
    if src.get("name"):
        kw["name"] = src["name"]
    ds = load_dataset(src["path"], **kw)
    field, ufield, mfield = src.get("text_field", "text"), src.get("url_field", "url"), src.get("meta_field")
    for row in ds:
        yield row[field], row.get(ufield), (row.get(mfield) if mfield else None)


def _iter_hf_sentences(src):
    """Cümlə-cümlə saxlanmış kitablar (məs. LocalDoc/books_dataset): ardıcıl cümlələr eyni
    group_field (kitab məlumatı) daxilində ~doc_chars simvolluq parçalara yığılır."""
    from datasets import load_dataset
    kw = {"split": src.get("split", "train"), "streaming": bool(src.get("streaming", True))}
    ds = load_dataset(src["path"], **kw)
    tf, gf, size = src.get("text_field", "text"), src["group_field"], int(src.get("doc_chars", 20000))
    cur, buf, n = None, [], 0
    for row in ds:
        g, t = row.get(gf), (row.get(tf) or "").strip()
        if g != cur or n >= size:
            if buf:
                yield " ".join(buf), None, cur
            cur, buf, n = g, [], 0
        if t:
            buf.append(t)
            n += len(t) + 1
    if buf:
        yield " ".join(buf), None, cur


def _read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("PDF oxumaq üçün: pip install pypdf")
    t = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    # PDF-də hər sətir ayrıca qırılır: abzas daxilindəki sətir sonlarını birləşdir
    return re.sub(r"(?<![.!?:;\"»])\n(?!\n)", " ", t)


def _read_epub(path):
    """EPUB: kitabın oxunma sırası (spine) ilə HTML fayllarından mətn."""
    import zipfile
    import html
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        cont = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in cont.iter() if e.tag.endswith("rootfile"))
        opf = ET.fromstring(z.read(opf_path))
        base = os.path.dirname(opf_path)
        items = {e.get("id"): e.get("href") for e in opf.iter() if e.tag.endswith("item")}
        order = [items[e.get("idref")] for e in opf.iter() if e.tag.endswith("itemref") and e.get("idref") in items]
        parts = []
        for href in order:
            name = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
            try:
                h = z.read(name).decode("utf-8", "replace")
            except KeyError:
                continue
            h = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", h)
            h = re.sub(r"(?i)</(p|div|h\d|li|br)\s*>|<br\s*/?>", "\n\n", h)
            parts.append(html.unescape(re.sub(r"<[^>]+>", "", h)))
    return re.sub(r"\n\s*\n\s*", "\n\n", "\n".join(parts))


def _iter_files(src):
    paths = sorted(glob.glob(src["glob"], recursive=True))
    if not paths:
        _log(f"  diqqət: {src['glob']} üzrə fayl tapılmadı")
    for p in paths:
        low = p.lower()
        try:
            if low.endswith(".pdf"):
                t = _read_pdf(p)
            elif low.endswith(".epub"):
                t = _read_epub(p)
            else:
                with open(p, encoding="utf-8", errors="replace") as f:
                    t = f.read()
        except Exception as e:
            _log(f"  oxunmadı: {p} ({e})")
            continue
        yield re.sub(r"(\w)-\n(\w)", r"\1\2", t), None, p     # sətir sonu defisi: "Gürcüs-\nki"


# ---------------------------------------------------------------- 1) yığım
_OPT = {}


def _init_worker(opt):
    _OPT.clear()
    _OPT.update(opt)


def _route(o, t, url, meta):
    """Səbət və seçmə payı: domen (xəbər saytları), meta (kitab kateqoriyası, AzTC mənbəsi),
    uzunluq (K9). İlk uyğun gələn marşrut seçilir; heç biri uyğun deyilsə, əsas səbət."""
    host = ""
    if url and o["routes"]:
        host = (urlparse(url).hostname or "").lower()
        host = host[4:] if host.startswith("www.") else host
    for r in o["routes"]:
        if r.get("domains") and not any(host == d or host.endswith("." + d) for d in r["domains"]):
            continue
        if r.get("meta_regex") and not re.search(r["meta_regex"], str(meta or "")):
            continue
        if r.get("min_chars") and len(t) < r["min_chars"]:
            continue
        return r["label"], r.get("keep_frac", 1.0)
    return o["label"], o["keep_frac"]


def _process(item):
    """Bir sənəd: (çıxış adı və ya None, mətn və ya atılma səbəbi, band-lar)."""
    t, url, meta = item
    o = _OPT
    if o["drop_meta"] and meta is not None and re.search(o["drop_meta"], str(meta)):
        return None, "mənbə xaric", None
    t = (t or "").strip()
    out, keep = _route(o, t, url, meta)
    # Bərabər seçmə (təmizləmədən əvvəl, vaxta qənaət): mətnin hash-i ilə, təkrarlana bilən;
    # max_chars yalnız ehtiyat həddidir, ona görə seçim faylın əvvəlinə meyilli olmur
    if keep < 1.0 and zlib.crc32(t[:2000].encode("utf-8", "replace")) / 2**32 >= keep:
        return None, "seçmə", None
    if o["cyrillic"] and T.is_az_cyrillic(t):     # K10: kiril dövrü nəşrləri latına çevrilir
        t = T.cyr2lat(t)
    t = T.fix_homoglyphs(T.normalize(t))
    if o["wiki"]:
        t = T.clean_wiki(t)
    if o["rules"] and o["rules"].get("min_az") is not None:
        t = T.strip_foreign_lines(t)
    if len(t) < o["min_chars"]:
        return None, "qısa", None
    if o["rules"]:
        why = T.quality(t, o["rules"])
        if why:
            return None, why, None
    if o["pii"]:
        t = T.mask_pii(t)
    return out, t, (T.minhash_bands(t) if o["dedup"] else None)


def outputs_of(src):
    """Bir mənbənin yaratdığı çıxışlar: əsas + routes."""
    main = {"label": src.get("label", src["type"]), "repeat": src.get("repeat", 1),
            "max_chars": src.get("max_chars"), "tok_weight": src.get("tok_weight", 1),
            "dedup": src.get("dedup", True)}
    outs = [main]
    for r in src.get("routes") or []:
        outs.append({"label": r["label"], "repeat": r.get("repeat", 1), "max_chars": r.get("max_chars"),
                     "tok_weight": r.get("tok_weight", 1), "dedup": src.get("dedup", True)})
    return outs


class _ShardWriter:
    def __init__(self, d):
        self.d, self.n, self.f, self.in_shard = d, 0, None, 0
        self.bands = []
        os.makedirs(d, exist_ok=True)

    def write(self, t, bands):
        if self.f is None or self.in_shard >= SHARD_DOCS:
            if self.f:
                self.f.close()
            self.f = gzip.open(os.path.join(self.d, f"part_{self.n // SHARD_DOCS:05d}.jsonl.gz"), "wt",
                               encoding="utf-8", compresslevel=3)
            self.in_shard = 0
        self.f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        self.bands.append(bands if bands is not None else np.zeros(T.BANDS, dtype=np.uint64))
        self.n += 1
        self.in_shard += 1

    def close(self):
        if self.f:
            self.f.close()
        np.save(os.path.join(self.d, "bands.npy"),
                np.stack(self.bands) if self.bands else np.zeros((0, T.BANDS), dtype=np.uint64))


def collect(src, text_dir, workers):
    outs = outputs_of(src)
    done = os.path.join(text_dir, outs[0]["label"], "done.json")
    if os.path.exists(done):
        _log(f"  {outs[0]['label']}: artıq yığılıb")
        return
    for o in outs:                                   # yarımçıq qalmış yığımı sıfırla
        d = os.path.join(text_dir, o["label"])
        for f in glob.glob(os.path.join(d, "*")):
            os.remove(f)
    rules = src.get("filter", True)
    rules = T.DEFAULT_RULES if rules is True else (rules or None)
    if isinstance(rules, dict) and rules is not T.DEFAULT_RULES:
        rules = {**T.DEFAULT_RULES, **rules}
    opt = {"label": outs[0]["label"], "min_chars": src.get("min_chars", 200), "rules": rules,
           "wiki": src.get("wiki", "wikipedia" in str(src.get("path", ""))),
           "pii": src.get("pii", True), "dedup": src.get("dedup", True),
           "cyrillic": src.get("cyrillic", True), "drop_meta": src.get("drop_meta_regex"),
           "keep_frac": float(src.get("keep_frac", 1.0)),
           "routes": src.get("routes") or []}
    writers = {o["label"]: _ShardWriter(os.path.join(text_dir, o["label"])) for o in outs}
    caps = {o["label"]: (o["max_chars"] or float("inf")) for o in outs}
    chars = {o["label"]: 0 for o in outs}
    dropped, seen, t0 = {}, 0, time.time()
    it = {"hf": _iter_hf, "hf_sentences": _iter_hf_sentences, "files": _iter_files}[src["type"]](src)
    main = outs[0]["label"]
    finite = [k for k in caps if caps[k] < float("inf")]
    step = 1000 * workers
    with Pool(workers, initializer=_init_worker, initargs=(opt,)) as pool:
        while True:
            chunk = list(itertools.islice(it, step))      # hissə-hissə: RAM-da yalnız bir hissə olur
            if not chunk:
                break
            for out, t, bands in pool.map(_process, chunk, chunksize=64):
                seen += 1
                if out is None:
                    dropped[t] = dropped.get(t, 0) + 1
                elif chars[out] >= caps[out]:
                    dropped["limit"] = dropped.get("limit", 0) + 1
                else:
                    writers[out].write(t, bands)
                    chars[out] += len(t)
            _log(f"    {main}: {seen:,} sənəd oxundu, " +
                 ", ".join(f"{k} {v/1e6:.0f}M simvol" for k, v in chars.items()))
            if main in finite and all(chars[k] >= caps[k] for k in finite):
                break
    stats = {}
    for o in outs:
        writers[o["label"]].close()
        stats[o["label"]] = {"docs": writers[o["label"]].n, "chars": chars[o["label"]], **o}
    for o in outs:
        with open(os.path.join(text_dir, o["label"], "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats[o["label"]], f, ensure_ascii=False, indent=1)
    with open(done, "w", encoding="utf-8") as f:
        json.dump({"seen": seen, "dropped": dropped, "outputs": stats,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, ensure_ascii=False, indent=1)
    _log(f"  {main}: {seen:,} sənəd oxundu ({(time.time()-t0)/60:.1f} dəq); " +
         ", ".join(f"{k}: {s['docs']:,} sənəd, {s['chars']/1e6:.1f}M simvol" for k, s in stats.items()) +
         ("; atıldı: " + ", ".join(f"{k} {v:,}" for k, v in sorted(dropped.items(), key=lambda x: -x[1]))
          if dropped else ""))


def _read_docs(d):
    for p in sorted(glob.glob(os.path.join(d, "part_*.jsonl.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


# ---------------------------------------------------------------- 2) dublikat
def dedup(labels, text_dir):
    marker = os.path.join(text_dir, "dedup.json")
    if os.path.exists(marker):
        _log("  dublikat yoxlaması artıq edilib")
        return
    parts, use = [], []
    for lab in labels:
        b = np.load(os.path.join(text_dir, lab, "bands.npy"))
        st = json.load(open(os.path.join(text_dir, lab, "stats.json"), encoding="utf-8"))
        parts.append(b)
        use.append(np.full(len(b), st.get("dedup", True)))
    allb, use = np.concatenate(parts), np.concatenate(use)
    idx = np.flatnonzero(use)
    dup = np.zeros(len(allb), dtype=bool)
    dup[idx] = T.duplicates(allb[idx])
    res, o = {}, 0
    for lab, b in zip(labels, parts):
        d = dup[o:o + len(b)]
        np.save(os.path.join(text_dir, lab, "drop.npy"), d)
        res[lab] = {"docs": int(len(b)), "duplicates": int(d.sum())}
        o += len(b)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    _log("  " + ", ".join(f"{k}: {v['duplicates']:,}/{v['docs']:,} dublikat" for k, v in res.items()))


def _kept_docs(text_dir, lab):
    drop_p = os.path.join(text_dir, lab, "drop.npy")
    drop = np.load(drop_p) if os.path.exists(drop_p) else None
    for i, t in enumerate(_read_docs(os.path.join(text_dir, lab))):
        if drop is None or not drop[i]:
            yield t


# ---------------------------------------------------------------- 3) tokenizer
def build_tokenizer(vocab_size, digits=True):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    pts = [pre_tokenizers.ByteLevel(add_prefix_space=False)]
    if digits:                                    # hər rəqəm ayrıca token (Llama üsulu)
        pts.insert(0, pre_tokenizers.Digits(individual_digits=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pts)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    return tok, trainer


def train_tokenizer(text_dir, outs, vocab_size, max_chars, path, digits=True):
    """Hər çıxışdan çəkisinə görə nümunə: simvol x repeat x tok_weight."""
    w = {o["label"]: o["chars"] * float(o["repeat"]) * float(o.get("tok_weight", 1)) for o in outs}
    tot = sum(w.values()) or 1
    quota = {k: max_chars * v / tot for k, v in w.items()}
    _log("  nümunə: " + ", ".join(f"{k} {q/1e6:.0f}M simvol" for k, q in quota.items()))

    def it():
        gens = {k: _kept_docs(text_dir, k) for k in quota}
        used = {k: 0 for k in quota}
        while gens:
            for k in list(gens):
                try:
                    t = next(gens[k])
                except StopIteration:
                    del gens[k]
                    continue
                yield t
                used[k] += len(t)
                if used[k] >= quota[k]:
                    del gens[k]
    tok, trainer = build_tokenizer(vocab_size, digits)
    tok.train_from_iterator(it(), trainer=trainer)
    tok.save(path)


# ---------------------------------------------------------------- 4) tokenlər
def _doc_is_val(t, frac):
    h = int.from_bytes(hashlib.md5(t.encode("utf-8", "replace")).digest()[:4], "little")
    return h / 2**32 < frac


def tokenize_output(tok, text_dir, data_dir, lab, val_frac, val_max, dtype):
    eot = tok.token_to_id("<|endoftext|>")
    trp, vap = os.path.join(data_dir, f"{lab}.train.bin"), os.path.join(data_dir, f"{lab}.val.bin")
    n_tr = n_va = 0
    with open(trp + ".tmp", "wb") as ftr, open(vap + ".tmp", "wb") as fva:
        batch = []

        def flush():
            nonlocal n_tr, n_va
            for t, e in zip(batch, tok.encode_batch(batch)):
                a = np.array(e.ids + [eot], dtype=dtype)
                if n_va < val_max and _doc_is_val(t, val_frac):
                    fva.write(a.tobytes()); n_va += len(a)
                else:
                    ftr.write(a.tobytes()); n_tr += len(a)
            batch.clear()
        for t in _kept_docs(text_dir, lab):
            batch.append(t)
            if len(batch) >= 2000:
                flush()
        flush()
    os.replace(trp + ".tmp", trp)
    os.replace(vap + ".tmp", vap)
    return n_tr, n_va


# ---------------------------------------------------------------- əsas
def paths(cfg):
    d = cfg["data"]["dir"]
    return {"dir": d, "train": os.path.join(d, "train.npy"), "val": os.path.join(d, "val.npy"),
            "tok": os.path.join(d, "tokenizer.json"), "meta": os.path.join(d, "meta.json")}


def _ready(P):
    if os.path.exists(P["meta"]) and os.path.exists(P["tok"]):
        with open(P["meta"], encoding="utf-8") as f:
            m = json.load(f)
        if m.get("format") == 2:
            return bool(m.get("complete"))
    return all(os.path.exists(P[k]) for k in ("train", "val", "tok"))      # köhnə format


def prepare(cfg, force=False):
    from tokenizers import Tokenizer
    dc = cfg["data"]
    P = paths(cfg)
    if not force and _ready(P):
        _log(f"Hazır data tapıldı: {dc['dir']}")
        return P
    if not force and dc.get("hub_dataset") and pull_data(dc["hub_dataset"], dc["dir"]) and _ready(P):
        _log(f"Hazır data Hub-dan endirildi: {dc['hub_dataset']}")
        return P
    if not dc["sources"]:
        raise SystemExit("data.sources boşdur: config-də ən azı bir mənbə yaz")
    os.makedirs(dc["dir"], exist_ok=True)
    text_dir = os.path.join(dc["dir"], "text")
    workers = int(dc.get("workers") or os.cpu_count() or 1)
    t0 = time.time()

    _log(f"1) Mətn yığılır və təmizlənir ({workers} proses)")
    labels = []
    for src in dc["sources"]:
        collect(src, text_dir, workers)
        labels += [o["label"] for o in outputs_of(src)]
    outs = [json.load(open(os.path.join(text_dir, l, "stats.json"), encoding="utf-8")) for l in labels]

    _log("2) Oxşar dublikatlar (MinHash)")
    dedup(labels, text_dir)
    dups = json.load(open(os.path.join(text_dir, "dedup.json"), encoding="utf-8"))

    _log("3) Tokenizer")
    if dc.get("tokenizer"):
        import shutil
        shutil.copy(dc["tokenizer"], P["tok"])
        _log(f"  hazır tokenizer götürüldü: {dc['tokenizer']}")
    elif not os.path.exists(P["tok"]):
        train_tokenizer(text_dir, outs, dc["vocab_size"], dc["tok_train_chars"], P["tok"],
                        digits=dc.get("split_digits", True))
        _log(f"  öyrədildi: {dc['vocab_size']} token ({(time.time()-t0)/60:.1f} dəq)")
    else:
        _log("  artıq var")
    tok = Tokenizer.from_file(P["tok"])
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32

    _log("4) Tokenlərə çevrilir")
    meta = {"format": 2, "vocab_size": tok.get_vocab_size(), "dtype": np.dtype(dtype).name, "sources": []}
    old = {}
    if os.path.exists(P["meta"]):
        with open(P["meta"], encoding="utf-8") as f:
            old = {s["label"]: s for s in json.load(f).get("sources", [])}
    val_max = int(dc.get("val_max_tokens", 2_000_000))
    for o in outs:
        lab = o["label"]
        if lab in old and os.path.exists(os.path.join(dc["dir"], f"{lab}.train.bin")):
            meta["sources"].append(old[lab])
            _log(f"  {lab}: artıq var")
            continue
        n_tr, n_va = tokenize_output(tok, text_dir, dc["dir"], lab, dc["val_frac"], val_max, dtype)
        kept_chars = o["chars"]
        meta["sources"].append({"label": lab, "repeat": o["repeat"], "train": f"{lab}.train.bin",
                                "val": f"{lab}.val.bin", "train_tokens": n_tr, "val_tokens": n_va,
                                "docs": o["docs"], "duplicates": dups.get(lab, {}).get("duplicates", 0),
                                "chars_per_token": round(kept_chars / max(1, n_tr + n_va), 3)})
        _log(f"  {lab}: {n_tr/1e6:.1f}M token təlim, {n_va/1e6:.2f}M validasiya (x{o['repeat']})")
        with open(P["meta"], "w", encoding="utf-8") as f:      # hər mənbədən sonra: kəsilsə itməsin
            json.dump({**meta, "complete": False}, f, ensure_ascii=False, indent=1)
    eff = sum(s["train_tokens"] * float(s["repeat"]) for s in meta["sources"])
    meta.update(complete=True, epoch_tokens=int(eff), minutes=round((time.time() - t0) / 60, 1))
    with open(P["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    _log(f"Hazır: bir epoxa {eff/1e6:.1f}M token (təkrarlarla), {meta['minutes']} dəq")
    if dc.get("hub_dataset"):
        push_data(dc["hub_dataset"], dc["dir"])
    return P


# ---------------------------------------------------------------- Hub dataset
DATA_FILES = ["meta.json", "tokenizer.json", "*.bin"]


def pull_data(repo, data_dir):
    """Hazır tokenləri gizli HF dataset-dən endirir (təlim sessiyası datanı yenidən hazırlamır)."""
    from .hubutil import hf_token
    token = hf_token()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=data_dir, allow_patterns=DATA_FILES,
                          token=token)
        return True
    except Exception as e:
        _log(f"  Hub-dan data endirilmədi ({type(e).__name__}); lokal hazırlanacaq")
        return False


def push_data(repo, data_dir):
    from .hubutil import hf_token
    token = hf_token()
    if not token:
        _log("  diqqət: HF_TOKEN yoxdur, data Hub-a yüklənmədi")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=data_dir, allow_patterns=DATA_FILES,
                      commit_message="data")
    api.super_squash_history(repo, repo_type="dataset", commit_message="data")   # K7: köhnə versiyalar yer tutmasın
    _log(f"  data Hub-a yükləndi: {repo} (gizli)")


def load_parts(P):
    """Təlim hissələri (seçici üçün) və mənbə üzrə validasiya massivləri."""
    if os.path.exists(P["meta"]):
        with open(P["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") == 2:
            if not meta.get("complete", True):
                raise SystemExit("data yarımçıqdır: python -m odlar.data <config> ilə tamamla")
            dt = np.dtype(meta.get("dtype", "uint16"))
            parts, vals = [], {}
            for s in meta["sources"]:
                tr = os.path.join(P["dir"], s["train"])
                if s["train_tokens"]:
                    parts.append({"arr": np.memmap(tr, dtype=dt, mode="r"), "repeat": s["repeat"],
                                  "label": s["label"]})
                if s.get("val_tokens"):
                    vals[s["label"]] = np.memmap(os.path.join(P["dir"], s["val"]), dtype=dt, mode="r")
            return parts, vals
    return ([{"arr": np.load(P["train"], mmap_mode="r"), "repeat": 1, "label": "hamısı"}],
            {"hamısı": np.load(P["val"], mmap_mode="r")})


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
