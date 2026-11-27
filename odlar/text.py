"""Mətn təmizləmə: normallaşdırma, Vikipediya bölmələri, keyfiyyət filtrləri, şəxsi məlumat, MinHash.

Hamısı saf funksiyalardır: data hazırlığı (odlar.data) və testlər eyni kodu işlədir.
Saytdakı JS tokenizer `normalize`-in eynisini edir (docs/tokenizer.js), ona görə model təlimdə
və brauzerdə eyni yazılışı görür.
"""
import re
import unicodedata
import zlib

import numpy as np

# ---------------------------------------------------------------- normallaşdırma
# Kiril "ә/Ә" latin mətnlərdə tez-tez "ə/Ə" əvəzinə yazılır (məs. rəsmi PDF-lər: "Mәdәniyyәt").
# Dırnaq və apostrof növləri birləşdirilir; yumşaq defis və görünməz simvollar silinir.
_CHAR_MAP = str.maketrans({
    "ә": "ə", "Ә": "Ə",        # ә Ә -> ə Ə
    "‘": "'", "’": "'", "ʼ": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": None, "‌": None, "‍": None, "﻿": None,
})
_SOFT_HYPHEN = re.compile("­\\s*")           # "Gürcüs­ ki" -> "Gürcüski"


def normalize(t):
    t = unicodedata.normalize("NFC", t)
    t = _SOFT_HYPHEN.sub("", t)
    return t.translate(_CHAR_MAP)


# ---------------------------------------------------------------- Vikipediya
# Məqalənin sonundakı siyahı bölmələri: model bunları öyrənəndə uzun mətndə "İstinadlar"a düşür.
WIKI_TAIL = ("İstinadlar", "Həmçinin bax", "Xarici keçidlər", "Ədəbiyyat", "Mənbələr", "Mənbə",
             "Qeydlər", "Əlavə ədəbiyyat", "Həmçinin baxın", "Xarici keçid")
_WIKI_TAIL_RE = re.compile(r"^\s*(?:" + "|".join(map(re.escape, WIKI_TAIL)) + r")\s*:?\s*$", re.M)
_WIKI_CAT_RE = re.compile(r"^\s*Kateqoriya:.*$", re.M)


def clean_wiki(t):
    m = _WIKI_TAIL_RE.search(t)
    if m:
        t = t[:m.start()]
    t = _WIKI_CAT_RE.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ---------------------------------------------------------------- şəxsi məlumat
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Azərbaycan nömrələri: +994 50 123 45 67, (012) 498-12-34, 050 123 45 67, 0501234567.
# Yalnız "+" və ya "0" ilə başlayanlar: əhali sayı kimi böyük ədədlər toxunulmaz qalır.
_PHONE = re.compile(r"(?<![\w+])(?:\+\d{3}[\s-]?\(?\d{2}\)?|\(?0\d{2}\)?)[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\w)")


def mask_pii(t):
    t = _EMAIL.sub("[e-poçt]", t)
    return _PHONE.sub("[telefon]", t)


# ---------------------------------------------------------------- keyfiyyət filtrləri
_AZ_LETTERS = set("abcçdeəfgğhxıijkqlmnoöprsştuüvyzABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ")


def strip_foreign_lines(t, min_share=0.5, min_letters=20):
    """Hərflərinin yarıdan çoxu Azərbaycan əlifbasından olmayan sətirləri silir (rusca, ingiliscə,
    xarab kodlaşma). Bütöv sənəd əvəzinə yalnız həmin sətirlər gedir: qarışıq səhifələrdə
    Azərbaycan hissəsi saxlanır."""
    out = []
    for ln in t.split("\n"):
        letters = [c for c in ln if c.isalpha()]
        if len(letters) >= min_letters and sum(c in _AZ_LETTERS for c in letters) / len(letters) < min_share:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def quality(t, rules):
    """None = keçdi; əks halda atılma səbəbi (statistika üçün).
    rules: {"max_digit": 0.15, "max_dup_lines": 0.20, "word_len": [3, 12], "min_az": 0.80,
            "min_schwa": 0.015}"""
    n = len(t)
    if not n:
        return "boş"
    letters = sum(c.isalpha() for c in t)
    if rules.get("max_digit") is not None and sum(c.isdigit() for c in t) / n > rules["max_digit"]:
        return "rəqəm"
    if rules.get("min_az") is not None and letters:
        az = sum(c in _AZ_LETTERS for c in t)
        if az / letters < rules["min_az"]:
            return "skript"
    if rules.get("min_schwa") is not None and letters >= 400:
        # "ə" Azərbaycan mətnində hərflərin ~7-9%-idir; türk dilində demək olar yoxdur
        if (t.count("ə") + t.count("Ə")) / letters < rules["min_schwa"]:
            return "dil"
    if rules.get("max_dup_lines") is not None:
        # Təkrarlanan sətirlərin simvol payı (Gopher üsulu). Sətir sayı ilə ölçəndə qısa təkrarlar
        # ("-", başlıq) yaxşı məqalələri, stenoqramları və nəqarətli şeirləri atırdı.
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if len(lines) >= 5:
            seen, dup = set(), 0
            for ln in lines:
                if ln in seen:
                    dup += len(ln)
                seen.add(ln)
            if dup / max(1, sum(map(len, lines))) > rules["max_dup_lines"]:
                return "təkrar sətir"
    if rules.get("word_len"):
        words = t.split()
        if words:
            avg = sum(map(len, words)) / len(words)
            lo, hi = rules["word_len"]
            if not lo <= avg <= hi:
                return "söz uzunluğu"
    return None


DEFAULT_RULES = {"max_digit": 0.15, "max_dup_lines": 0.20, "word_len": [3, 12], "min_az": 0.80,
                 "min_schwa": 0.015}


# ---------------------------------------------------------------- MinHash (oxşar dublikat)
N_PERM, BANDS = 128, 16             # 16 band x 8 sətir: Jaccard ~0.7-dən oxşar sənədlər tutulur
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_rng = np.random.default_rng(20261127)
_A = _rng.integers(1, 2**63, N_PERM, dtype=np.uint64) | np.uint64(1)
_B = _rng.integers(0, 2**63, N_PERM, dtype=np.uint64)
_WORD = re.compile(r"\w+")


def _shingles(t, k=5):
    words = _WORD.findall(t.lower())
    if not words:
        return np.zeros(1, dtype=np.uint64)
    h = np.fromiter((zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words))
    if len(h) < k:
        k = len(h)
    with np.errstate(over="ignore"):
        s = np.zeros(len(h) - k + 1, dtype=np.uint64)
        for j in range(k):                               # k sözlük pəncərənin hash-i
            s = s * np.uint64(1000003) + h[j:len(h) - k + 1 + j]
    return np.unique(s)


def minhash_bands(t):
    """Sənədin 16 band hash-i (uint64). İki sənəd hansısa bandda üst-üstə düşürsə, oxşardır."""
    s = _shingles(t)
    with np.errstate(over="ignore"):
        hv = (s[:, None] * _A[None, :] + _B[None, :]) >> np.uint64(16)   # (n, 128)
    sig = hv.min(axis=0)
    rows = N_PERM // BANDS
    sig = sig.reshape(BANDS, rows)
    with np.errstate(over="ignore"):
        out = np.zeros(BANDS, dtype=np.uint64)
        for j in range(rows):
            out = out * np.uint64(0x9E3779B97F4A7C15) + sig[:, j]
    return out


def duplicates(bands):
    """bands: (N, BANDS). Qaytarır: bool massiv, True = əvvəlki sənədlərdən birinin oxşarıdır.
    Sıra önəmlidir: eyni qrupdan ilk gələn saxlanır (keyfiyyətli mənbələr config-də əvvəl yazılır)."""
    N = len(bands)
    dup = np.zeros(N, dtype=bool)
    for b in range(bands.shape[1]):
        _, first, inv = np.unique(bands[:, b], return_index=True, return_inverse=True)
        dup |= first[inv] != np.arange(N)
    return dup
