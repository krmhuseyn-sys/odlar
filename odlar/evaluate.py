"""Standart testlər: hər yeni model eyni ölçü ilə yoxlanılır (məqalə üçün).

    python -m odlar.evaluate runs/odlar15
    python -m odlar.evaluate runs/odlar15 runs/odlar2 --out results/muqayise.json

Testlər (data eval/ qovluğundadır):
  bpc       simvol başına bit, 4 üslub üzrə (lüğətdən asılı deyil, az = yaxşı)
  grammar   40 minimal cüt: düzgün cümləyə səhvdən yüksək ehtimal verirmi (50% = təsadüf)
  capitals  20 ölkənin paytaxtı: düzgün cavab ilk yerdədirmi
  fertility söz başına token (az = tokenizer dili yaxşı tanıyır)
"""
import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F

from .generate import load

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
CAPITALS = [("Azərbaycanın", "Bakı"), ("Türkiyənin", "Ankara"), ("Gürcüstanın", "Tiflis"),
            ("Rusiyanın", "Moskva"), ("İranın", "Tehran"), ("Fransanın", "Paris"),
            ("Almaniyanın", "Berlin"), ("İtaliyanın", "Roma"), ("İspaniyanın", "Madrid"),
            ("Yaponiyanın", "Tokio"), ("Çinin", "Pekin"), ("Misirin", "Qahirə"),
            ("Ukraynanın", "Kiyev"), ("Qazaxıstanın", "Astana"), ("Özbəkistanın", "Daşkənd"),
            ("Böyük Britaniyanın", "London"), ("Yunanıstanın", "Afina"), ("Polşanın", "Varşava"),
            ("Pakistanın", "İslamabad"), ("Macarıstanın", "Budapeşt")]


def _json(name):
    with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def nll(model, ids):
    """Mətnin mənfi log-ehtimalı (nat). Uzun mətn kontekstdən böyükdürsə, hissə-hissə."""
    T = model.config.block_size
    total = 0.0
    for s in range(0, len(ids) - 1, T):
        chunk = ids[s:s + T + 1]
        logits, _ = model(torch.tensor([chunk[:-1]]))
        total += F.cross_entropy(logits[0], torch.tensor(chunk[1:]), reduction="sum").item()
    return total


def bpc(model, tok):
    eot = tok.token_to_id("<|endoftext|>")
    res, a, c = {}, 0.0, 0
    for style, texts in _json("eval_texts.json").items():
        sa = sc = 0
        for t in texts:
            sa += nll(model, [eot] + tok.encode(t).ids)
            sc += len(t)
        res[style] = round(sa / sc / math.log(2), 4)
        a += sa
        c += sc
    res["cəmi"] = round(a / c / math.log(2), 4)
    return res


def grammar(model, tok):
    eot = tok.token_to_id("<|endoftext|>")
    res, fails, ok_all, n_all = {}, [], 0, 0
    for cat, pairs in _json("minimal_pairs.json").items():
        ok = 0
        for good, bad in pairs:
            if -nll(model, [eot] + tok.encode(good).ids) > -nll(model, [eot] + tok.encode(bad).ids):
                ok += 1
            else:
                fails.append(good)
        res[cat] = f"{ok}/{len(pairs)}"
        ok_all += ok
        n_all += len(pairs)
    res["cəmi"] = f"{ok_all}/{n_all}"
    return res, fails


@torch.no_grad()
def capitals(model, tok):
    top1 = top10 = 0
    for country, city in CAPITALS:
        ctx = tok.encode(f"{country} paytaxtı").ids
        wid = tok.encode(" " + city).ids[0]
        p = F.softmax(model(torch.tensor([ctx]))[0][0, -1], -1)
        rank = int((p > p[wid]).sum()) + 1
        top1 += rank == 1
        top10 += rank <= 10
    return {"top1": f"{top1}/{len(CAPITALS)}", "top10": f"{top10}/{len(CAPITALS)}"}


def fertility(tok):
    words = toks = 0
    for texts in _json("eval_texts.json").values():
        for t in texts:
            words += len(re.findall(r"\w+", t))
            toks += len(tok.encode(t).ids)
    return round(toks / words, 3)


def evaluate(model_dir):
    model, tok = load(model_dir)
    g, fails = grammar(model, tok)
    return {"params_M": round(model.num_params() / 1e6, 1), "arch": model.config.arch,
            "context": model.config.block_size, "bpc": bpc(model, tok), "grammar": g,
            "grammar_fails": fails, "capitals": capitals(model, tok), "tokens_per_word": fertility(tok)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Odlar standart testləri")
    ap.add_argument("model_dirs", nargs="+")
    ap.add_argument("--out", default=None, help="nəticəni JSON faylına yaz")
    a = ap.parse_args()
    allres = {}
    for d in a.model_dirs:
        name = os.path.basename(os.path.normpath(d))
        print(f"== {name}")
        r = evaluate(d)
        allres[name] = r
        print(f"  {r['params_M']}M parametr, {r['arch']}, kontekst {r['context']}")
        print(f"  BPC: {r['bpc']['cəmi']}  ({', '.join(f'{k} {v}' for k, v in r['bpc'].items() if k != 'cəmi')})")
        print(f"  Qrammatika: {r['grammar']['cəmi']}")
        print(f"  Paytaxtlar: ilk yer {r['capitals']['top1']}, ilk 10 {r['capitals']['top10']}")
        print(f"  Söz başına token: {r['tokens_per_word']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1)
        print("Yazıldı:", a.out)
