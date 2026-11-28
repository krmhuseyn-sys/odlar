"""İş masasının əsas hissələri üçün yoxlamalar: şəbəkəsiz, CPU-da ~1 dəqiqə.

    python tests/test_workbench.py

Hər test 0-cı mərhələdə ölçülən bir xassəni qoruyur. Biri pozularsa, səbəbi adından bəllidir.
"""
import math
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odlar import text as T                                    # noqa: E402
from odlar.model import GPT, GPTConfig, doc_mask              # noqa: E402
from odlar.sampler import BlockSampler, fixed_blocks          # noqa: E402
from odlar.train import SpikeGuard, make_lr                   # noqa: E402


def tiny(arch, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
                         arch=arch, **kw)).eval()


# ---------------------------------------------------------------- model
def test_doc_mask_isolates_documents():
    for arch in ("gpt2", "llama"):
        m = tiny(arch)
        x = torch.randint(1, 100, (2, 20))
        assert torch.allclose(m(x)[0], m(x, doc_sep=0)[0], atol=1e-5), "ayırıcı yoxdursa nəticə dəyişməməlidir"
        x[:, 7] = 0
        y = x.clone()
        y[:, :7] = torch.randint(1, 100, (2, 7))
        a, b = m(x, doc_sep=0)[0], m(y, doc_sep=0)[0]
        assert torch.allclose(a[:, 8:], b[:, 8:], atol=1e-5), f"{arch}: sonrakı sənəd əvvəlkini görür"
    assert doc_mask(torch.tensor([[1, 0, 2]]), 0)[0, 0].tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]


def test_kv_cache_matches_full_forward():
    for arch in ("gpt2", "llama"):
        m = tiny(arch)
        x = torch.randint(0, 100, (1, 24))
        full = m(x)[0]
        past = torch.zeros(2, 2, 1, 4, 0, 16)
        lg, past = m.forward_cached(x[:, :10], past)
        diff = [(lg - full[:, 9]).abs().max().item()]
        for t in range(10, 24):
            lg, past = m.forward_cached(x[:, t:t + 1], past)
            diff.append((lg - full[:, t]).abs().max().item())
        assert max(diff) < 1e-5, f"{arch}: keşli addım fərqlidir ({max(diff):.2e})"
        assert past.shape[4] == 24


# ---------------------------------------------------------------- seçici (K2)
def test_sampler_sees_every_block_once_per_epoch():
    T_ = 16
    arr = np.arange(16 * 500 + 7, dtype=np.uint16)
    s = BlockSampler([{"arr": arr, "repeat": 1, "label": "a"}], T_, seed=1)
    n = s.epoch_len()
    starts = []
    while s.epoch == 0:
        b = s.next_batch(10)
        starts += [int(r[0]) for r in b]
    starts = starts[:n]
    assert len(set(starts)) == n, "epoxada bir blok iki dəfə gəldi"
    covered = np.zeros(len(arr), bool)
    for st in starts:
        covered[st:st + T_] = True
    assert covered.mean() > 0.97, f"epoxa datanın {covered.mean():.0%}-ni əhatə edir"
    # köhnə üsul (təsadüfi pəncərələr) eyni sayda nümunə ilə ~63% əhatə edirdi
    rng = np.random.default_rng(0)
    old = np.zeros(len(arr), bool)
    for st in rng.integers(0, len(arr) - T_ - 1, n):
        old[st:st + T_] = True
    assert old.mean() < 0.7


def test_sampler_repeat_weights_and_resume():
    a = np.zeros(16 * 400, np.uint16)
    b = np.ones(16 * 400, np.uint16)
    s = BlockSampler([{"arr": a, "repeat": 1, "label": "a"}, {"arr": b, "repeat": 2.5, "label": "b"}], 16, seed=3)
    cov = s.coverage()
    assert abs(cov["b"] / cov["a"] - 2.5) < 0.05, cov
    for _ in range(7):
        s.next_batch(8)
    st = s.state_dict()
    nxt = s.next_batch(8)
    s2 = BlockSampler([{"arr": a, "repeat": 1, "label": "a"}, {"arr": b, "repeat": 2.5, "label": "b"}], 16, seed=3)
    s2.load_state_dict(st)
    assert (s2.next_batch(8) == nxt).all(), "davam edəndə eyni batch gəlmədi"


def test_sampler_ddp_ranks_split_the_global_batch():
    arr = np.arange(16 * 300, dtype=np.uint16)
    one = BlockSampler([{"arr": arr, "repeat": 1}], 16, seed=5).next_batch(8)
    r0 = BlockSampler([{"arr": arr, "repeat": 1}], 16, seed=5, rank=0, world=2).next_batch(4)
    r1 = BlockSampler([{"arr": arr, "repeat": 1}], 16, seed=5, rank=1, world=2).next_batch(4)
    assert (np.concatenate([r0, r1]) == one).all()


def test_fixed_val_blocks_are_stable():
    arr = np.arange(5000, dtype=np.uint16)
    assert (fixed_blocks(arr, 32, 10) == fixed_blocks(arr, 32, 10)).all()


# ---------------------------------------------------------------- öyrənmə sürəti və sıçrayış
def test_wsd_schedule():
    lr = make_lr({"lr": 1.0, "warmup": 10, "max_steps": 100, "min_lr_frac": 0.1, "schedule": "wsd",
                  "decay_frac": 0.2})
    assert lr(0) == 0.1 and lr(50) == 1.0 and lr(80) == 1.0
    assert abs(lr(100) - 0.1) < 1e-9 and lr(90) < 1.0


def test_spike_guard():
    random.seed(0)
    g = SpikeGuard(0.5, 25)
    assert not any(g.update(s, 3 - 0.0005 * s + random.gauss(0, 0.08)) for s in range(2000)), "adi küydə işə düşdü"
    g = SpikeGuard(0.5, 25)
    fired = []
    for s in range(400):
        l = 3 + random.gauss(0, 0.05) + (2 if 200 <= s < 210 else 0) + (1 if s >= 300 else 0)
        if g.update(s, l):
            fired.append(s)
            g.reset()
    assert fired and fired[0] >= 300, f"qısa sıçrayışda işə düşmədi, uzunda düşdü gözlənilirdi: {fired}"
    g = SpikeGuard(0.5, 25)
    g.update(0, 3.0)
    assert any(g.update(s, float("nan")) for s in range(1, 40))


# ---------------------------------------------------------------- mətn
def test_normalize():
    assert T.normalize("Mәdәniyyәt «a» ‘b’ Gürcüs­ ki") == 'Mədəniyyət "a" \'b\' Gürcüski'


def test_pii_masks_phones_not_numbers():
    t = T.mask_pii("+994 50 123 45 67, (012) 498-12-34, 0501234567, a@b.az")
    assert t == "[telefon], [telefon], [telefon], [e-poçt]", t
    keep = "Əhalisi 10 119 100 nəfər, 2 345 678 manat, 12345678, 1990-2000"
    assert T.mask_pii(keep) == keep


def test_wiki_tail_removed():
    t = T.clean_wiki("Mətn.\n\nTarixi\nBir.\n\nİstinadlar\n1. X\nKateqoriya:Y")
    assert t == "Mətn.\n\nTarixi\nBir."


def test_quality_rules():
    az = "Azərbaycan Respublikası Cənubi Qafqazda yerləşən dövlətdir. Paytaxtı Bakı şəhəridir. " * 10
    tr = "Türkiye Cumhuriyeti Anadolu ve Trakya'da yer alan bir ülkedir. Başkenti Ankara şehridir. " * 10
    ascii_az = "Azerbaycan edebiyyati xalq yaradiciligi ile baglidir ve cox qedimdir. " * 10
    R = T.DEFAULT_RULES
    assert T.quality(az, R) is None
    assert T.quality(tr, R) == "dil" and T.quality(ascii_az, R) == "dil"
    assert T.quality("mənzil 26\n" * 30, R) == "rəqəm"
    # nəqarətli şeir (hər dördlükdə bir təkrar): sətirlərin 25%-i təkrardır, simvolların ~10%-i.
    # Sətir sayı ilə ölçən köhnə qayda (30%) belə şeirlərə yaxın idi; real nümunədə pay 8% idi.
    poem = "\n".join(f"Bu gün səndən küsmüşəm, ürəyim yaman dolub, sözüm qalıb {i}-{j}." if j < 3
                     else "Yenə nədən küsmüsən?" for i in range(6) for j in range(4))
    assert T.quality(poem, R) is None
    assert T.quality("Menyu\nƏsas səhifə\nXəbərlər\n" * 10 + "Qısa mətn burada.", R) == "təkrar sətir"


def test_foreign_lines_stripped():
    t = "Bakı gözəl şəhərdir və çox qədimdir.\nЭто русский текст, который надо удалить отсюда.\nSon sətir burada."
    assert T.strip_foreign_lines(t) == "Bakı gözəl şəhərdir və çox qədimdir.\nSon sətir burada."


def test_minhash_near_duplicates():
    rnd = random.Random(1)
    vocab = "şəhər tarix dövlət xalq mədəniyyət dil ədəbiyyat musiqi kitab yazıçı şair il ölkə dəniz dağ çay kənd".split()
    docs = [" ".join(rnd.choice(vocab) for _ in range(400)) for _ in range(60)]
    near = []
    for d in docs[:30]:
        w = d.split()
        w[rnd.randrange(len(w))] = "dəyişdi"
        near.append(" ".join(w))
    bands = np.stack([T.minhash_bands(d) for d in docs + near])
    dup = T.duplicates(bands)
    assert not dup[:60].any(), "müstəqil sənədlər dublikat sayıldı"
    assert dup[60:].mean() > 0.95, f"oxşar sənədlərin {dup[60:].mean():.0%}-i tutuldu"


def test_cyrillic_conversion():
    az = "Азәрбајҹан Республикасы Ҹәнуби Гафгазда јерләшән дөвләтдир. Һүсејн Ҹавид бөјүк шаир иди. " * 3
    ru = "Азербайджанская Республика расположена на Южном Кавказе. Хусейн Джавид был поэтом. " * 3
    assert T.is_az_cyrillic(az) and not T.is_az_cyrillic(ru)
    assert T.cyr2lat(az).startswith("Azərbaycan Respublikası Cənubi Qafqazda yerləşən dövlətdir. Hüseyn Cavid")
    assert T.cyr2lat("ИСТИГЛАЛ Ырмаг Гарабағ ҸӘФӘР") == "İSTİQLAL Irmaq Qarabağ CƏFƏR"
    assert not T.is_az_cyrillic("Bakı gözəl şəhərdir. " * 20)


def test_routes_by_domain_and_length():
    from odlar import data as D
    D._init_worker({"label": "veb", "min_chars": 10, "rules": None, "wiki": False, "pii": False,
                    "dedup": False, "cyrillic": True,
                    "routes": [{"label": "xeber", "domains": ["oxu.az"]},
                               {"label": "veb_uzun", "min_chars": 500}]})
    short, long_ = "Qısa mətn burada yazılıb.", "Uzun mətn burada yazılıb. " * 40
    assert D._process((short, "https://www.oxu.az/x"))[0] == "xeber"
    assert D._process((long_, "https://news.oxu.az/x"))[0] == "xeber"      # domen birinci yoxlanır
    assert D._process((long_, "https://blog.example.az/x"))[0] == "veb_uzun"
    assert D._process((short, None))[0] == "veb"
    out, t, _ = D._process(("Азәрбајҹан дили гәдим дилдир. Бакы шәһәри. " * 10, None))
    assert t.startswith("Azərbaycan dili qədim dildir.")


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_digits_and_specials():
    from odlar.data import build_tokenizer, SPECIAL_TOKENS
    tok, trainer = build_tokenizer(400)
    tok.train_from_iterator(["2021-ci ildə Bakıda 1500 manat. " * 50, "Qış gecəsi idi. " * 50], trainer=trainer)
    e = tok.encode("2021-ci ildə 1500")
    assert [t for t in e.tokens if any(c.isdigit() for c in t)] == ["2", "0", "2", "1", "1", "5", "0", "0"]
    assert [tok.token_to_id(s) for s in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    assert tok.decode(e.ids) == "2021-ci ildə 1500"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        for n, f in tests:
            try:
                f()
                print(f"  keçdi   {n}")
            except Exception as e:                       # noqa: BLE001
                failed += 1
                print(f"  XƏTA    {n}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} keçdi")
    sys.exit(1 if failed else 0)
