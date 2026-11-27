# Odlar iş masası

Bütün Odlar modelləri eyni kodla öyrədilir. Yeni model = yeni YAML faylı. Kod dəyişmir.

## Quruluş

```
odlar/              model kodu (paket)
  model.py          GPT: "gpt2" (Odlar-1, 1.5) və "llama" (Odlar-2 üçün) quruluşları
  config.py         YAML config-lərin oxunması, base: ilə miras, --set ilə dəyişiklik
  data.py           data hazırlığı: yığım -> təmizləmə -> MinHash dublikat -> tokenizer -> tokenlər
  text.py           normallaşdırma, Vikipediya təmizliyi, keyfiyyət filtrləri, şəxsi məlumat, MinHash
  sampler.py        təlim blokları: epoxada hər blok bir dəfə, mənbə çəkiləri, davam etmə (K2)
  train.py          təlim: bir və ya bir neçə GPU (DDP), WSD, sənəd maskası, loss sıçrayışında geri dönüş,
                    mənbə üzrə val loss, checkpoint, HF Hub (tarixçə sıxılır)
  hubutil.py        HF_TOKEN-i Colab/Kaggle Secrets-dən tapır
  generate.py       terminaldan mətn yazdırmaq
  evaluate.py       standart testlər (BPC, qrammatika, paytaxtlar, söz başına token)
  export.py         brauzer üçün ONNX int8 (KV-keşli, ortaq embedding bir dəfə) + tokenizer JSON
configs/            hər model üçün bir fayl
notebooks/train.ipynb   Colab/Kaggle üçün bir notebook, bütün modellər üçün
eval/               test dataları (eval_texts.json, minimal_pairs.json)
tests/              iş masasının yoxlamaları: python tests/test_workbench.py (şəbəkəsiz, ~2 san)
docs/               sayt (GitHub Pages)
```

## Komandalar

```bash
# Yoxlamalar (şəbəkəsiz, ~2 saniyə)
python tests/test_workbench.py

# Sınaq (GPU-suz, ~2 dəqiqə): yeni data boru xətti və təlim uçdan-uca
python -m odlar.train configs/tiny2.yaml

# Bir neçə GPU (Kaggle T4x2): eyni config, qlobal batch dəyişmir
torchrun --standalone --nproc_per_node=2 -m odlar.train configs/odlar2.yaml

# Təlim (Colab-da notebooks/train.ipynb bunu edir)
python -m odlar.train configs/odlar15.yaml
python -m odlar.train configs/odlar15.yaml --set train.max_steps=100 out_dir=/content/run

# Yalnız data hazırlığı
python -m odlar.data configs/odlar2.yaml

# Mətn yazdır
python -m odlar.generate runs/odlar15 "Qış gecəsi idi" -n 3

# Testlər (bir neçə modeli müqayisə)
python -m odlar.evaluate runs/odlar1 runs/odlar15 --out results/muqayise.json

# Sayt üçün (defolt: KV-keşli; --no-kv köhnə format)
python -m odlar.export runs/odlar2 --name odlar2 --to web
```

## Modellər

| Model | Config | Quruluş | Parametr | Kontekst | Data | Val loss | BPC | Qrammatika |
|---|---|---|---|---|---|---|---|---|
| Odlar-1 | `odlar1.yaml` | gpt2 | 13.8M | 256 | Vikipediya | 3.156 | 1.543 | 31/40 |
| Odlar-1.5 | `odlar15.yaml` | gpt2 | 40.0M | 512 | Vikipediya + FineWeb-2 | 3.353 | 1.224 | 35/40 |
| Odlar-2 | `odlar2.yaml` | llama | ~110M | 1024 | + bədii ədəbiyyat | qaralama | | |

Keçid sınağı: `sinaq_t4x2.yaml` (Odlar-2 ölçüsü, 300 addım, iki GPU). Kiçik sınaq: `tiny2.yaml`.

Val loss-ları müqayisə etmək olmaz (lüğətlər fərqlidir). Müqayisə üçün BPC-yə bax.

## Config necə yazılır

```yaml
name: odlar2
model:
  arch: llama          # gpt2 | llama
  block_size: 1024     # kontekst
  n_layer: 12
  n_head: 12
  n_embd: 768
data:
  vocab_size: 32000
  sources:
    - type: hf                       # Hugging Face dataseti
      path: wikimedia/wikipedia
      name: 20231101.az
      max_chars: 300000000
    - type: files                    # öz faylların: .txt, .md, .pdf
      glob: books/**/*.txt
      repeat: 3                      # kiçik, dəyərli data 3 dəfə çox görünür
train:
  batch_size: 16
  grad_accum: 4
  lr: 3.0e-4
  max_steps: 30000
  hub_repo: krmhus/odlar-2           # istəyə bağlı, HF_TOKEN lazımdır
```

`base: odlar2.yaml` yazsan, həmin faylın üzərinə yalnız dəyişənləri yazırsan (bax `odlar2_ctx4096.yaml`).

Mənbə seçimləri (`data.sources`, hamısı istəyə bağlı; tam siyahı `odlar/data.py`-nin əvvəlində):

| Seçim | Defolt | Nə edir |
|---|---|---|
| `repeat` | 1 | təlimdə neçə nüsxə; kəsr ola bilər (1.5) |
| `filter` | true | keyfiyyət filtrləri; `false` və ya dəyişən qaydalar, məs. `{max_digit: 0.25}` |
| `wiki` | path-da "wikipedia" varsa | "İstinadlar", "Xarici keçidlər" və s. bölmələri kəsir |
| `pii` | true | e-poçt və Azərbaycan telefon nömrələrini maskalayır |
| `dedup` | true | MinHash oxşar dublikat yoxlaması (config-də əvvəl gələn mənbə saxlanır) |
| `tok_weight` | 1 | tokenizer nümunəsində əlavə çəki |
| `routes` | yox | domenə görə ayrı səbət, məs. xəbər saytları öz `max_chars` limiti ilə |

Təlim seçimləri (`train`): `schedule: cosine | wsd`, `decay_frac`, `doc_mask` (sənəd maskası),
`spike_rollback`, `spike_delta`, `spike_patience`, `max_rollbacks`, `hub_every`, `hub_squash`,
`ckpt_dir` (checkpoint ayrı qovluqda; Colab-da Drive zibil qutusu dolmasın deyə).

Data formatı: `data/meta.json` (format 2) + hər mənbə üçün `<ad>.train.bin` və `<ad>.val.bin` (uint16).
`data.hub_dataset` yazılıbsa, hazır data gizli HF dataset-ə yüklənir və növbəti sessiyada oradan endirilir.
Köhnə formatda (train.npy / val.npy) hazırlanmış data da oxunur.

## Qaydalar

- Hər yeni model `odlar.evaluate` ilə yoxlanılır və nəticə bu cədvələ yazılır.
- Hər təlim qovluğunda `config.yaml` saxlanılır: hansı modelin necə öyrədildiyi həmişə bəllidir.
- Kontekst uzatmaq yalnız `llama` quruluşunda mümkündür (RoPE): `block_size` və `rope_scale` artırılır,
  `init_from` ilə köhnə model yüklənir, qısa əlavə təlim edilir.
- Köhnə `model.pt` faylları (Odlar-1, 1.5) yeni kodla dəyişmədən yüklənir.
- Kökdəki `model.py` və köhnə notebook-lar (`train_colab*.ipynb`) tarixçə üçün saxlanılır, yeni iş `odlar/` ilədir.
