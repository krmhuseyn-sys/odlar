# Odlar iş masası

Bütün Odlar modelləri eyni kodla öyrədilir. Yeni model = yeni YAML faylı. Kod dəyişmir.

## Quruluş

```
odlar/              model kodu (paket)
  model.py          GPT: "gpt2" (Odlar-1, 1.5) və "llama" (Odlar-2 üçün) quruluşları
  config.py         YAML config-lərin oxunması, base: ilə miras, --set ilə dəyişiklik
  data.py           mənbələr -> tokenizer -> train.npy / val.npy
  train.py          təlim, qaldığı yerdən davam, checkpoint, log.csv, HF Hub
  generate.py       terminaldan mətn yazdırmaq
  evaluate.py       standart testlər (BPC, qrammatika, paytaxtlar, söz başına token)
  export.py         brauzer üçün ONNX int8 + tokenizer JSON
configs/            hər model üçün bir fayl
notebooks/train.ipynb   Colab/Kaggle üçün bir notebook, bütün modellər üçün
eval/               test dataları (eval_texts.json, minimal_pairs.json)
docs/               sayt (GitHub Pages)
```

## Komandalar

```bash
# Sınaq (GPU-suz, ~1 dəqiqə): iş masası işləyirmi?
python -m odlar.train configs/tiny.yaml

# Təlim (Colab-da notebooks/train.ipynb bunu edir)
python -m odlar.train configs/odlar15.yaml
python -m odlar.train configs/odlar15.yaml --set train.max_steps=100 out_dir=/content/run

# Yalnız data hazırlığı
python -m odlar.data configs/odlar2.yaml

# Mətn yazdır
python -m odlar.generate runs/odlar15 "Qış gecəsi idi" -n 3

# Testlər (bir neçə modeli müqayisə)
python -m odlar.evaluate runs/odlar1 runs/odlar15 --out results/muqayise.json

# Sayt üçün
python -m odlar.export runs/odlar2 --name odlar2 --to docs
```

## Modellər

| Model | Config | Quruluş | Parametr | Kontekst | Data | Val loss | BPC | Qrammatika |
|---|---|---|---|---|---|---|---|---|
| Odlar-1 | `odlar1.yaml` | gpt2 | 13.8M | 256 | Vikipediya | 3.156 | 1.543 | 31/40 |
| Odlar-1.5 | `odlar15.yaml` | gpt2 | 40.0M | 512 | Vikipediya + FineWeb-2 | 3.353 | 1.224 | 35/40 |
| Odlar-2 | `odlar2.yaml` | llama | ~110M | 1024 | + bədii ədəbiyyat | qaralama | | |

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

## Qaydalar

- Hər yeni model `odlar.evaluate` ilə yoxlanılır və nəticə bu cədvələ yazılır.
- Hər təlim qovluğunda `config.yaml` saxlanılır: hansı modelin necə öyrədildiyi həmişə bəllidir.
- Kontekst uzatmaq yalnız `llama` quruluşunda mümkündür (RoPE): `block_size` və `rope_scale` artırılır,
  `init_from` ilə köhnə model yüklənir, qısa əlavə təlim edilir.
- Köhnə `model.pt` faylları (Odlar-1, 1.5) yeni kodla dəyişmədən yüklənir.
- Kökdəki `model.py` və köhnə notebook-lar (`train_colab*.ipynb`) tarixçə üçün saxlanılır, yeni iş `odlar/` ilədir.
