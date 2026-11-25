# Odlar

Azərbaycan dilində **sıfırdan** öyrədilmiş dil modelləri. Hazır modelin üzərində yox: tokenizer, model və təlim tamamilə bu repodadır, pulsuz GPU ilə (Colab, Kaggle).

**Canlı demo:** [krmhuseyn-sys.github.io/odlar](https://krmhuseyn-sys.github.io/odlar/). Modellər tamamilə brauzerdə işləyir (ONNX int8), server lazım deyil.

| Model | Parametr | Kontekst | Data | BPC (az = yaxşı) | Qrammatika |
|---|---|---|---|---|---|
| Odlar-1 | 13.8M | 256 | Vikipediya | 1.543 | 31/40 |
| Odlar-1.5 | 40.0M | 512 | Vikipediya + FineWeb-2 | 1.224 | 35/40 |
| Odlar-2 | ~110M | 1024 | + bədii ədəbiyyat | hazırlanır | |

Ətraflı nəticələr: [results/meqale_ucun_neticeler.md](results/meqale_ucun_neticeler.md)

## Model öyrətmək

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krmhuseyn-sys/odlar/blob/main/notebooks/train.ipynb)

1. Linki aç, **Runtime → Change runtime type → T4 GPU** seç.
2. İlk xanada `CONFIG`-i seç (`configs/odlar15.yaml` və s.).
3. **Runtime → Run all**. Model hər 500 addımda Google Drive-a (`MyDrive/Odlar/runs/...`) yazılır. Sessiya kəsilsə, yenidən **Run all**: qaldığı yerdən davam edir.

İş masasının quruluşu, komandalar və config yazmaq qaydası: **[MODELS.md](MODELS.md)**.

## Lokal sayt (kompüterdə)

```
pip install -r requirements.txt
python app.py
```

Qovluqda `odlar/` (kod), `model.pt` + `tokenizer.json` (Odlar-1) və `odlar15/model.pt` + `odlar15/tokenizer.json` (Odlar-1.5) olmalıdır. Brauzer özü açılır. GPU lazım deyil.

## Gözlənti

Bu modellər **sual-cavab etmir**: yazdığın mətni davam etdirir. Qrammatikanı yaxşı bilir, faktları bilmir: yerləri, tarixləri, adları uydurur. Növbəti mərhələ, söhbət modeli Simurq, bunun üzərində qurulacaq.
