# Odlar-1: addım-addım

## 1-ci hissə: Kaggle-da modeli öyrət (pulsuz GPU)

1. kaggle.com-da hesab aç və **telefon nömrəni təsdiqlə**. Bu olmadan GPU və internet açılmır.
2. **Create → New Notebook**, sonra **File → Import Notebook** → `train_kaggle.ipynb` faylını seç.
3. Sağ paneldə **Session options**:
   - Accelerator: **GPU T4 x2** (və ya P100)
   - Internet: **On**
4. **Run All** düyməsini bas. Təlim təxminən 1–2.5 saat çəkir. Hər 500 addımda model nümunə cümlə yazır: əvvəl qarışıq olacaq, sonra get-gedə daha düzgün cümlələr çıxacaq.
   - `val` rəqəmi azalırsa, model öyrənir. 5-dən başlayıb 3-ə yaxın və ya daha aşağı düşməlidir.
5. Bitəndə sağdakı **Output** bölməsindən `my_llm.zip` faylını endir.

> Kaggle səhifəni bağlasan təlim dayana bilər. Uzun təlimdə **Save Version → Save & Run All (Commit)** istifadə et: o, arxa planda sona qədər işləyir və nəticəni saxlayır.

## Alternativ: Google Colab-da öyrət

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krmhuseyn-sys/odlar/blob/main/train_colab.ipynb)

Kaggle-da GPU növbəsi uzun çəkəndə istifadə et. Linki aç, **Runtime → Change runtime type → T4 GPU** seç, sonra **Run all** bas. Model hər 500 addımda Google Drive-a (`MyDrive/Odlar`) yazılır. Sessiya kəsilsə, yenidən **Run all** bas, təlim qaldığı yerdən davam edəcək.

## 2-ci hissə: VS Code-da lokal sayt

Layihə qovluğu: `C:\Users\Admın\PycharmProjects\Odlar`

1. VS Code-da **File → Open Folder** seç və `Odlar` qovluğunu aç. Sol paneldə **Extensions** (Ctrl+Shift+X) bölməsinə keç və **Python** əlavəsini (Microsoft) quraşdır, əgər yoxdursa.
2. Kaggle-dan endirdiyin `my_llm.zip`-i aç və içindəki `model.pt` və `tokenizer.json` fayllarını bu qovluğa at. `model.py` artıq burada var.
3. VS Code-da terminal aç (**Ctrl + `**) və bir dəfə bu əmri işlət:

```
pip install -r requirements.txt
```

4. `app.py` faylını aç və sağ yuxarıdakı ▶ (**Run Python File**) düyməsini bas. Brauzerdə **http://localhost:7860** açılacaq.

Model cəmi ~14M parametrdir, 8 GB RAM-da saniyələr içində cavab verir. GPU lazım deyil.

## Ayarlarla oynamaq

- **Temperatur:** 0.5 seçsən, model daha ehtiyatlı və təkrarlanan mətn yazır. 1.2 seçsən, daha yaradıcı, amma daha "dəli" mətn yazır.
- **Daha ağıllı model:** notebook-un ilk xanasında `N_LAYER, N_HEAD, N_EMBD = 8, 8, 512` və `MAX_STEPS = 30_000` et. Model ~30M parametr olacaq, təlim daha uzun çəkəcək.
- **Başqa data:** "2) Data" xanasında `texts` siyahısını öz mətnlərinlə əvəz et (kitablar, xəbərlər və s.).

## Gözlənti

Bu model **sual-cavab etmir**. Mətni davam etdirir, Vikipediya üslubunda Azərbaycan dilində cümlələr yazır. Faktlar tez-tez uydurma olacaq. Bu normaldır, çünki böyük modellər bundan min dəfələrlə çox data və hesablama ilə öyrədilir. Amma bunu sən sıfırdan qurmusan. 