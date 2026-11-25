# Odlar-1: təlim nəticələri

**Təlim:** Google Colab, T4 GPU, 25 noyabr 2026
**Model:** 13.8M parametr (6 qat, 6 baş, 384 ölçü, kontekst 256 token)
**Data:** Azərbaycan Vikipediyası (`wikimedia/wikipedia`, `20231101.az`, 230 MB parquet). Dump-da 196,158 məqalə var; 200 simvoldan qısa olanlar atıldı və 134,447 məqalə qaldı: 300M simvol, 84.6M token
**Tokenizer:** Byte-level BPE, 8000 token

Addım-addım loss və nümunələr: [`odlar1_training_log.csv`](odlar1_training_log.csv)

## Müşahidələr
- **Addım 0:** təsadüfi söz yığını (val loss 9.05)
- **Addım 500:** sözlər bir-birinə bağlanmağa başlayır (5.49)
- **Addım 1000:** qrammatika ortaya çıxır, şəkilçilər düzgün işlənir (4.71)
- **Addım 1500–3500:** Vikipediya üslubu: tarixlər, təşkilat adları, kateqoriyalar (4.38 → 3.74)
- 500 addım ≈ 0.7 dəqiqə; təlimin tamamı təxminən 20–25 dəqiqə

## Ekran şəkilləri
- [Data yükləmə və tokenizer](screenshots/01_data_and_tokenizer.png): tokenizer "Bakı", " Azərbaycanın", " paytaxtı" sözlərini bütöv tanıyır, "dır" şəkilçisini ayırır (ekranda byte-level formada görünür: `Ġ` = boşluq, `Ä±` = ı)
- [Təlim: addım 0–1500](screenshots/02_training_steps_0-1500.png)
- [Təlim: addım 0–3500](screenshots/03_training_steps_0-3500.png)
