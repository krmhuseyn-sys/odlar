# Odlar-1: təlim nəticələri

**Təlim:** Google Colab, T4 GPU, 25 noyabr 2026
**Model:** 13.8M parametr (6 qat, 6 baş, 384 ölçü, kontekst 256 token)
**Data:** Azərbaycan Vikipediyası (`wikimedia/wikipedia`, `20231101.az`): 134,447 məqalə, 300M simvol, 84.6M token
**Tokenizer:** Byte-level BPE, 8000 token

Addım-addım loss və nümunələr: [`odlar1_training_log.csv`](odlar1_training_log.csv)

## Müşahidələr
- **Addım 0:** təsadüfi söz yığını (val loss 9.05)
- **Addım 500:** sözlər bir-birinə bağlanmağa başlayır (5.49)
- **Addım 1000:** qrammatika ortaya çıxır, şəkilçilər düzgün işlənir (4.71)
- **Addım 1500–3500:** Vikipediya üslubu: tarixlər, təşkilat adları, kateqoriyalar (4.38 → 3.74)
- 500 addım ≈ 0.7 dəqiqə; təlimin tamamı təxminən 20–25 dəqiqə
