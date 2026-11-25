# Odlar-1 vs Odlar-1.5

| | Odlar-1 | Odlar-1.5 |
|---|---|---|
| Parametr | 13.8M | 40.0M |
| Qat / ölçü / baş | 6 / 384 / 6 | 10 / 512 / 8 |
| Kontekst | 256 token | 512 token |
| Lüğət | 8,000 | 16,000 |
| Data | Vikipediya, 84.6M token | Vikipediya + FineWeb-2, 345M token |
| Təlim | 15,000 addım, 22 dəq, T4 | 15,000 addım, 118 dəq, T4 |
| Ən yaxşı val loss (öz tokenizeri ilə) | 3.156 | 3.353 |
| Brauzer modeli (int8) | 17 MB | 49 MB |

## Simvol başına bit (bits-per-character), heç bir modelin görmədiyi mətnlər
Lüğətdən asılı olmayan ədalətli müqayisə. Az = yaxşı. Mətnlər: [`../eval/eval_texts.json`](../eval/eval_texts.json), skript: [`../eval/bpc.py`](../eval/bpc.py).

| Üslub | Odlar-1 | Odlar-1.5 | Yaxşılaşma |
|---|---|---|---|
| Ensiklopedik | 0.952 | 0.926 | 3% |
| Xəbər | 1.333 | 0.913 | 32% |
| Bədii | 1.860 | 1.598 | 14% |
| Gündəlik danışıq | 2.551 | 1.691 | 34% |
| **Cəmi** | **1.530** | **1.208** | **21%** |

Tokenizer səmərəliliyi: Odlar-1 bir tokendə orta hesabla 3.86 simvol, Odlar-1.5 isə 4.51 simvol saxlayır.

## Paytaxt testi, 20 ölkə ("X-in paytaxtı" → düzgün cavabın yeri)
Skript: [`../eval/facts.py`](../eval/facts.py)

| | Odlar-1 | Odlar-1.5 |
|---|---|---|
| Düzgün cavab 1-ci yerdə | 2 / 20 | 7 / 20 |
| Düzgün cavab ilk 10-da | 6 / 20 | 13 / 20 |

Ən böyük dəyişikliklər: İran 36 → 1, Macarıstan 34 → 1, Özbəkistan 36 → 1, Yaponiya 28 → 1. Pisləşənlər: Yunanıstan 10 → 190, Fransa 1 → 5, Almaniya 12 → 27. Veb datasında qonşu və tez-tez xəbərlərdə keçən ölkələr güclənib.

## Riyaziyyat və sual-cavab
Hər iki model sadə toplamanı bacarmır ("5 + 3 =" → Odlar-1: "4 = 4 = 4", Odlar-1.5: "3 = 3 = 3"). Nümunə verildikdə də öyrənmir (in-context learning yoxdur).
"Sual: Azərbaycanın paytaxtı hansı şəhərdir? Cavab:" → Odlar-1 mövzudan çıxır ("Mənim bu gün..."), Odlar-1.5 mövzuda qalır, amma cavab vermir ("Azərbaycan paytaxtının ən böyük şəhəri...").

## Nəticə
Model 3 dəfə böyüdükdə və data 4 dəfə artdıqda dil keyfiyyəti xeyli yaxşılaşır (xüsusən xəbər və danışıq dilində), fakt bilgisi qismən artır, amma riyazi və məntiqi bacarıqlar bu ölçüdə hələ yaranmır.
