# Kor nöqtələr: xam ölçülər (26 noyabr 2026)

Odlar-2 planında olmayan mövzuların yoxlanması. Qərarlar plan sənədinin "Kor nöqtələr" tabındadır, burada ölçülərin özü və onları təkrar etmək üçün skriptlər var.

## 1. Baza modellər: Belebele azj_Latn paraqraflarında BPC

Belebele-nin Azərbaycan hissəsindəki 488 unikal paraqraf (253 571 simvol, FLORES-dən peşəkar tərcümə, CC BY-SA 4.0). Simvol başına bit, az olan yaxşıdır. Hər model öz tokenizeri ilə ölçülür; BPC tokenizerdən asılı olmadığı üçün müqayisə ədalətlidir.

| Model | Parametr | BPC (Belebele) | BPC (Odlar-ın 16 mətni) |
|---|---|---|---|
| Odlar-1 | 13.8M | 1.530 | 1.543 |
| Odlar-1.5 | 40.0M | 1.394 | 1.224 |
| ai-forever/mGPT-1.3B-azerbaijan | 1 417.6M | 1.269 | 1.145 |
| Qwen/Qwen3-0.6B-Base | 596.0M | 2.036 | 2.380 |
| HuggingFaceTB/SmolLM2-135M (ingilis dilli) | 134.5M | 3.424 | 3.762 |

Ölçülə bilməyənlər:
- `allmalab/gpt2-aze`: giriş razılıq tələb edir.
- `nijatzeynalov/gpt2-azerbaijani-small`: köhnə pickle formatı, təhlükəsiz yüklənmir (`weights_only` xətası). Yoxlamanı söndürmək faylın içindəki kodu işə sala bilər, ona görə edilmədi.

Skriptlər: `eval/baselines/` (`belebele_passages.py`, `bpc_belebele.py`, `bpc_hf.py`, `bpc_mgpt.py`). mGPT 7 GB RAM-da bir prosesdə yaddaşı doldurur, ona görə 100 paraqraflıq hissələrlə ölçüldü.

## 2. Təsadüfi pəncərə seçimi: datanın nə qədəri görünür

`train.py` hər addımda B təsadüfi başlanğıc seçir (təkrarla). L tokenlik data üzərində N token görəndən sonra əhatə ≈ 1 − e^(−N/L).

| Görülən / unikal | Simulyasiya | Nəzəri |
|---|---|---|
| 0.71 (Odlar-1.5: 245.8M / 345M) | 0.486 | 0.508 |
| 1.0 | 0.626 | 0.632 |
| 2.0 | 0.868 | 0.865 |

## 3. Sənəd sərhədləri (FineWeb-2, 300 sənəd, Odlar-1.5 tokenizeri)

- Sənəd uzunluğu: median 302 token, orta 750, 83%-i 1024 tokendən qısa.
- Təsadüfi 1024 tokenlik pəncərədə sənəd sayı: median 2, orta 2.38; yalnız 41%-ində tək sənəd.

## 4. Qrammatika testinin statistik etibarı (40 minimal cüt)

- Wilson 95% intervalları: Odlar-1 31/40 → 0.625–0.877; Odlar-1.5 35/40 → 0.739–0.945.
- Cütləşdirilmiş müqayisə: yalnız Odlar-1-in səhv etdiyi 5 cüt, yalnız Odlar-1.5-in səhv etdiyi 1 cüt, ikisinin də səhv etdiyi 4 cüt. Dəqiq McNemar p = 0.219.
- Nəticə: 40 cütlə bu fərq təsadüfdən ayırd edilmir.

## 5. Hostinq yoxlamaları

- GitHub 100 MiB-dən böyük faylı bloklayır; Pages: sayt ≤ 1 GB, ayda ~100 GB trafik (yumşaq limit).
- Hugging Face `resolve` linki `Origin: https://krmhuseyn-sys.github.io` sorğusuna `access-control-allow-origin` ilə cavab verir, CDN `access-control-allow-origin: *` və `206 Partial Content` qaytarır (26 noyabr 2026, curl ilə).
- `huggingface_hub` 2.4.0-da `HfApi.super_squash_history` var.

## 6. Hüquqi mənbə

21 aprel 2026-cı il tarixli 387-VIIQD nömrəli qanun (Milli Məclisin saytındakı PDF-dən birbaşa oxundu) Cinayət Məcəlləsinə 148-2 və 242-1-ci maddələri əlavə edir. 148-2.1: şəxsin razılığı olmadan onun görüntüsündən və ya səsindən istifadə etməklə süni intellektlə reallığı əks etdirməyən foto, video və ya audio hazırlama və ya yayma; 3000–7000 manat cərimə, 360–480 saat ictimai iş, 3 ilədək azadlığın məhdudlaşdırılması və ya 3 ilədək azadlıqdan məhrum etmə. 148-2.2 (ağırlaşdırıcı hallar): 3–5 il. Maddə mətnə deyil, görüntü və səsə aiddir. Bu hüquqi məsləhət deyil.
