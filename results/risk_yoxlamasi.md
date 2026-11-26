# Risk yoxlaması (26 noyabr 2026)

Odlar-2 planındakı risklər ehtimala görə yox, ölçü ilə yoxlanıldı. Burada xam nəticələr və üsullar var.

## R1. Brauzerdə sürət (KV-keş yoxdur)

Təsadüfi çəkili 110M Llama modeli (12 qat, 768, lüğət 32k) ONNX int8-ə ixrac olundu və onnxruntime-web 1.20.1 (WASM, 1 axın, Node) ilə ölçüldü. Saytdakı kod hər yeni token üçün bütün konteksti yenidən hesablayır, ona görə bir tokenin qiyməti kontekstin uzunluğundan asılıdır.

| Model | Kontekst | ms/token | token/san |
|---|---|---|---|
| Odlar-2 (110M, test) | 32 | 206 | 4.9 |
| Odlar-2 (110M, test) | 256 | 1397 | 0.7 |
| Odlar-2 (110M, test) | 512 | 3092 | 0.3 |
| Odlar-2 (110M, test) | 1024 | 7470 | 0.1 |
| Odlar-1.5 (saytdakı) | 32 | 70 | 14.3 |
| Odlar-1.5 (saytdakı) | 256 | 537 | 1.9 |
| Odlar-1.5 (saytdakı) | 512 | 1270 | 0.8 |

Nəticə: KV-keş olmadan Odlar-2 saytda istifadə oluna bilməz. Skript: `eval/risk/brauzer_suret.mjs`.

## R5. Brauzer modelinin ölçüsü

Eyni test modelinin int8 ONNX faylı 134.8 MB-dır (planda 112 MB təxmin edilmişdi). Səbəb: giriş və çıxış çəkiləri PyTorch-da ortaqdır, ONNX-də isə iki dəfə yazılır (~25 MB).

## R3. Kitab korpusunda OCR (LocalDoc/books_dataset)

30 təsadüfi yerdən 20 ardıcıl cümlə, cəmi 600 cümlə (Dataset Viewer API).

| Əlamət | Cümlə |
|---|---|
| yumşaq tire (U+00AD) | 90 (15%) |
| söz içində qırılma ("ehtiyatla­ növündə") | 32 |
| kiril hərf | 12 |
| rəqəm-hərf yapışması (səhifə nömrəsi) | 8 |
| ən azı bir əlamət | 133 (22%) |

Bu əlamətlər hərf səhvlərini ("əsasmda", "düşmüson", "toı can") tutmur, ona görə real pay daha yüksəkdir. Nümunədəki 30 kitabdan 3-ü bədii ədəbiyyatdır.

## R7. FineWeb-2 azj_Latn tərkibi

7 291 231 sənəddən 25 təsadüfi yerdə 12 ardıcıl sənəd, cəmi 300.

- Xəbər saytlarından (URL siyahısı ilə): 75 (25%). Siyahı natamamdır, real pay bir az yüksəkdir.
- Mətndə xəbər dili (ən azı 2 ifadə: "xəbər verir", "bildirib" və s.): 95 (32%).
- Median sənəd uzunluğu 1377 simvol, 39%-i 1000 simvoldan qısadır (modellərin tez dayanmasının səbəbi).
- Başqa dil şübhəsi: 3 sənəd (1%). Kiril "ә" işlədən: 1 (0.3%). Xüsusi hərfsiz yazı (ə əvəzinə e): 3 (1%).

## R10. fp16 sabitliyi (Odlar-1.5 log-u)

15 000 addımda NaN yoxdur. Val loss 3 dəfə 0.03–0.05 artıb (9 500, 11 000, 14 000-ci addımlar): 40 batch-lik ölçmədə səs-küy səviyyəsidir.

## R11. Pulsuz GPU şərtləri

- Kaggle P100-ü 15 sentyabr 2026-da çıxarıb, 2×T4 qalıb ([AIMultiple, 6 noyabr 2026](https://aimultiple.com/gpu-cluster)).
- Google Cloud T4-ü 1 avqust 2027-də bağlayır ([Google Cloud](https://docs.cloud.google.com/compute/docs/eol/t4-eos)). Kaggle və Colab-ın pulsuz T4-ünə təsiri elan olunmayıb, amma 2027-də dəyişiklik gözlənilir.
