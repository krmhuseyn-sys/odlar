# 0-cı mərhələ: iş masası əlavələri (27 noyabr 2026)

Odlar-2 planının 0-cı mərhələsi: təlimdən əvvəl kodda olmalı olanlar. Bütün rəqəmlər bu gün ölçülüb.
Ölçmələrin çoxu `tests/test_workbench.py`-də yoxlama kimi saxlanılıb.

## Təlim

| Nə | Necə yoxlanıb | Nəticə |
|---|---|---|
| Bloklar (K2) | 500 blokluq sintetik data | yeni: epoxada hər blok bir dəfə, datanın 99.7%-i görünür (qalanı sürüşmənin kəsdiyi kənar); köhnə təsadüfi pəncərələr eyni sayda nümunədə 65.2% (nəzəri 1-e^-1 = 63%) |
| Davam etmə | 200 addım, dayandır, 300-ə qədər davam | data mövqeyi checkpoint-dən bərpa olundu (6432 = 201 addım x 32 blok) |
| DDP (torchrun) | eyni config, 1 proses və 2 proses (CPU, gloo) | log.csv 4 rəqəm dəqiqliyində eyni: train 8.3004 / 6.7530 / 5.5721, val 8.2943 / 6.0041 / 5.3321 |
| WSD (K4) | addım 0, 100, 200-də lr | 1e-4 (isinmə), 2e-3 (sabit), 2e-4 (son, 10%) |
| Sənəd maskası (K3) | ayırıcıdan sonrakı tokenlər əvvəlki sənədi görməməlidir | hər iki quruluşda fərq < 1e-5; ayırıcı yoxdursa nəticə dəyişmir |
| Loss sıçrayışı | süni sıçrayış (spike_delta = -10) | 25 addımdan sonra son checkpoint-ə qayıdır, data irəli gedir; 3-cü dəfədə dayanır. 10 addımlıq qısa sıçrayışda işə düşmür |
| Hub tarixçəsi (K7) | `huggingface_hub` 2.4.0 imzaları | `create_commit`, `super_squash_history` çağırışları uyğundur. Real yükləmə tokensiz sınanmayıb: Kaggle sınağında yoxlanacaq |

## Data

FineWeb-2 azj_Latn-in ilk 6000 sənədi üzərində:

- **Filtrlərdən sonra 98.97% keçir** (62 sənəd atılır).
  - "dil" (58): "ə" əvəzinə "e" ilə yazılmış mətnlər ("Xalq Cebhesi", "edileceyi"). Atılması qəsdəndir: model bu yazılışı öyrənməməlidir.
  - "skript" (4): latın hərfi kimi görünən kiril hərflərlə yazılmış türkcə spam və iki qarışıq səhifə.
- **Qarışıq dilli səhifələr.** Əvvəl rusca hissəsi olan səhifələr bütöv atılırdı (14 sənəd, o cümlədən day.az xəbəri). İndi yalnız yad dildəki sətirlər silinir. Silinən simvolların payı cəmi 0.30%-dir.
- **Təkrar sətir qaydası.** Sətir sayı ilə ölçəndə qayda (30%) 7 yaxşı sənədi atırdı: Sputnik məqaləsi, Milli Məclisin stenoqramı və nəqarətli şeirlər. Onların təkrarlanan simvol payı 1–8% idi. Qayda simvol payına keçirildi (Gopher üsulu, hədd 20%). Nümunədə bu həddi keçən sənəd yoxdur.
- **Xəbər payı.** Sənədlərin ~55%-i xəbər domenlərindəndir (publika.az 684, azadliq.org 633, amerikaninsesi.org 396, report.az 343, metbuat.az 290 və s.). Domen bölgüsü (`routes`) bu payı məhdudlaşdırır. Siyahı bu nümunədən çıxarılıb, 1-ci mərhələdə tam data üzrə yoxlanacaq.
- **Vikipediya rəqəm həddi.** 16 621 məqalədə 0.15 həddi 72 məqaləni atırdı, onların çoxu cədvəlli normal məqalədir (Bişkek, "Xalq qəzeti"). 0.25 həddində 15 qalır. Vikipediya üçün 0.25 seçildi.
- **MinHash** (16 band x 8 sətir). Sözlərin 1%-i dəyişəndə 100/100, 3%-də 85/100, 5%-də 37/100, 10%-də 0/100 tutulur. 300 müstəqil sənəddə səhv tutma yoxdur. Sürət: 1800 sözlük sənəd ~0.7 ms.
- **Normallaşdırma:**
  - kiril ә → latın ə;
  - dırnaqlar və apostroflar vahid formaya salınır;
  - yumşaq defis silinir ("Gürcüs­ ki" → "Gürcüski").
  - JS tokenizer eyni normallaşdırmanı edir; 414 sətirdə Python ilə nəticə 414/414 eynidir (həm yeni, həm Odlar-1.5 tokenizer-i ilə).
- **Tokenizer:**
  - hər rəqəm ayrıca token olur;
  - Simurq üçün 15 xüsusi token ayrılıb (id 0–14);
  - decode itkisizdir.

## Brauzer

Odlar-1.5 üzərində, saytdakı onnxruntime-web 1.20.1 ilə:

| | Köhnə ixrac | Yeni ixrac (KV-keş + ortaq embedding) |
|---|---|---|
| Fayl | 48.6 MB | 40.4 MB (R5: embedding iki dəfə yazılırdı, 8.2 MB) |
| BPC, 16 sınaq mətni | | fp32 1.2237, int8 1.2250 |
| Qısa kontekst | 105 ms/token | 22 ms/token |
| 512 tokenlik kontekst | 1256 ms/token | 120 ms/token (ilk addım daxil) |

Keşli hesablama tam hesablama ilə eynidir: PyTorch-da fərq ~1e-7, fp32 ONNX-də acgöz seçimlə 5 başlanğıcın 5-də mətn hərfbəhərf eynidir. int8-də mətnlər bir müddət sonra ayrılır. Bunun ehtimal olunan səbəbi aktivasiya miqyasının hər çağırışda giriş tensorunun özündən hesablanmasıdır (`DynamicQuantizeLinear`): keyfiyyət eynidir (BPC yuxarıda).

**Odlar-2 ölçüsündə model** (109.5M, təsadüfi çəkilər, eyni ixrac):

- **Fayl:** 110.2 MB, embedding düzəlişi 24.6 MB qənaət edir. Bu, GitHub-un 100 MiB limitindən böyükdür, ona görə K1 qüvvədədir: model faylları Hugging Face-də saxlanılmalıdır.
- **Sürət:**
  - 32 token kontekstdə 30 ms/token;
  - 256 tokendə 49 ms/token;
  - 512 tokendə 80 ms/token;
  - 1000 tokendə 132 ms/token.
  - Əvvəlki keşsiz ölçü 1–7 saniyə/token idi (R1 həll olundu).
- **Uzun prompt:** ilk addımda bütün prompt bir dəfə hesablanır. 1000 tokenlik prompt üçün bu 6.3 saniyədir.

Sayt real Chrome-da (Playwright, başsız) sınandı:

- Odlar-1.5 keşli, Odlar-1 köhnə formatla işləyir;
- UI-dan göndərilən mətn 120 tokenə 3.1 saniyədə yazıldı;
- konsolda xəta yoxdur (sınaqda qəsdən bloklanan şrift faylından başqa).

## Real GPU sınağı (Colab, bir T4, 27 noyabr axşamı)

Kaggle T4x2 növbəsi uzun çəkdiyi üçün sınaq əvvəlcə Colab-da bir T4 ilə edildi.
İki GPU-nun birgə işləməsi (DDP) burada sınana bilməz, o Kaggle-da qalır.

**Tapılan iki səhv (hər ikisi düzəldildi):**

1. **T4-də bf16 seçilirdi.** torch 2.14-də `torch.cuda.is_bf16_supported()` T4 (sm75) üçün də True qaytarır, halbuki T4 bf16-nı aparat səviyyəsində dəstəkləmir.
   - Nəticədə yaddaşa qənaət edən attention işləmirdi, tam attention matrisi saxlanırdı.
   - Sürət 3.2k token/san idi, batch 16 yaddaşa sığmırdı.
   - İndi bf16 yalnız sm80+ GPU-larda seçilir. Əlavə olaraq RoPE cədvəli vektorun tipinə salınır ki, fp16-nı fp32-yə qaldırmasın.
2. **Plandakı "batch 16 T4-ün yaddaşına sığır" fərziyyəsi səhv idi.** `tools/gpu_probe.py` ilə ölçmə (fp16, hər ölçü ayrıca prosesdə):

| Batch x 1024 | Pik yaddaş | Sürət |
|---|---|---|
| 4 | 4.9 GiB | 15.5k token/san |
| 6 | 6.6 GiB | 16.2k |
| 8 | 8.4 GiB | 16.4k |
| 10 | 10.1 GiB | 16.5k |
| 12 | 11.8 GiB | 16.3k |
| 16 | sığmır | |

Odlar-2 üçün batch 8 x grad_accum 8 seçildi. Qlobal batch dəyişmir (65 536 token/addım).

**Sınaq təlimi** (`configs/sinaq_t4x2.yaml`, 109.5M parametr, 300 addım, 19.7M token):

| Addım | train | val | val: vikipediya / veb / xəbər |
|---|---|---|---|
| 0 | 10.506 | 10.465 | 10.442 / 10.477 / 10.477 |
| 100 | 7.957 | 6.593 | 6.181 / 6.810 / 6.787 |
| 200 | 5.996 | 5.592 | 5.198 / 5.828 / 5.751 |
| 300 | 5.293 | 5.152 | 4.726 / 5.398 / 5.332 |

- **Başlanğıc loss:** 10.5, gözlənilən ln(32 000) = 10.37-yə yaxındır.
- **Sürət:** qiymətləndirmə və nümunə yazma ilə birlikdə orta 14.8k token/san.
- **Ümumi vaxt:** 22.2 dəqiqə.
- **WSD:** son addımda öyrənmə sürəti 3e-5-dir (10%).
- **Data:** 200M simvolluq yığım, dublikat yoxlaması, 32k tokenizer və tokenləşdirmə Colab-ın 2 CPU-sunda ~25 dəqiqə çəkdi. Bir epoxa 60.4M token oldu: vikipediya 38%, veb 52%, xəbər 10%.

**Plan üçün nəticə:**

- **Bir T4-də 2.2 mlrd token ~41 saat çəkir.** Plan 49 saat təxmin edirdi.
- **T4x2-də vaxt DDP səmərəsindən asılıdır.** Plandakı 1.8 dəfəlik fərziyyə ilə ~23 saat alınır, amma bu fərziyyə hələ ölçülməyib: Kaggle sınağı bunu göstərəcək.
- **Kaggle-da yeganə GPU seçimi T4x2-dir.** "Save & Run All" versiyası 27 noyabr 22:15-də (Bakı vaxtı) növbəyə düşdü və 28 noyabr 00:45-də hələ başlamamışdı (2 saat 30 dəqiqə). Əsas təlimin hər sessiyasında belə gözləmə ola bilər.
- **Kvota:** həftədə 30 saat, 27 noyabrda 29 saat qalıb.

## Tapılan və düzəldilən səhvlər

- **`odlar/export.py`-nin keyfiyyət yoxlaması** PyTorch modelini dropout açıq halda ölçürdü: `torch.onnx.export` sarğı modulun rejimini bərpa edəndə modeli train rejiminə qaytarır. Nəticədə int8 ONNX PyTorch-dan "yaxşı" görünürdü (fərq -0.03 ... -0.09). Düzəldildi. Əvvəlki nəticə sənədlərində bu rəqəm işlədilməyib.
- **`Pool.imap`** axınla gələn datasetin hamısını arxa planda oxuyub RAM-a yığardı. Hissə-hissə emala keçirildi.

## Açıq qalanlar (növbəti addımlar)

- **Kaggle T4x2 keçid sınağı (DDP):** notebook Kərəmin hesabında qurulub və növbədədir (`krmhseynov/train`, versiya 2). Başlayanda repodan son kodu çəkir.
- **Hub-a real yükləmə və tarixçə sıxılması:** Kərəmin fine-grained token-i ilə sınanmalıdır (K13).
- **Uzun sənəd çəkisi (K9) və kiril dövrü kitabların çevrilməsi (K10):** 1-ci mərhələdə, data v1 ilə.
- **Saytdakı köhnə `odlar15.onnx`:** brauzer keşindəki köhnə JS üçün saxlanılıb. Bir həftə sonra silinə bilər.
