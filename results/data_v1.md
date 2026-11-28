# 1-ci mərhələ: data v1 (28 noyabr 2026)

Odlar-2-nin təlim datası. Bu sənəddə hər mənbənin ölçüsü, tapılan problemlər və onlara görə verilən
qərarlar var. Kod: `odlar/data.py`, `odlar/text.py`. Config: `configs/odlar2.yaml`.
Rəqəmlər ölçülüb; "təxmini" yazılanlar nümunədən hesablanıb.

## Mənbələr

| Mənbə | Həcm | Necə ölçülüb |
|---|---|---|
| FineWeb-2 azj_Latn | 7.29M sənəd, 28.9 GB mətn (~26 mlrd simvol) | 3 faylın bütün URL-ləri; mətn payları 12 təsadüfi sətir qrupunda (46k sənəd) |
| AzTC-full | 7.50 mlrd simvol, 17 alt mənbə | tam oxunub |
| Kitablar (LocalDoc/books_dataset) | 2764 kitab, 1.03 mlrd simvol | tam oxunub, 52 772 parça (20k simvol) |
| Vikipediya az | ~360M simvol | tam |
| Vikimənbə az | 30.8M simvol | tam |

**AzTC-full-un tərkibi** (simvol): musavat 1371M, ANL (mətbuat arxivi) 1317M, books 1033M,
publika 610M, axar 495M, report 464M, oxu.az 437M, legislation 409M, wikipedia 348M, sesqazeti 302M,
azadliq 261M, medeniyyet 150M, apasport 142M, tehsil 94M, valyuta 39M, marja 29M, muselmanlar 3M.

- **Xəbər/mətbuat hissəsi ~5.46 mlrd simvoldur**, yəni AzTC-nin 73%-i.
- **Kitablar və Vikipediya** ayrı mənbə kimi götürüldüyü üçün AzTC-dən atılır.

## Tapılan problemlər və qərarlar

### 1. FineWeb-2-də qumar spamı

FineWeb-2 azj-də kazino və mərc saytlarının SEO mətnləri çoxdur: 1xBet, Mostbet, Pin-Up, "bonus",
"slot oyunları". Saytın adı çox vaxt adi görünür, ona görə domenlə tutulmur. Qumar sözlərinin sıxlığına
görə ölçülüb (1000 sözə 3-dən çox və ən azı 3 dəfə):

| Səbət (FineWeb-2) | Simvol payı | Spam payı |
|---|---|---|
| xəbər domenləri | 23% | 7% |
| uzun (8000+ simvol), xəbər deyil | 57% | **80%** |
| qalan veb | 20% | 25% |

- **Spamın ümumi payı.** FineWeb-2 azj mətninin **~52%-i qumar spamıdır**.
  - Köhnə config uzun səhifələri iki dəfə təkrarlayırdı (K9). Onların 80%-i spam olduğu üçün təkrar ləğv edildi.
- **Qayda.** `max_gambling` `DEFAULT_RULES`-a əlavə olundu: atılma səbəbi "qumar", bütün mənbələrə tətbiq olunur.
- **İlk ifadənin səhvləri.** İlk variant "mübahisə" (içində "bahis"), "mərcan", "mərci plov", "rulet" (yemək) sözlərini və kompüterdəki "slot"-u tuturdu.
  - Düzəliş: söz əvvəli şərti qoyuldu, "mərc" yalnız mərc mənalı şəkilçilərlə tutulur, "slot" yalnız "slot oyun/maşın" birləşməsində.
- **Səhv atılmalar.** Vikipediyanın 196 158 məqaləsindən ~25-i atılır, məsələn "Las-Veqas" və "Kart" (pokerlə bağlı). Vikimənbədə səhv atılan yoxdur.

### 2. Maşın tərcüməli saytlar

Xarici saytların `az.` alt domenləri avtomatik tərcümə olunmuş mətndir: az.jf-vnsbento.pt,
az.military-review.com, az.delachieve.com, az.urbandictionary.com və s.

- **Pay:** qalan vebin 13%-i.
- **Qayda:** `drop_domain_regex` ilə atılır (səbəb "domen xaric").
- **İstisnalar:** az.armradio.am (insan yazısı xəbər xidməti) və az.ambafrance.org.
- **az.wikipedia.org** da bu qayda ilə gedir, çünki onsuz da Vikipediyanın dublikatıdır.

### 3. Xəbər domenləri

Köhnə 34 domenlik siyahı FineWeb-2 sənədlərinin cəmi 27.7%-ni xəbər kimi tanıyırdı.

- **Yeni qayda** (`domain_regex`): genişləndirilmiş siyahı (~90 domen) və ad əlaməti (xeber, news, press, info, media, qazet, sport, tv, report...).
  - "forum" olan domenlər istisnadır, məsələn azinforum.az.
- **Nəticə:** 3 faylın 7.29M sənədində xəbər payı sayca 52.9%, simvolca 23%-dir. Xəbər yazıları qısa olduğu üçün simvol payı azdır.

### 4. Kitablarda iki sütunun qarışması

Kitab korpusu PDF-dən çıxarılıb və iki sütunlu səhifələrdə sətirlər qarışıb. Sol sütunun bir sətri
ilə sağ sütunun bir sətri növbələşir. Təsadüfi 24 parçanın 10-u belədir. Nümunə:

> ...geniş mənəvi-estetik rezonans **baycanda** sovet-kommunist ideologiyasının... **Azərideyaları** XX əsrin...

Burada sətir sonunda bölünmüş "Azər-baycanda" o biri sütunun "ideyaları" sözü ilə birləşib.
OCR yoxlaması (tanınmayan söz payı) bunu tutmur: qarışıq parçalarda bu pay 0.05–0.07-dir,
təmiz kitab parçalarında da təxminən eynidir.

- **Yeni yoxlama `column_joints`.** Lüğətdə olmayan bir söz m axtarılır ki, m = p + r olsun və r lüğətdə olsun. Sonra yaxın 30 söz içində elə f axtarılır ki, p + f lüğətdə olsun ("azər" + "baycanda").
- **Nəticə 1000 sözə düşən izdir.** Ölçülər:

| Mənbə | Sənəd | 1000 sözə iz > 1 |
|---|---|---|
| Vikipediya, veb, xəbər, AzTC, OSCAR, Vikimənbə | 5 700+ | 0 (p99 = 0) |
| kitab (ümumi) | 153 | 15% |
| bədii | 156 | 42% |
| dilçilik və ədəbiyyatşünaslıq | 156 | 26% |
| siyasət | 154 | 54% |

- **Əl ilə yoxlama.** 24 əl ilə etiketlənmiş parçada yoxlama 10 qarışıq parçanın 9-nu tutdu. Tutmadığı defissiz şeirdir: şeirdə sözlər bölünmür, iz qalmır.
  - Bir parçanı əl ilə "təmiz" saymışdım, amma yoxlama onda real birləşmələr tapdı: parçanın başqa yeri qarışıq idi.
- **Qayda:** `max_joints: 0.5`, atılma səbəbi "sütun".

**Kitabların tam ölçüsü** (52 772 parça):

| Səbət | Hamısı | OCR-dan sonra (0.12) | Sütundan sonra (0.5) |
|---|---|---|---|
| kitab (digər) | 644M | 568M | 374M |
| bədii + folklor | 81M | 66M | **47M** |
| dilçilik + ədəbiyyatşünaslıq | 75M | 64M | 38M |
| siyasət | 228M | 191M | 105M |
| **cəmi** | 1030M | 889M | **564M** |

- **Kitabların payı.** Kitab mətninin 45%-i atılır.
- **Kitab səviyyəsində bölgü.** 2536 kitabın 1342-də heç bir iz yoxdur, 858-də parçaların yarıdan çoxu izlidir.
- **Qalan risk.** İzli kitabların "təmiz" görünən parçaları defissiz qarışıq ola bilər, məsələn şeir. Bu qalıq risk kimi qeyd olunur.

### 5. OSCAR çıxarıldı

`azerbaijani-pretrain-corpus`-un OSCAR sətirlərində URL yoxdur (yalnız `text` və `source`).
Ona görə xəbər payını idarə etmək mümkün deyil. Üstəlik OSCAR FineWeb-2 ilə eyni Common Crawl
mənbəyidir, spamdan təmizlənmiş FineWeb-2 isə kifayət qədər həcm verir.

### 6. Texniki düzəlişlər

- **Yaddaş.** Yığım prosesləri `spawn` ilə başlayır.
  - Əvvəl `fork` köməkçi proseslərə əsas prosesin əvvəlki mənbələrdən qalmış yaddaşını kopyalayırdı və 7 GB-lıq maşında proses öldürüldü.
  - Marşrut və seçmə (`keep_frac`) indi əsas prosesdə edilir: atılan sənəd köməkçiyə ümumiyyətlə göndərilmir.
- **Proses çıxışı.** Yarımçıq oxunmuş HF axını (datasets 5.2 + hf-xet) proses çıxışında asılı qalır, `it.close()` kömək etmir.
  - `python -m odlar.data` iş bitəndə `os._exit(0)` ilə çıxır; hər şey artıq diskdədir.
  - Bu olmasa, Kaggle hüceyrəsi data hazır olduqdan sonra da bitməyəcəkdi.
- **Kiçik miqyasda tam sınaq.** Hər səbət 3M simvol ilə 4 addım keçdi: yığım, MinHash, tokenizer, `.bin`. Testlər 22/22.

## Büdcə (bir epoxa, təxmini)

Bir parça ×N ilə N dəfə təkrar olunur.

| Səbət | Unikal | Təkrar | Epoxada |
|---|---|---|---|
| Vikimənbə (bədii və digər) | ~31M | ×4 / ×1 | ~105M |
| kitab | 374M | ×1 | 374M |
| bədii (kitab) | 47M | ×4 | 188M |
| kitab_dil | 38M | ×2 | 76M |
| kitab_siyasi | 105M | yarısı | 52M |
| Vikipediya | ~330M | ×2 | 660M |
| AzTC xəbər | 5.46 mlrd-ın 13%-i | ×1 | ~710M |
| AzTC qanun | 409M-in yarısı | ×1 | ~205M |
| AzTC mədəniyyət, təhsil, din | 247M | ×1 | 247M |
| FineWeb-2 xəbər | ~6 mlrd-ın 15%-i | ×1 | ~850M |
| FineWeb-2 veb (spamsız) | ~3.0 mlrd | ×1 | ~3.0 mlrd |
| FineWeb-2 uzun (spamsız) | ~2.0 mlrd | ×1 | ~2.0 mlrd |
| **cəmi** | **~7.9 mlrd** | | **~8.4 mlrd simvol, ~1.96 mlrd token** |

## Qapı (planda) və vəziyyət

| Şərt | Hədəf | İndi | Qərar |
|---|---|---|---|
| Token | ≥ 2.2 mlrd/epoxa | ~1.96 mlrd | 33 500 addım ~1.15 epoxa deməkdir. Bu qədər təkrar zərərsizdir (Muennighoff və b., 2023: 4 epoxaya qədər fərq çox kiçikdir). Şərt "≤ 1.3 epoxa" kimi yenilənir. |
| Xəbər payı | ≤ 20% | ~18.5% | keçir |
| Bədii (unikal) | ≥ 150M simvol | **~72M** (Vikimənbə ~25M + kitab 47M); dilçiliklə ~110M | **keçmir**: Kərəmin qərarı lazımdır |

## Növbəti addım

1. Kitabların lisenziyası (CC BY-NC) barədə qərar. Daxil olsa, Odlar-2 qeyri-kommersiya qalır.
2. Bədii qapısı barədə qərar: əlavə bədii mətn və ya qapını endirmək.
3. HF tokeni (K13) və repolar: `krmhus/odlar-2-data` (gizli dataset), `krmhus/odlar-2` (gizli model).
4. Kaggle CPU sessiyasında tam hazırlıq: `python -m odlar.data configs/odlar2.yaml`, təxminən 1–2 saat.
