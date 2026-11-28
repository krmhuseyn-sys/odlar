# Pilot: sənəd maskası (K3) və WSD (K4), 28 noyabr 2026 gecəsi

Odlar-2-dən əvvəl iki qərar kiçik modeldə yoxlandı. Üç config yalnız bir seçimlə fərqlənir:

- **`pilot`** (əsas): WSD cədvəli, sənəd maskası yoxdur.
- **`pilot_mask`**: üstəlik sənəd maskası. Hər token yalnız öz sənədinin tokenlərinə baxır.
- **`pilot_cos`**: WSD əvəzinə cosine cədvəli.

**Dizayn:**

- **Model:** Llama quruluşu, 6 qat x 512, 35.7M parametr, kontekst 1024, lüğət 32 000.
- **Təlim:** 6000 addım x 32 768 token = ~197M token, öyrənmə sürəti 6e-4, isinmə 300 addım.
  - WSD-də sürət son 20% addımda 1-sqrt əyrisi ilə enir.
  - Cosine-də bütün təlim boyu enir.
  - Hər iki cədvəl sonda eyni minimuma (10%) çatır.
- **Data:** eyni 174.3M token/epoxa. Vikipediya x2 32%, veb 58%, xəbər 9%.
  - Hər iki platformada data eyni koddan və eyni axından hazırlanıb.
  - Mənbələrin token sayları hərfbəhərf eynidir.
- **Toxumlar:** 1337 (Colab, bir T4) və 2 (Kaggle, hər qaçış ayrıca T4-də).
  - Eyni toxumlu qaçışlar eyni başlanğıc çəkilər və eyni data sırası ilə gedir. Ona görə fərqlər "cüt-cüt" ölçülür: fərq seçimdən gəlir, təsadüfdən yox.
- **Validasiya:** hər mənbədən sabit bloklar, həmişə sənəd daxilində hesablanır (maska ilə).
  - Saytda model yalnız bir mətnə baxır, və maskalı/maskasız öyrədilmiş qaçışlar eyni qayda ilə müqayisə olunur.

## Nəticə (son addım, val loss, az = yaxşı)

| Toxum | pilot (WSD) | pilot_mask | pilot_cos | maska - əsas | cosine - əsas |
|---|---|---|---|---|---|
| 1337 | 3.4463 | 3.4001 | 3.5015 | **-0.0462** | **+0.0552** |
| 2 | 3.4513 | 3.3990 | 3.5041 | **-0.0523** | **+0.0528** |

Mənbə üzrə (toxum 1337 / 2):

| | veb | vikipediya | xəbər |
|---|---|---|---|
| pilot | 3.3837 / 3.3899 | 3.3657 / 3.3704 | 3.5894 / 3.5937 |
| pilot_mask | 3.3504 / 3.3522 | 3.3135 / 3.3127 | 3.5364 / 3.5321 |
| pilot_cos (1337) | 3.4386 | 3.4181 | 3.6477 |

- **Toxum 2-nin pilot_cos rəqəmi:** cəmi val loss (3.5041) Kaggle konsolunda çap olunan cüt fərqdən (+0.0528) hesablanıb. Mənbə üzrə rəqəmlər ekranda qalmayıb.
- **Toxumlar arası səpələnmə:** əsas qaçışın iki toxumu 0.005 fərqlənir; effektlərin cüt fərqləri 0.046–0.055-dir. Hər iki effektin işarəsi iki toxumda da eynidir və bütün üç mənbədə özünü göstərir.
- **3-cü toxum:** pilot s3 bitdi, pilot_mask s3 və pilot_cos s3 gedirdi. Kompüter söndürüldüyü üçün tamamlanmadı, onların nəticəsi yoxdur.

## Qərarlar

**K3, sənəd maskası: QƏBUL.**
- **Fayda:** val loss -0.046 / -0.052. Bu, kiçik modelin 197M tokenində təxminən sonuncu 10% addımların verdiyi qazanc qədərdir.
- **Qiymət:** sürət 35.7k-dan 31.9k token/san-a düşür (-11%, Colab-da tək qaçışda ölçülüb). Kaggle-da iki qaçış paralel gedirdi və CPU-nu paylaşırdı, ona görə oradakı sürət rəqəmləri müqayisə üçün etibarsızdır.
- **Odlar-2-yə təsiri:** ~24 saatlıq təlimə təxminən +3 saat deməkdir. Bu, həftəlik 30 saatlıq Kaggle kvotasına hələ sığır.

**K4, WSD: QƏBUL** (cosine-dan +0.053 / +0.055 yaxşı).
- Aralıq addımlarda cosine öndə idi, məsələn addım 4000-də 3.624 / 3.713, çünki onun sürəti daha erkən enir. Sonda WSD qalib gəlir.
- WSD-nin əlavə üstünlüyü: sabit fazanın istənilən nöqtəsindən qısa enmə ilə ara model almaq olur.

`configs/odlar2.yaml`-da `doc_mask: true` və `schedule: wsd` yazıldı.

## Məhdudiyyətlər

- **Ölçü:** model 35.7M-dir, Odlar-2 isə 109.5M olacaq. Token sayı da Odlar-2-nin onda biridir. Effektlərin işarəsinin dəyişəcəyini gözləmirəm, amma ölçüsü fərqli ola bilər.
- **Data:** yalnız bir qarışıq yoxlanıb. Bədii kitablar (uzun sənədlər) hələ datada yoxdur. Sənəd maskasının faydası sənəd uzunluğundan asılıdır, ona görə 1-ci mərhələdən sonra Odlar-2-nin ilk 1000 addımında val loss izlənəcək.
- **Toxumlar:** üç deyil, iki tam toxum var.

## Gecə hadisəsi (dərs)

Kaggle interaktiv sessiyası brauzer tabı başqa səhifəyə aparılandan sonra ləğv oldu ("Cancelled"). Toxum 2-nin ilk ~1.5 saatlıq işi itdi və yenidən başladıldı.

Qayda:
- Kaggle-da təlim gedən redaktor tabına toxunulmur.
- Başqa Kaggle səhifəsi ayrıca tabda açılır.
- İnteraktiv sessiyanın faylları sessiya ilə birlikdə silinir. Odlar-2-də checkpoint-lər Hugging Face-ə yazılır (`hub_repo`), ona görə orada bu itki baş verməz.

Başqa müşahidələr:
- **Arxa plan versiyası:** "Save & Run All" 27 noyabr 22:15-dən səhərə qədər növbədə qaldı.
- **İnteraktiv sessiya:** gecə iki dəfə 2 dəqiqə ərzində açıldı.
