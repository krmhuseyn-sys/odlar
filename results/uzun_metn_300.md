# Uzun mətn testi: 300 token (26 noyabr 2026)

Sual: model 300 token boyunca başlanğıcın mövzusunda qalırmı? Bu test Odlar-2 planındakı "300 tokenlik mətndə mövzudan çıxmama" hədəfinin baza ölçüsüdür.

## Üsul

- 10 neytral başlanğıc, hər model üçün hərəsinə bir mətn, sabit seed (2000 + i).
- Saytın ayarları: temperatur 0.8, top-k 40, təkrar cəzası 1.3.
- `<|endoftext|>` qadağandır: model düz 300 token yazmalıdır. Normal rejimdə 20 mətndən yalnız 8-i 300 tokenə çatır, qalanları tez bitir (Odlar-1.5: 11-300 token, median ~200).
- Qiymət 1-5, bir qiymətləndirici (Claude), aşağıdakı şkala ilə. İnsan qiymətləndirməsi deyil, ilkin ölçüdür.

| Bal | Mənası |
|---|---|
| 1 | mövzu ilk cümlələrdə itir və ya mətn zibilə (siyahı, kateqoriya, cədvəl) çevrilir |
| 2 | mövzuya yaxın sahədə qalır, amma cümlələr arasında əlaqə yoxdur |
| 3 | mövzu əsasən qorunur, ardıcıllıq zəifdir |
| 4 | mövzu qorunur, mətn əsasən ardıcıldır, bir-iki qırılma var |
| 5 | insan yazısı kimi ardıcıl |

## Nəticə

| Model | Orta bal | 3 və yuxarı | Zibilə (siyahı/kateqoriya) çevrilən |
|---|---|---|---|
| Odlar-1.5 | 2.0 | 3/10 | 2/10 |
| Odlar-1 | 1.4 | 1/10 | 6/10 |

Əsas müşahidələr:

1. **Mövzu tez itir.** Odlar-1.5-də əksər mətnlərdə mövzu ilk ~50 token ərzində dəyişir. Model sözü tutur, mətnin istiqamətini saxlamır.
2. **Sürüşmə istiqaməti datanı göstərir.** Odlar-1.5 rəsmi xəbər və bürokratiya dilinə keçir (3/10), bir dəfə dini mətnə, bir dəfə veb zibilinə. Odlar-1 Vikipediya kateqoriya siyahılarına düşür (6/10).
3. **Vikipediya datası təmizlənməyib.** "İstinadlar", "Həmçinin bax", "Xarici keçidlər" bölmələri və kateqoriya sətirləri təlim datasında qalıb. Model bunları məzmun kimi öyrənib. Odlar-2 üçün bu bölmələr kəsilməlidir.
4. **Açar söz üsulu kifayət etmir.** Xəbər/siyasət sözlərini sayan avtomatik ölçü Odlar-1.5-də yalnız 3/20 mətni tutdu. Oxuyanda isə 10 mətndən 3-ü açıq-aşkar rəsmi xəbərə keçir. Bu test oxumaqla və ya daha yaxşı təsnifatçı ilə aparılmalıdır.
5. **Mətnlər tez bitir.** Normal rejimdə mətnlərin yarıdan çoxu 300 tokendən əvvəl `<|endoftext|>` ilə bitir: təlim datasındakı sənədlər qısadır (xəbərlər). Uzun bədii mətnlər bunu da dəyişməlidir.

## Mətnlər

### Odlar-1.5

| # | Başlanğıc | Bal | Qeyd |
|---|---|---|---|
| 0 | Qış gecəsi idi | 1 | uşaqlardan xəbər sitatına, sonra Facebook dedi-qodusuna keçir |
| 1 | Qədim zamanlarda | 2 | tarix üslubu, uydurma faktlar, sonda Vikipediya kateqoriya siyahısı |
| 2 | Xəzər dənizi | 1 | dərhal Mərkəzi Bank, büdcə, nazirlik xəbərinə çevrilir |
| 3 | Azərbaycan mətbəxi | 1 | veb zibili: cədvəl, ingiliscə məhsul adları |
| 4 | Kür çayı | 2 | coğrafiya üslubu qalır, uydurma adalar, Arktikaya sürüşür |
| 5 | Kəndimizin kənarında | 2 | kənd hekayəsi ilə başlayır, dini mətnə (hədis) keçir |
| 6 | Nənəm mənə danışırdı ki | 3 | ailə dialoqu: mövzu qalır, ardıcıllıq zəif |
| 7 | Bahar gələndə | 2 | şəxsi monoloq; bahar bir daha xatırlanmır |
| 8 | Məktəbin ilk günü | 3 | məktəb mövzusu qalır, amma rəsmi xəbər dilində |
| 9 | Dağların başında | 3 | dağ, təbiət, hava: mövzu qalır, cümlələr əlaqəsiz |

### Odlar-1

| # | Başlanğıc | Bal | Qeyd |
|---|---|---|---|
| 0 | Qış gecəsi idi | 1 | dərhal Vikipediya kateqoriya siyahısı |
| 1 | Qədim zamanlarda | 3 | arxeologiya: mövzu qalır, təkrarçı |
| 2 | Xəzər dənizi | 1 | adalar siyahısı |
| 3 | Azərbaycan mətbəxi | 2 | yemək və bitkilər: yaxın, amma mənasız |
| 4 | Kür çayı | 1 | uydurma coğrafiya və tarix |
| 5 | Kəndimizin kənarında | 1 | mənasız cümlələr, sonra kateqoriyalar |
| 6 | Nənəm mənə danışırdı ki | 1 | mənasız sevgi şeiri |
| 7 | Bahar gələndə | 1 | aktyor bioqrafiyası, kateqoriyalar |
| 8 | Məktəbin ilk günü | 1 | kateqoriya siyahısı |
| 9 | Dağların başında | 2 | kənd coğrafiyası, kateqoriyalar |

Tam mətnlər: `eval/long300_texts.json`. Yenidən yaratmaq: `python eval/long300.py` (eyni seed-lər).
