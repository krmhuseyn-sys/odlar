# Odlar-1: son model nümunələri

Ən yaxşı checkpoint (addım 14,500, val loss 3.156). Ayarlar: temperatur 0.8, top-k 40, 80 token.

## "Bakı şəhəri"
> Bakı şəhəri və onun ətrafındakı şəhərlərin sui-qanunu qoruduğunu təsdiq edir.
>
> Tarixi
> Mayklendorfun görməli yerləri
>
> Mayklendorfun görməli yerləri
> Mayklendorfun görməli yerləri

## "Azərbaycan Respublikası"
> Azərbaycan Respublikası Mədəniyyət və Turizm Nazirliyi

## "Nizami Gəncəvi"
> Nizami Gəncəvi adına Azərbaycan Dövlət İncəsənət Universitetinin Filologiya fakültəsini bitirib.
>
> Karyerası
> 1947-ci ildən Azərbaycan Dövlət İncəsənət Universitetinin Fəlsəfə fakültəsini bitirib.
>
> 1930-cu ildə Azərbaycan Dövlət İncəsənət İnstitutunun Fəlsəfə fakultəsinin Fəlsəfə fakultəsində, 1950-ci ilin sentyabr ayından isə Azərbaycan SSR Mədəniyyət Nazirliyi yanında Dövlət Opera və Balet Teatrının direktoru vəzifələrində çalışıb.

## Təhlil
**Nəyi öyrənib:**
- Qrammatika və şəkilçilər demək olar ki, səhvsizdir ("ətrafındakı şəhərlərin", "vəzifələrində çalışıb").
- Vikipediya məqalə quruluşu: "Tarixi", "Karyerası", "Görməli yerləri" başlıqları.
- Real qurumların adları: Mədəniyyət və Turizm Nazirliyi, Opera və Balet Teatrı.
- "Nizami Gəncəvi" sözündən sonra "...adına" yazması məntiqlidir: Vikipediyada bu ad ən çox qurum adlarında keçir.

**Zəif cəhətləri:**
- **Uydurma (hallucination):** "Mayklendorf" adlı mövcud olmayan yer; "sui-qanunu" kimi mənasız ifadə.
- **Ziddiyyət:** eyni şəxs 1947-ci ildə universiteti bitirir, amma 1930-cu ildə artıq orada işləyir. Model ardıcıl fakt yaddaşı saxlamır.
- **Təkrar:** "Mayklendorfun görməli yerləri" üç dəfə, "Fəlsəfə fakultəsinin Fəlsəfə fakultəsində".

**Nəticə:** 13.8M parametr dilin *formasını* (qrammatika, üslub, struktur) öyrənməyə kifayət edir, amma *məzmunu* (faktlar, məntiqi ardıcıllıq) öyrənməyə yox. Bu, Odlar-2-də ölçünü və datanı artırmağın əsas motivasiyasıdır.
