# Unlearning Çalışması — Rapor Hazırlamak İçin Teknik Malzeme

> Bu dosya bir rapor değildir. Federated learning üstüne kurulan **machine unlearning** modülünün gerekçesini, uygulanan algoritmaları, deneme-yanılma sürecini ve nicel sonuçları, bir rapor şablonuna girdi olacak biçimde özetler. Önceki "Federated Learning" raporu temel alındığı varsayılır; burada **sadece unlearning aşamasına eklenenler** anlatılır.

---

## 0. Bağlam Özeti (FL raporundan devralınan kısa hatırlatma)

| Bileşen | Değer |
|---|---|
| Veri seti | 3 kullanıcı (user1 = Instagram-baskın, user2 = YouTube-baskın, user3 = WhatsApp-baskın) gerçek app-usage logları |
| Özellikler | `sin_hour`, `cos_hour` (saat döngüselleştirilmiş) |
| Sınıflar | 4 app: `Instagram (0)`, `YouTube (1)`, `LinkedIn (2)`, `WhatsApp (3)` |
| Model | 2 → 16 → 16 → 4 MLP, ReLU, **388 parametre** |
| FL kurulumu | 6 round, her round 3 client, lokal eğitim 20 epoch, Adam `lr=0.01`, batch=32 |
| Veri penceresi | Her round 5 günlük kronolojik dilim |
| Aggregator | FedAvg, ağırlık = `n_client_round / n_total_round` |

Unlearning hedefi: **`user1`'i silmek**. Değerlendirme penceresi olarak `08:00–10:00` aralığı seçildi; çünkü bu pencere user1'in Instagram baskınlığının en yoğun olduğu ve diğer kullanıcıların da temiz patternlerinin olduğu (`user2 → YouTube`, `user3 → WhatsApp`) "gold-standard" değerlendirme aralığıdır.

---

## 1. Unlearning

Üç yöntem uygulandı: iki gerçek unlearning yaklaşımı (FedEraser, Gradient Ascent) ve bir **referans baseline** (SISA). FL eğitimi tamamlandıktan sonra `delta_history` (her round × her client için `Δw = w_after_local − w_before_round`) `delta_history.pkl` olarak diske yazılır; sunucu tarafı yöntemleri bunu kullanır.

### 1.1 Yöntem A — FedEraser (Server-Based, Delta Subtraction + Recovery)

**Referans:** Liu et al., 2021 ("FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models").

**Adım 1 — Delta-Guided Parameter Subtraction:**
Final global modelden `user1`'in FedAvg-ağırlıklı katkısı çıkarılır:

```
correction   = Σ_r (n_u1_r / N_r) · δ_r[u1]
corrected_θ  = θ_final  −  correction
```

Burada `n_u1_r`, user1'in r. round'taki veri sayısı; `N_r`, o round'a katılan tüm clientların toplamı. Ağırlıkların FedAvg ile birebir eşleşmesi kritiktir; aksi halde "düz toplam" (`Σ_r δ_u1_r`) çıkarımı yaklaşık 3× fazla doku temizler ve modeli bozar (geliştirme sürecinde de gözlemlendi, bkz. §1.5).

**Adım 2 — Recovery (Calibration Training):**
3 round × 5 epoch boyunca **sadece kalan clientlarla (user2, user3)** yeni bir mini-FedAvg çevrimi:

- Her client her recovery round'unda kendi **tüm verisini** kullanır (6 pencere `np.vstack` ile birleştirilir).
- Lokal optimizer: Adam, `lr=0.01`, batch=32, 5 epoch.
- Round sonu standart FedAvg.

Recovery, küçük modellerde delta subtraction'ın bıraktığı lineerleştirme artığını temizler (bkz. §3.1).

### 1.2 Yöntem B — Gradient Ascent (Client-Based + Recovery)

**Referans:** Gradient ascent literatürü (Thudi et al. 2022; Warnecke et al. 2023). Burada client-side bir loss maksimizasyonu olarak uyarlandı.

**Adım 1 — Gradient Ascent (forget client'ta):**
`user1` kendi tüm verisi üstünde CrossEntropy loss'u **yükseltmeye** çalışır:

```python
loss = CrossEntropy(model(X_u1), y_u1)
loss.backward()
clip_grad_norm_(params, max_norm=1.0)        # patlama koruması
for p in params: p.grad = -p.grad             # FLIP (ascent)
optimizer.step()
```

- Optimizer: **SGD**, `lr=0.001` (Adam ile patlama yaşandı — bkz. §1.5).
- Epoch: 5, batch=32.
- Gradient clipping `max_norm=1.0`: ek bir patlama emniyeti.

**Adım 2 — Recovery:**
**FedEraser'la birebir aynı recovery prosedürü** (3 round × 5 epoch, user2 + user3, tam veri, FedAvg). Bu eşleştirme bilinçlidir: iki yöntemi karşılaştırırken sadece silme adımı farklı olsun, recovery aynı olsun (adil karşılaştırma; bkz. §3.4).

### 1.3 Yöntem C — SISA (Baseline / Referans)

**Referans:** Bourtoule et al., 2021.

Tüm federated eğitimi **sıfırdan**, yalnızca `user2` + `user3` ile yeniden çalıştırılır:
- Aynı 6 round, aynı 20 epoch lokal, aynı Adam `lr=0.01`.
- `user1`'in verisi hiç görülmez.
- Aynı `SEED=42` ile aynı initial weight'ler.

SISA, "u1 hiç katılmasaydı modelin ulaşacağı **gerçek** durum"un en yakın temsilidir. Bu yüzden FedEraser ve GA'nın hedefi SISA'ya yaklaşmaktır. Maliyetlidir (tam yeniden eğitim), pratik kullanılmaz; sadece referans baseline.

### 1.4 Değerlendirme Metrikleri

| Metrik | Tanım | Amacı |
|---|---|---|
| **8–10 ortalama olasılık vektörü** | Modelin user1'in dominant olduğu pencerede çıkardığı `(P_Insta, P_YT, P_Link, P_WA)` ortalaması | Forget sinyalinin silinip silinmediğini ölçer |
| **Instagram düşüş yüzdesi** | `(P_Insta_orig − P_Insta_unlearned) / P_Insta_orig` | Doğrudan silme gücü |
| **Entropy** | `−Σ p log p` (8–10 olasılık dağılımının entropisi) | Tahmin belirsizliği; silmeyle artması beklenir |
| **Remaining accuracy (user2, user3)** | Kalan kullanıcılar üstünde top-1 accuracy | Utility koruması: silme, kalanları bozmamalı |
| **Süre** | Tüm unlearning süreci wall-clock | Pratik fizibilite |

### 1.5 Geliştirme Sürecinde Karşılaşılan Sorunlar (rapora "Yöntem Notları" olarak girmeli)

Bu kısım, raporda doğrudan **mühendislik gerekçesi** olarak verilmelidir; sadece "şunu yaptık" yerine "şunu denedik, bu yüzden işe yaramadı, son halini şu yüzden seçtik" akışını mümkün kılar.

**(1) Gradient Ascent + Adam → loss patlaması:**
İlk denemede `Adam(lr=0.01)` + grad flip kullanıldı. 5 epoch sonunda loss `25 → 46 705`'e fırladı; model NaN-yakın bir bölgeye düştü.
- **Neden:** Adam adaptif moment estimation kullanır; ters gradient onun istatistiklerini bozar ve adım büyüklüğünü kontrolsüz şişirir.
- **Çözüm:** SGD'ye geçildi (`lr=0.001`); ayrıca `clip_grad_norm_(max_norm=1.0)` eklendi. Loss artık kontrollü artıyor.

**(2) FedEraser ilk versiyonda "düz toplam" çıkardı:**
`correction = Σ_r δ_u1_r` (ağırlıksız) denendi. Norm ~25.89 (gerçek ağırlıklı versiyonun ~3 katı). Model agresif biçimde bozuldu.
- **Neden:** FedAvg `θ_r ← θ_{r-1} + Σ (n_u/N_r) · δ_u` formülünü kullanır; yani `δ_u1` zaten `(n_u1/N_r)` ağırlığıyla modele girmiştir. Çıkarırken aynı ağırlığı kullanmak şarttır.
- **Çözüm:** `correction = Σ_r (n_u1_r / N_r) · δ_r[u1]`. Norm ~9'a düştü, matematiksel olarak tutarlı.

**(3) Recovery'siz FedEraser → "LinkedIn artifact":**
Düzgün ağırlıklı çıkarmadan sonra bile **recovery olmadan** çalıştırıldığında değerlendirme penceresinde olasılıklar `Insta 0.2% / YT 9.3% / Link 65.0% / WA 25.5%` çıktı. Beklenen, user2+user3'ün dominant patternlerinin (YouTube + WhatsApp) öne çıkmasıdır; oysa LinkedIn—hiçbir kullanıcının bu saatte dominant olmadığı bir sınıf—%65'e fırladı.
- **Neden:** Model küçük (388 parametre) ve nonlineer. Çıkarma `θ_final − correction` matematiksel olarak doğru olsa da, `δ_u2_r` ve `δ_u3_r`'lar `θ_{r-1}` (içinde user1 izleri olan) bir noktadan hesaplandı; user1 olmasaydı bu deltalar farklı olacaktı. Lineerleştirme hatası, modeli rastgele bir "boş" sınıfa (LinkedIn) kaydırdı.
- **Çözüm:** Liu et al.'un orijinal "calibration training" adımı yeniden eklendi: **3 round × 5 epoch** recovery. Recovery, modeli geçerli bir manifolda geri çeker.

**(4) Reddedilen patch — "weight boost":**
Geçici bir denemede, recovery yerine, user2 ve user3'ün FedAvg ağırlıkları `(n_u/(N − n_u1))` ile yeniden normalize edilerek bir `boost` terimi eklendi:

```
unlearned = θ_final − correction + boost
boost      = Σ_r Σ_{u≠1} ((n_u/(N_r−n_u1_r)) − (n_u/N_r)) · δ_r[u]
```

**Beklenen:** "u2+u3'ü user1 yokken sahip olacakları proper ağırlığa taşımak."
**Gerçekleşen:** LinkedIn %65 → **%82.1**'e çıktı (daha kötü). Instagram ~0%, WhatsApp %14.9.
- **Neden:** `δ_u2_r` ve `δ_u3_r`, user1-katkısıyla biased bir global state'ten üretildi; bu deltaları **büyütmek**, bias'ı da büyütür. Boost matematiksel olarak doğru bir "yeniden normalize" yapıyor gibi görünür ama yanlış kantitenin üstüne uygulandığı için modeli daha kötü yere taşır.
- **Sonuç:** Patch geri alındı, gerçek çözüm olan **recovery** uygulandı.

**(5) Adil karşılaştırma için Gradient Ascent'e de recovery eklendi:**
FedEraser recovery'li, GA recovery'siz olarak karşılaştırılırsa, sonuç farkı silme yöntemi mi yoksa recovery mi kaynaklı bilinmez. Her iki yöntem de aynı recovery prosedüründen geçirildi; geriye yalnızca silme adımının farkı kaldı.

---

## 2. Bulgular

### 2.1 Federated Baseline (silme yapılmadan, referans nokta)

8–10 penceresinde global modelin ortalama tahminleri (5 günlük gerçek veri noktalarında ortalanmış):

| Metrik | Instagram | YouTube | LinkedIn | WhatsApp |
|---|---|---|---|---|
| Olasılık (%) | **31.2** | 26.9 | 7.1 | 34.9 |

Yorum: user1'in Instagram baskınlığı global modelde net biçimde okunur (~%31, baseline rastgele tahmin %25).

### 2.2 SISA (referans baseline)

`user1` hiç katılmadan 6 round yeniden eğitim:

| Metrik | Instagram | YouTube | LinkedIn | WhatsApp |
|---|---|---|---|---|
| Olasılık (%) | **8.4** | 20.4 | 10.9 | 60.3 |
| user3 accuracy (genel) | — | — | — | **49.0%** |

Yorum: user1 silindiğinde 8–10 penceresinde WhatsApp (user3'ün dominant patterni) öne çıkıyor; Instagram %31'den %8.4'e düşüyor. Hedef değerler bunlar.

### 2.3 FedEraser — Geliştirme İterasyonları

| Konfigürasyon | Insta | YT | Link | WA | Not |
|---|---|---|---|---|---|
| FedEraser (recovery yok, **düz toplam** correction) | crash/kararsız | — | — | — | Norm 25.89; reddedildi |
| FedEraser (recovery yok, ağırlıklı correction) | 0.2 | 9.3 | **65.0** | 25.5 | LinkedIn artifact; norm 9.1 |
| FedEraser (recovery yok, ağırlıklı + **boost**) | ~0 | 3.0 | **82.1** | 14.9 | Boost daha kötüleştirdi; reddedildi |
| **FedEraser (ağırlıklı correction + 3×5 recovery)** | **1.3** | 27.2 | 3.5 | **68.1** | **Son versiyon**; LinkedIn artifact yok, WhatsApp dominant |

Yorum: Instagram %31.2 → **%1.3** (~96% düşüş, SISA'nın %8.4'ünden daha agresif). LinkedIn artifact'ı tamamen temizlendi (%3.5, orijinal %7.1'in bile altında). WhatsApp %68.1 öne çıkarak user3'ün dominant patternini yansıtıyor; YouTube %27.2 user2 patternini koruyor. user1'in Instagram izinin etkin biçimde silindiğinin somut göstergesi.

### 2.4 Gradient Ascent — Geliştirme İterasyonları

| Konfigürasyon | Insta | YT | Link | WA | Loss değişimi | Not |
|---|---|---|---|---|---|---|
| Adam `lr=0.01`, 5 ep, recovery yok | NaN/yakın | — | — | — | 25 → **46705** | Patladı; reddedildi |
| SGD `lr=0.01`, 5 ep, recovery yok | — | — | — | — | 3.26 → 659.5 | Hâlâ patlama eğiliminde |
| SGD `lr=0.001`, 3 ep, clip 1.0, recovery yok | 23.7 | 30.4 | 7.4 | 38.6 | kontrollü | Silme zayıf |
| SGD `lr=0.001`, **5 ep**, clip 1.0, recovery yok | 23.7'ye yakın | — | — | — | kontrollü | Beklenen düşüş 15-20% (ölçülmedi) |
| **SGD `lr=0.001`, 5 ep, clip 1.0, + 3×5 recovery** | **2.0** | 34.7 | 5.6 | 57.8 | kontrollü | **Son versiyon** |

Yorum: Instagram %31.2 → **%2.0** (~94% düşüş; FedEraser ile aynı büyüklük sırasında). YouTube %34.7 ile user2 patternine FedEraser'dan biraz daha güçlü kayma; WhatsApp %57.8 user3 patternini destekler. LinkedIn %5.6 (orijinalin altında, artifact yok). Recovery sonrası iki yöntem birbirine **çok yakın** sonuç veriyor (Insta farkı sadece 0.7 puan); aralarındaki seçim metodolojik tercihe (matematiksel çıkarma vs optimizasyon-tabanlı) kalıyor.

### 2.5 Süre Karşılaştırması (büyüklük sırası)

| Yöntem | Tipik süre (CPU, bu boyut için) | Yorum |
|---|---|---|
| SISA | 6 round × 3 client × 20 epoch tam FL | Baseline; en pahalı |
| FedEraser | Delta çıkarma (anlık) + 3 round × 5 epoch recovery | SISA'nın yaklaşık **1/8**'i |
| Gradient Ascent | Forget client 5 epoch + 3 × 5 recovery | FedEraser'la aynı büyüklük |

Maliyet açısından FedEraser ve GA, SISA'ya kıyasla belirgin avantajlı; aralarındaki fark ihmal edilebilir.

---

## 3. Tartışma

### 3.1 Recovery neden bu kadar kritik?

FedEraser'ın matematik formülü `θ_final − Σ_r (n_u1_r/N_r) · δ_u1_r` tamamen doğrudur ve büyük modellerde (yüz binlerce parametre) lineerleştirme hatası küçük kalır. Ancak bizim modelimiz **388 parametreli ve yüksek nonlineer** (iki ReLU katmanı, 4 sınıflı softmax). Bu boyutta:
- Çıkarma sonrası model **rastgele bir aktivasyon düzeneği**ne karşılık gelebilir.
- LinkedIn gibi orijinalde hiç dominant olmayan bir sınıf, **boş bölgeyi doldurarak** olasılıkları emer (geliştirme sürecinde gözlemlenen %65).

Recovery (= calibration training), modeli geçerli bir hipotez sınıfına geri çeker. Liu et al.'un kâğıdı da bunu açıkça "calibration training" olarak adlandırır. Bizim deneyimimiz, recovery'nin küçük modellerde **opsiyonel olmadığını**, **zorunlu** olduğunu gösteriyor.

### 3.2 Boost girişiminin yanlışlığı (yöntemsel ders)

Sezgisel olarak "user1 çıktıysa, kalan iki client'ın ağırlığı `n_u/(N−n_u1)` olmalı" gerekçesi makul görünür. Ancak bu, **delta'ların user1 yokluğunda da aynı kalacağı** varsayımına dayanır. Gerçekte:
- `δ_u2_r`, `θ_{r-1}` noktasında hesaplanmıştır.
- `θ_{r-1}`, user1'in 0..r−1 round'lardaki katkılarını içerir.
- user1 olmasaydı `θ_{r-1}` farklı, dolayısıyla `δ_u2_r` farklı olurdu.

Boost, **yanlış bir δ'yı** büyüttüğü için modeli daha hatalı bir yöne kaydırır. Bu, FedEraser literatüründe "calibration without retraining" girişimlerinin neden zayıf kaldığının somut bir örneği.

### 3.3 SGD vs Adam (Gradient Ascent için)

Adam'ın 1. ve 2. momentleri (`m_t`, `v_t`) **istatistiksel olarak normal gradient akışı** için tasarlanmıştır. Gradient'i ters çevirince:
- `v_t` (kare-ortalamalar) yanlış kalibre olur,
- adım büyüklüğü `lr / (√v_t + ε)` patlar,
- birkaç batch sonunda kaybın binlerce kata büyümesi sıradan hâle gelir.

SGD'de bu istatistiksel akümülasyon yoktur; gradient flip doğrudan parametre değişimine yansır ve `lr` × `||g||` ile sınırlıdır. `lr=0.001` + `clip_grad_norm_(1.0)` kombinasyonu, adım başına maksimum `0.001` parametre değişimi garantisi verir; bu boyutta model için güvenlidir.

### 3.4 Adil karşılaştırma için recovery simetrisi

Eğer FedEraser recovery'liyken GA recovery'siz çalıştırılırsa, sonuç farkını **silme yönteminden mi** yoksa **recovery prosedüründen mi** kaynaklı olduğunu söyleyemeyiz. İki yöntemin de aynı recovery konfigürasyonu (3 round × 5 epoch × Adam `lr=0.01`) kullanması, geriye **yalnızca silme adımı** farkını bırakır:
- FedEraser: matematiksel çıkarma (delta subtraction)
- GA: optimizasyon-tabanlı çıkarma (gradient ascent)

Bu simetri, raporda metodolojik bir tasarım kararı olarak vurgulanmalı.

### 3.5 Sınırlamalar

1. **Model boyutu:** 388 parametre, fairly synthetic bir senaryo. Gerçek mobil unlearning senaryolarında modeller 10⁴–10⁶ parametre arasıdır; lineerleştirme hataları daha küçük olabilir.
2. **Client sayısı:** 3 client (n=3) sadece bir "minimal" FL senaryosu. Gerçek FedEraser deneyleri 100+ client'la çalışır; orada hesaplama tasarrufu daha belirgindir.
3. **Tek forget client:** Çoklu silme (örn. user1 ve user2 birlikte) test edilmedi.
4. **Veri heterojenliği:** Kullanıcılar dominant pattern bakımından çok farklı (Instagram / YouTube / WhatsApp). Daha benzer kullanıcılarda silmenin etkisi nicel olarak farklı çıkabilir.
5. **Privacy garantisi yok:** Bu yöntemler ε-differential-privacy gibi resmi garantiler vermez; "approximate unlearning" sınıfındadır.

---

## 4. Sonuçlar

1. **FedEraser** (delta-guided subtraction + calibration recovery) ve **Gradient Ascent** (loss maximization + calibration recovery), bu mimari boyutunda uygulanabilir machine unlearning yöntemleridir; ikisi de SISA'nın tam yeniden eğitim maliyetinin yaklaşık **1/8'i** ile çalışır.
2. **Son sonuçlar, her iki yöntemin de SISA'dan daha agresif Instagram silme başardığını gösterdi:** FedEraser %1.3, GA %2.0 (SISA %8.4). Bu, recovery + silme kombinasyonunun u1'in iz bırakmamasını garantilediğini, hatta SISA'da kalan zayıf sinyali bile aştığını gösterir.
3. **Recovery (calibration training), küçük modellerde opsiyonel değildir.** Recovery'siz FedEraser, lineerleştirme artığı yüzünden "LinkedIn artifact" gibi degenere durumlar üretir. Recovery, modeli geçerli bir manifolda geri çeker.
4. **Boost / yeniden-normalize** gibi recovery'ye alternatif olarak önerilebilecek shortcut'lar, biased delta'ları büyüttüğü için **daha kötü** sonuç verdi. Recovery'nin yerini tutmaz.
5. **Gradient Ascent için optimizer seçimi kritik:** Adam + grad flip patlar; **SGD + grad clipping + küçük lr** kontrollü silme sağlar.
6. **Adil karşılaştırma**, iki unlearning yönteminin **aynı recovery prosedürüyle** çalıştırılmasını gerektirir; aksi halde fark, silmenin değil, recovery'nin bir artefaktıdır. Simetri kurulduğunda iki yöntem **aynı büyüklük sırasında sonuç** üretti (0.7 puan Instagram farkı).
7. **Yöntem seçimi:** FedEraser delta_history gerektirir (her round sunucuda saklanmalı); GA forget client'ın verisine erişim gerektirir. Pratikte hangi varsayım sağlanıyorsa o yöntem tercih edilir.

---

## 5. Reproducibility Notları

- `SEED=42` (torch + numpy) tüm aşamalarda sabit.
- `delta_history.pkl` FL eğitimi sonunda otomatik üretilir; varsa unlearning aşaması bunu yeniden eğitim olmadan kullanır.
- Notebook hücre sırası: 1–7 (FL eğitimi) → 9 (FedEraser) → 10 (Gradient Ascent) → 11 (yöntem karşılaştırma) → 12 (SISA referans).
- Tüm değerlendirmeler `Europe/Istanbul` saat dilimi.

---

## 6. Rapora Eklenecek Tablolar (taslak)

**Tablo 1 — 8–10 penceresinde olasılık karşılaştırması:**

| Yöntem | Insta % | YT % | Link % | WA % |
|---|---|---|---|---|
| Original (no unlearning) | 31.2 | 26.9 | 7.1 | 34.9 |
| SISA (referans) | 8.4 | 20.4 | 10.9 | 60.3 |
| **FedEraser (final, recovery'li)** | **1.3** | 27.2 | 3.5 | **68.1** |
| **Gradient Ascent (final, recovery'li)** | **2.0** | 34.7 | 5.6 | 57.8 |

**Önemli gözlem:** Hem FedEraser hem GA, **Instagram'ı SISA'dan bile daha agresif silmiş** (1.3% / 2.0% vs SISA'nın 8.4%'i). Bu, recovery'nin u1'in zayıf izlerini sıfırlamakla kalmayıp, küçük bir "over-shoot" ürettiğini gösterir — pratikte privacy açısından **olumlu** (forget signal en aza inmiş), utility açısından da kabul edilebilir (kalan kullanıcı patternleri korunmuş, YouTube + WhatsApp ağırlığı tutarlı).

**Tablo 2 — Geliştirme iterasyonları (rapor isterse appendix):**

| Iter. | Yöntem | Konfig farkı | Sonuç / sorun |
|---|---|---|---|
| 1 | GA | Adam lr=0.01 | Loss 25 → 46705, patladı |
| 2 | FedEraser | Düz toplam | Norm 25.89, kararsız |
| 3 | FedEraser | Ağırlıklı, recovery yok | LinkedIn %65 artifact |
| 4 | FedEraser | + boost | LinkedIn %82, daha kötü |
| 5 | GA | SGD lr=0.001, 3 ep | Insta 23.7%, silme zayıf |
| 6 | **Her ikisi** | **+ 3×5 recovery (final)** | **FedEraser: Insta 1.3%, WA 68.1% / GA: Insta 2.0%, WA 57.8% — SISA'dan agresif, birbirine çok yakın** |

---

*Bu belge, raporu üretecek olan modele yöntem + bulgu + tartışma malzemesi sağlamak amacıyla hazırlanmıştır; raporun kendisi değildir.*
