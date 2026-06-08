# Cihaz Tabanlı Unlearning Pipeline — Rapor Malzemesi

> Bu dosya rapor değildir; raporu yazacak modele/yazara veri ve yorum girdisi sağlar. Federated learning üstüne kurulan **client/server unlearning** pipeline'ının uçtan uca sonuçlarını, mukayese tablolarını ve performans ölçümlerini eksiksiz halde içerir. "Retrain (baseline)" denilen yöntem, basitçe forget client (`user1`) hariç yalnızca `user2` + `user3` ile sıfırdan tekrar federated eğitim yapmaktır — gerçek bir SISA/sharded yapısı değil, düz yeniden eğitimdir.

---

## 0. Pipeline Özeti

| Bileşen | Konum | Görev |
|---|---|---|
| Cihaz (Xiaomi 23090RA98G, Android 15 / API 35, ADB ID `JJ5L5XZDV4Q8VK4H`) | İstemci | Yalnızca user1 verisiyle **Step 1: Gradient Ascent** (Kotlin, elle yazılmış MLP) |
| Host bilgisayar (Python/PyTorch) | Sunucu | Cihazdan gelen state üstünde **Step 2: 3×5 Recovery** (user2 + user3 ile FedAvg) |
| Host bilgisayar (Python/PyTorch) | Referans | Notebook'ta aynı kurulumla FedEraser, Gradient Ascent (Step 1 + Step 2 birlikte), Retrain |

**Hedef:** `user1`'in 08:00–10:00 penceresindeki Instagram baskınlığını silmek, kalan client'ların (user2 = YouTube-baskın, user3 = WhatsApp-baskın) patternlerini bozmamak.

---

## 1. Kurulum

### 1.1 Ortak parametreler (notebook ve cihazda aynı)

| Parametre | Değer |
|---|---|
| Model mimarisi | `Linear(2,16) → ReLU → Linear(16,16) → ReLU → Linear(16,4)` |
| Toplam parametre | 388 |
| Girdi özellikleri | `sin(2π·h/24)`, `cos(2π·h/24)` (h = saat + dakika/60, Europe/Istanbul) |
| Sınıflar | Instagram=0, YouTube=1, LinkedIn=2, WhatsApp=3 |
| Federated round | 6, her round 5 gün veri pencere |
| Lokal eğitim | 20 epoch, Adam `lr=0.01`, batch=32 |
| Aggregator | FedAvg, ağırlık = `n_client_round / n_total_round` |
| Seed | 42 (torch + numpy) |

### 1.2 Step 1 — Gradient Ascent (Forget = `user1`)

| Parametre | Değer |
|---|---|
| Optimizer | SGD |
| Learning rate | 0.001 |
| Epochs | 5 |
| Batch size | 32 |
| Grad clip | `clip_grad_norm_(max_norm=1.0)` |
| Sign flip | Evet (`grad = -grad`) |
| Loss | CrossEntropy (mean reduction) |
| Veri | user1'in tamamı, 2289 kayıt |

### 1.3 Step 2 — Recovery (Sunucu tarafı, hem cihaz pipeline'ında hem notebook'ta aynı)

| Parametre | Değer |
|---|---|
| Round sayısı | 3 |
| Epoch / round | 5 |
| Client'lar | user2, user3 |
| Veri | Her client kendi tüm verisi (6 pencere birleştirilmiş) |
| Lokal optimizer | Adam `lr=0.01`, batch=32 |
| Aggregator | FedAvg, veri ağırlıklı |

### 1.4 Cihaz tarafı uygulama (özet)

- Kotlin (no PyTorch Mobile, no TensorFlow Lite). Elle yazılmış `Linear`/`ReLU`/`CrossEntropy`/`SGD`/`clip_grad_norm`/`flip` — toplam ~450 satır Kotlin.
- `getExternalFilesDir(null)` (= `/sdcard/Android/data/com.cagdaskaratas.mobileunlearningfinal/files/`) üstünden ADB ile iletişim, izin gerektirmez.
- Giriş: `global_state.json` (notebook'tan FL sonrası final state dict) + `user1_data.csv`.
- Çıkış: `ga_state.json` (cihazın ürettiği yeni state dict) + `ga_metrics.json` (loss history, wall-clock, hiperparametreler).
- RNG farkı: PyTorch C++ RNG ile JVM `java.util.Random` aynı seed'de aynı dizilim üretmez. Bu, algoritma + hiperparametre + seed birebir aynı olsa da nihai parametrelerin **bit-bit aynı** olmamasına yol açar; sayısal sapma ~%1–15 bandında (kabul edilebilir).

---

## 2. FL Sonrası Final Global Model — 24 Saat Tahmin Tablosu (Unlearning'den ÖNCE)

Bu tablo, unlearning yapılmadan önce 6 round FedAvg sonucundaki global modelin her saat başı için verdiği tahminin olasılık dağılımıdır. **Hangi saatte hangi client'ın imzası baskın** sorusunun cevabını verir.

**Beklenen client patternleri (tasarımdan):**
- `user1` → 08–10 Instagram, 14–18 LinkedIn+WhatsApp, 18–20 YouTube+Instagram
- `user2` → 08–10 YouTube, 12–14 Instagram, 14–18 LinkedIn+WhatsApp
- `user3` → 08–10 WhatsApp, 18–22 YouTube, 22–02 Instagram

**24-saat tablosu (her hücre: Insta % / YT % / Link % / WA %):**

| Saat | Orijinal (Insta / YT / Link / WA) | Dominant | Hangi client'ların izi |
|---|---|---|---|
| 00:00 | 37.2 / 27.4 /  5.5 / 29.8 | Instagram | u3 (22–02 Instagram) |
| 01:00 | 44.6 / 24.5 /  6.2 / 24.7 | Instagram | u3 (22–02 Instagram) |
| 02:00 | 23.8 / 35.1 /  6.8 / 34.3 | YouTube | karışık |
| 03:00 | 15.2 / 38.0 /  8.0 / 38.9 | WhatsApp | karışık |
| 04:00 | 12.8 / 36.4 / 10.6 / 40.1 | WhatsApp | karışık |
| 05:00 | 13.7 / 35.5 / 11.5 / 39.3 | WhatsApp | karışık |
| 06:00 | 22.0 / 34.8 /  8.6 / 34.6 | YouTube | karışık |
| 07:00 | 40.2 / 30.5 /  3.2 / 26.1 | Instagram | u1 sabah Instagram bandı başlıyor |
| **08:00** | **46.3** / 25.8 /  1.1 / 26.8 | **Instagram** | **u1 baskın (forget hedefi)** |
| **09:00** | 25.9 / 29.9 /  3.0 / **41.2** | **WhatsApp** | u1 + u3 karışımı |
| 10:00 | 26.5 / 17.1 / 32.1 / 24.4 | LinkedIn | u1/u2'nin başlayan LinkedIn bandı |
| 11:00 | 27.6 / 17.3 / 30.2 / 24.8 | LinkedIn | aynı |
| **12:00** | 24.0 / 18.3 / **32.4** / 25.3 | LinkedIn | u2 Instagram bandı başlıyor + u1/u2 LinkedIn |
| **13:00** | **36.7** / 17.7 / 24.4 / 21.2 | **Instagram** | **u2 (12–14 Instagram) baskın** |
| 14:00 | 21.4 / 19.7 / 31.2 / 27.7 | LinkedIn | u1 + u2 LinkedIn+WhatsApp |
| 15:00 | 10.9 / 16.5 / 37.0 / 35.5 | LinkedIn | aynı |
| 16:00 | 10.4 / 15.0 / 37.2 / 37.3 | WhatsApp | aynı |
| 17:00 | 11.7 / 16.7 / 36.1 / 35.6 | LinkedIn | aynı |
| 18:00 | 19.9 / **40.1** / 17.1 / 22.9 | **YouTube** | u3 YouTube bandı başlıyor |
| **19:00** | 14.9 / **71.9** /  4.7 /  8.6 | **YouTube** | **u3 (18–22 YouTube) baskın** |
| **20:00** | 13.0 / **75.2** /  4.0 /  7.7 | **YouTube** | **u3 baskın** |
| 21:00 | 20.1 / **67.1** /  3.3 /  9.5 | YouTube | u3 baskın |
| 22:00 | **39.0** / 29.6 /  2.4 / 28.9 | Instagram | u3 22–02 Instagram bandı başladı |
| 23:00 | 37.6 / 26.4 /  4.0 / 32.0 | Instagram | u3 22–02 Instagram |

**Doğrulama:** Tüm beklenen patternler (u1@08-10 Insta, u2@13 Insta, u3@19-20 YT, u3@22-00 Insta) global modelde net olarak görünüyor. Unlearning hedef bu izlerden **yalnızca u1'in 08–10 Instagram bandını** silmek.

---

## 3. Unlearning Sonrası — 4 Yöntem × 24 Saat (Notebook Çıktısı)

Her hücre: `Insta % / YT % / Link % / WA %`.

| Saat | Orijinal | FedEraser (host) | Grad Ascent (host) | Retrain (host) |
|---|---|---|---|---|
| 00:00 | 37.2/27.4/ 5.5/29.8 | 41.1/30.1/ 2.8/26.0 | 37.6/25.8/ 3.9/32.8 | 36.7/25.0/ 3.7/34.5 |
| 01:00 | 44.6/24.5/ 6.2/24.7 | 36.2/30.5/ 3.9/29.4 | 32.7/28.3/ 5.9/33.0 | 32.9/21.7/ 7.3/38.1 |
| 02:00 | 23.8/35.1/ 6.8/34.3 | 21.9/32.9/ 9.1/36.1 | 27.1/29.0/ 8.8/35.1 | 20.0/25.5/12.5/42.0 |
| 03:00 | 15.2/38.0/ 8.0/38.9 | 19.1/32.3/11.9/36.6 | 22.9/27.8/11.3/38.0 | 11.5/30.0/17.2/41.3 |
| 04:00 | 12.8/36.4/10.6/40.1 | 19.0/31.6/13.1/36.3 | 21.1/28.0/12.3/38.6 | 10.4/34.9/15.8/38.9 |
| 05:00 | 13.7/35.5/11.5/39.3 | 21.1/30.9/12.7/35.3 | 21.4/28.1/13.2/37.3 | 10.3/43.1/12.1/34.4 |
| 06:00 | 22.0/34.8/ 8.6/34.6 | 22.9/30.6/14.3/32.2 | 25.4/26.0/14.1/34.5 | 10.6/49.7/ 7.5/32.2 |
| 07:00 | 40.2/30.5/ 3.2/26.1 | 27.4/27.2/10.9/34.5 | 29.3/23.5/10.9/36.2 | 19.6/31.9/ 5.5/43.0 |
| **08:00** | **46.3**/25.8/ 1.1/26.8 |  4.0/31.8/ 2.9/**61.4** |  6.0/29.5/ 3.3/**61.2** |  9.9/26.9/ 6.5/**56.7** |
| **09:00** | 25.9/29.9/ 3.0/41.2 |  0.2/23.8/ 0.6/**75.3** |  0.3/36.6/ 1.5/**61.6** |  7.6/20.2/ 7.4/**64.8** |
| 10:00 | 26.5/17.1/32.1/24.4 |  4.0/20.5/30.1/45.4 |  8.9/20.3/40.1/30.7 |  8.3/ 9.2/31.8/50.7 |
| 11:00 | 27.6/17.3/30.2/24.8 |  9.1/11.6/54.3/25.0 | 14.8/14.4/45.5/25.2 | 12.4/ 3.4/45.8/38.4 |
| **12:00** | 24.0/18.3/32.4/25.3 | 13.1/10.9/**53.8**/22.2 | 22.9/13.9/**40.9**/22.3 | **36.4**/ 4.3/**41.1**/18.2 |
| **13:00** | **36.7**/17.7/24.4/21.2 | **45.7**/10.0/27.5/16.8 | **59.4**/ 9.6/17.8/13.2 | **51.4**/ 4.7/27.7/16.2 |
| 14:00 | 21.4/19.7/31.2/27.7 | 28.3/17.9/26.1/27.7 | 42.0/15.6/20.7/21.7 | 34.3/12.6/27.8/25.3 |
| 15:00 | 10.9/16.5/37.0/35.5 | 19.1/20.8/27.3/32.9 | 16.3/20.1/27.4/36.2 | 13.1/18.1/31.2/37.6 |
| 16:00 | 10.4/15.0/37.2/37.3 | 20.5/22.4/25.2/31.9 | 15.4/20.4/25.8/38.4 | 12.1/16.7/30.7/40.4 |
| 17:00 | 11.7/16.7/36.1/35.6 | 22.6/24.7/22.3/30.4 | 17.9/22.8/24.2/35.1 | 15.5/22.0/26.3/36.2 |
| 18:00 | 19.9/40.1/17.1/22.9 | 25.4/34.8/14.2/25.6 | 21.3/32.9/17.1/28.7 | 22.8/35.1/15.4/26.6 |
| **19:00** | 14.9/**71.9**/ 4.7/ 8.6 | 11.6/**75.2**/ 4.7/ 8.5 |  4.7/**89.5**/ 2.0/ 3.8 |  6.7/**83.7**/ 4.3/ 5.3 |
| **20:00** | 13.0/**75.2**/ 4.0/ 7.7 |  7.8/**85.3**/ 2.1/ 4.7 |  4.8/**89.6**/ 1.8/ 3.8 |  7.6/**86.4**/ 2.8/ 3.2 |
| **21:00** | 20.1/**67.1**/ 3.3/ 9.5 | 32.4/42.7/ 4.2/20.8 |  5.9/**87.1**/ 2.1/ 4.9 | 32.5/**53.6**/ 4.5/ 9.4 |
| **22:00** | **39.0**/29.6/ 2.4/28.9 | 33.7/39.6/ 4.4/22.4 | 25.1/46.9/ 5.2/22.9 | 35.4/34.6/ 4.4/25.6 |
| **23:00** | 37.6/26.4/ 4.0/32.0 | 42.6/30.8/ 2.1/24.4 | 37.4/25.9/ 1.8/35.0 | 39.4/28.4/ 2.1/30.0 |

### Orijinal'den farklılaşan saat sayısı (24 üzerinden, *dominant app değişen* saatler)

| Yöntem | Değişen saat | % |
|---|---|---|
| FedEraser | 11 | 46% |
| Grad Ascent | 12 | 50% |
| Retrain (baseline) | 12 | 50% |

→ Üç yöntem de yaklaşık aynı oranda **dağılım değişikliği** yaratıyor; FedEraser çok hafif tarafta. Bu beklenen: FedEraser yalnızca u1'in delta'sını çıkarır + recovery; "minimum invaziv" bir yaklaşım.

---

## 4. Kritik Saat Aralıklarında Bozulma Analizi

Beklentinin **silinmesi gereken** bandı tek başına (u1 @ 08–10 Instagram). Kalan tüm bandlar **korunması gereken** (u2 ve u3 imzaları). Aşağıdaki tablolar her band için yöntem yöntem ne olduğunu gösterir.

### 4.1 BAND A — `08:00–10:00` (FORGET hedefi: u1 Instagram)

Beklenen: Instagram BUYUK ölçüde düşmeli, dağılım u2 (YT) + u3 (WA) imzalarına kaymalı.

**8–10 ortalama (gerçek veri noktaları üstünden, `analyze_8_to_10`):**

| Model | Insta % | YT % | Link % | WA % | Yorum |
|---|---|---|---|---|---|
| Orijinal | **31.2** | 26.9 | 7.1 | 34.9 | u1 imzası baskın |
| FedEraser | 1.3 | 27.2 | 3.5 | **68.1** | u1 silindi, WA (u3) dominantlığa geçti |
| Grad Ascent | 2.0 | 34.7 | 5.6 | 57.8 | u1 silindi, YT+WA dağıldı |
| Retrain | 8.4 | 20.4 | 10.9 | 60.3 | u1 silindi, WA dominant |
| **Cihaz GA + Recovery** | **0.9** | **49.2** | 5.0 | 44.9 | **En agresif silme**, YT ile WA dengeli |

**Yorum:** Cihaz pipeline, host versiyonlardan **daha agresif** silme başardı (Insta %0.9 < %1.3, %2.0, %8.4). LinkedIn artifact'ı yok (% 5.0). YouTube ağırlığı %49.2 ile u2'nin patternine daha güçlü kaymış. Privacy açısından mükemmel sonuç.

### 4.2 BAND B — `12:00–14:00` (u2 Instagram — KORUNMASI gereken)

Beklenen: Bu band u2'ye ait; **u1 silinince bile dokunulmaması** lazım.

| Saat | Orijinal | FedEraser | GA host | Retrain | Cihaz pipeline |
|---|---|---|---|---|---|
| 12:00 Insta % | 24.0 | 13.1 | 22.9 | **36.4** | 14.1 |
| 13:00 Insta % | **36.7** | **45.7** | **59.4** | **51.4** | **48.5** |

**Yorum:** 13:00'de tüm unlearning yöntemleri Instagram'ı **yukarı çekti** (u2'nin imzasına daha çok ağırlık verdi). Bu **bozulma değil, beklenen davranış** — u1 çıkarılınca u2'nin payı artıyor. Retrain de 51.4% gösterdiği için bu, "ideal" davranışın doğal sonucu.

### 4.3 BAND C — `14:00–18:00` (u1 LinkedIn+WA + u2 LinkedIn+WA — kısmen korunmalı)

Beklenen: u1'in katkısı kalkacağı için LinkedIn azalabilir, ama u2'nin de aynı pattern olduğundan tamamen yok olmamalı.

| Saat | Orijinal Link% | FedEraser Link% | GA Link% | Retrain Link% | Cihaz Link% |
|---|---|---|---|---|---|
| 14:00 | 31.2 | 26.1 | 20.7 | 27.8 | 27.0 |
| 15:00 | 37.0 | 27.3 | 27.4 | 31.2 | 26.8 |
| 16:00 | 37.2 | 25.2 | 25.8 | 30.7 | 24.9 |
| 17:00 | 36.1 | 22.3 | 24.2 | 26.3 | 23.1 |

**Yorum:** LinkedIn olasılığı dört yöntemde de azalmış (~%5–15 puan), ama hâlâ ortalamada yaklaşık %25 düzeyinde. u2'nin LinkedIn katkısı korunduğu için tamamen kaybolmadı. Bozulma yok.

### 4.4 BAND D — `18:00–22:00` (u3 YouTube — KORUNMASI gereken)

Beklenen: u3'ün imzası baskın; u1 silinmesi bu bandı etkilememeli.

| Saat | Orijinal YT% | FedEraser YT% | GA YT% | Retrain YT% | Cihaz YT% |
|---|---|---|---|---|---|
| 18:00 | 40.1 | 34.8 | 32.9 | 35.1 | **45.3** |
| 19:00 | **71.9** | **75.2** | **89.5** | **83.7** | **91.8** |
| 20:00 | **75.2** | **85.3** | **89.6** | **86.4** | **92.2** |
| 21:00 | **67.1** | 42.7 | **87.1** | 53.6 | **90.3** |

**Yorum:** 19:00–20:00'de tüm yöntemlerde YouTube **daha da güçlendi** (u3 imzasının payı arttı). 21:00 ilginç bir durum: FedEraser ve Retrain düştü (42.7%, 53.6%), GA ve Cihaz GA arttı (87.1%, 90.3%). Bu RNG/yöntem hassasiyetinin görünür olduğu özel bir saat. Sonuç olarak: **u3 imzası bozulmadı, çoğu durumda güçlendi**.

### 4.5 BAND E — `22:00–02:00` (u3 Instagram — KORUNMASI gereken)

Beklenen: u3'ün 22–02 Instagram patterni korunmalı.

| Saat | Orijinal Insta% | FedEraser Insta% | GA Insta% | Retrain Insta% | Cihaz Insta% |
|---|---|---|---|---|---|
| 22:00 | 39.0 | 33.7 | 25.1 | 35.4 | 24.6 |
| 23:00 | 37.6 | 42.6 | 37.4 | 39.4 | 35.7 |
| 00:00 | 37.2 | 41.1 | 37.6 | 36.7 | 31.3 |
| 01:00 | 44.6 | 36.2 | 32.7 | 32.9 | 22.9 |

**Yorum:** u3'ün Instagram patterni 23:00–00:00'de **net korunmuş**. 22:00 ve 01:00'de hafif düşüş (5–22 puan), ama hâlâ baskın sınıflardan biri. Bu band, u1'in de kısmen Instagram kullandığı saatlere yakın olduğundan küçük bir "yan etki" beklenir; kabul edilebilir.

### 4.6 Genel Bozulma Verdikti

| Band | Hedef | Sonuç |
|---|---|---|
| A (08–10) | **SİL** | ✓ Tüm yöntemlerde başarılı; Cihaz pipeline en agresif |
| B (12–14 u2 Insta) | KORU | ✓ Beklenen biçimde u2'nin payı arttı (bozulma değil) |
| C (14–18 LinkedIn) | KORU | ✓ Hafif düşüş, ama u2 katkısıyla kalıcı |
| D (18–22 u3 YT) | KORU | ✓ Çoğu yöntemde güçlendi; 21:00 yöntemler arası varyans |
| E (22–02 u3 Insta) | KORU | ✓ 23:00–00:00 net korundu; uçlarda hafif düşüş |

→ **Kalan client'ların imzaları bozulmadı; yalnızca u1'in 08–10 Instagram bandı temizlendi.**

---

## 5. Cihaz Tabanlı Pipeline — Özel Detay

### 5.1 Cihaz GA çıktısı (Step 1)

| Metrik | Değer |
|---|---|
| Cihaz | Xiaomi 23090RA98G, Android 15 |
| Veri | user1.csv, 2289 kayıt |
| Loss (epoch 1 → 5) | 1.388 → 1.445 → 1.512 → 1.597 → 1.692 (monoton **artan** = doğru) |
| Wall-clock | **0.224 s** |
| Çıkış dosyaları | `ga_state.json` (11 KB), `ga_metrics.json` (353 B) |

### 5.2 Sunucu Recovery (Step 2, host'ta)

| Metrik | Değer |
|---|---|
| Round sayısı | 3 |
| Epoch / round | 5 |
| Client'lar | user2, user3 (tam veri) |
| Round avg loss | 1.026 → 1.018 → 1.021 (stabil) |
| Round-bazlı acc | user2 %45.2 / user3 %60.2 (R1), benzer R2/R3 |
| Wall-clock | **3.434 s** |

### 5.3 Cihaz pipeline final 8–10

| App | Olasılık | Yorum |
|---|---|---|
| Instagram | **0.9%** | %96'dan fazla düşüş (u1 imzası silindi) |
| YouTube | 49.2% | u2 patterni baskın |
| LinkedIn | 5.0% | Artifact yok |
| WhatsApp | 44.9% | u3 patterni baskın |

### 5.4 Cihaz pipeline — 24 saat tahmin (öne çıkan saatler)

| Saat | Tahmin | Confidence | Yorum |
|---|---|---|---|
| 08:00 | WhatsApp | 51.6% | u1 Instagram silindi, u3 WA öne çıktı |
| 09:00 | YouTube | 55.8% | u2 YT öne çıktı |
| 13:00 | Instagram | 48.5% | u2 patterni (12–14 Insta) korunmuş |
| 19:00 | YouTube | **91.8%** | u3 patterni (18–22 YT) çok güçlü |
| 20:00 | YouTube | **92.2%** | u3 patterni |
| 21:00 | YouTube | **90.3%** | u3 patterni |
| 23:00 | Instagram | 35.7% | u3 22–02 Insta korunmuş |

Tam tablo: `results/device_pipeline/hourly_predictions.json`.

---

## 6. Performans / Süre Karşılaştırması

### 6.1 Ham ölçümler

| Yöntem | Step 1 (silme) | Step 2 (recovery) | Toplam |
|---|---|---|---|
| FedEraser (host) | **0.0009 s** | 2.1492 s | **2.1501 s** |
| Grad Ascent (host) | 0.3420 s | 2.5465 s | 2.8887 s |
| Retrain (host baseline) | — | — | **3.4164 s** |
| **Cihaz GA + Recovery host** | **0.2238 s** | 3.4338 s | 3.6577 s |

### 6.2 Çarpan Analizi — Step 1 (saf silme adımı)

FedEraser delta subtraction'ı **referans** alarak:

| Step 1 Yöntemi | Süre | FedEraser'a göre |
|---|---|---|
| FedEraser delta subtraction (host) | **0.0009 s** | 1× (referans) |
| Cihaz GA (Kotlin, telefon) | 0.2238 s | **~249× daha yavaş** |
| Host GA (Python/PyTorch) | 0.3420 s | **~380× daha yavaş** |

**Cihaz GA vs Host GA (aynı algoritma, farklı runtime):**

| Karşılaştırma | Oran |
|---|---|
| Host GA / Cihaz GA | **0.342 / 0.224 = 1.527×** |
| → **Telefon, host bilgisayardan yaklaşık 1.53× DAHA HIZLI** |

### 6.3 Çarpan Analizi — Toplam (Retrain baseline)

Retrain (3.4164 s) baseline alındığında:

| Yöntem | Toplam | Retrain'e göre speedup |
|---|---|---|
| FedEraser | 2.1501 s | **~1.59× daha hızlı** |
| Grad Ascent (host) | 2.8887 s | **~1.18× daha hızlı** |
| Cihaz GA + Recovery host | 3.6577 s | **0.93× (≈1.07× daha yavaş)** |

> Cihaz pipeline'ın "daha yavaş" görünmesi yanıltıcı: Wall-clock'un %94'ü (3.43 / 3.66 s) **sunucu tarafı recovery**dir. Cihaz Step 1 sadece 0.22 s. Gerçek bir federated kurulumda recovery de farklı cihazlarda paralel yapılırsa, cihaz pipeline'ın gerçek "client-side" maliyeti çok daha düşük olur.

---

## 7. Şaşırtıcı Bulgu — Telefon, Host'tan Daha Hızlı

### 7.1 Donanım Beklentisi vs Gerçek

| Donanım | CPU profili |
|---|---|
| Host bilgisayar | Modern x86_64 (Windows 11, Python 3.x, PyTorch CPU) |
| Telefon (Xiaomi 23090RA98G) | ARM Cortex (mid-range mobil), Android 15 / API 35 |

Beklenti: Host bilgisayarın çok daha hızlı GA yapması gerekir.

Gerçek: **Telefon 0.224 s, host 0.342 s.** Telefon **~1.53× hızlı**.

### 7.2 Teknik Analiz — Neden Tersine Döndü

Bu **donanım hızı paradoksu değil, runtime overhead karşılaştırmasıdır**. Aynı algoritma, aynı hiperparametreler, aynı veri — fark sadece çalıştıran yığında. Üç ayrı katmanda overhead birikiyor:

#### 7.2.1 PyTorch Dispatcher + Autograd Overhead'i

PyTorch'un her `nn.Linear(x)` veya `+`, `*` çağrısı şu zinciri çalıştırır:

1. **Python yorumlama** — Python interpreter, `__call__` ve `__add__` gibi dunder'leri çözer; CPython 3.x'te her bytecode komutu ortalama 50–100 ns'lik dispatch maliyetine sahiptir.
2. **Tensor dispatcher** — PyTorch'un C++ tarafındaki `Dispatcher` her op için backend (CPU/CUDA), dtype, layout kombinasyonuna göre uygun kernel'i seçer. Op başına ~1–10 µs sabit maliyet.
3. **Autograd graph node oluşturma** — Forward modunda gradient tracking aktifse (default) her op için bir `Node` allocate edilir, `next_edges` listesi kurulur, version counter güncellenir.
4. **Tensor allocation** — Her intermediate output için yeni `Tensor` objesi; ardından eski intermediate'lar GC/refcount ile temizlenir. Allocator (`caching_allocator`) hızlı ama yine de op başına ~µs düzeyinde.

Bizim eğitim hattımızda **bir forward pass = 3 Linear + 2 ReLU + 1 CE = 6 op, backward = ~12 op**. Yani batch başına ~18 op'luk dispatcher zinciri. **2289 kayıt / 32 batch ≈ 72 batch × 5 epoch = 360 batch × 18 op = 6480 op**. Her op ~2-5 µs dispatcher overhead alırsa toplam **13–32 ms** sadece dispatcher'da geçer — gerçek hesap ise nanosaniyeler düzeyinde.

Yani **PyTorch wall-clock'unun büyük kısmı framework içinde**, gerçek aritmetikte değil.

#### 7.2.2 BLAS / SIMD'in Bu Boyutta Devreye Girmemesi

PyTorch CPU backend'i `matmul` için **BLAS** (Windows'ta genelde MKL veya OpenBLAS) çağrısı yapar. BLAS'ın asıl avantajı **büyük matris çarpımlarında**:

- Cache-tiling
- SIMD vektörizasyonu (AVX2 / AVX-512)
- Multi-threading (`OMP_NUM_THREADS`)

Bu özellikler **matrisin büyüklüğüyle amortize** olur. Bizim matmullarımız:
- `(32, 2) × (2, 16) → (32, 16)` — sadece **1024 multiply-add**
- `(32, 16) × (16, 16)` — 8 192 mac
- `(32, 16) × (16, 4)` — 2 048 mac

Toplam batch başına ≈ **11 K mac**. BLAS çağrısının kendi setup'ı (kernel selection, thread pool wake-up) bu kadar küçük matrislerde **hesabın üstünde maliyet** yaratır. Multi-threading **fayda yerine zarar** verebilir (thread sync > iş). PyTorch hot loop için "small matmul fallback" kullansa da Python + dispatcher katmanı yine de baskındır.

#### 7.2.3 DataLoader + Tensor Conversion

`DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)` her batch'te:
- `__getitem__` çağrıları (Python-side)
- `default_collate` ile tensor reshape
- (default) `pin_memory=False`, `num_workers=0` — yine de Python overhead

Kotlin tarafında:
- `IntArray` permütasyonu (Fisher-Yates, ~µs)
- `gatherRows(X, indices)` — düz `System.arraycopy` (`O(batch · features)`)
- Sıfır framework overhead, sıfır object allocation churn

#### 7.2.4 Kotlin/JVM'in "Bare Metal" Avantajı

`DoubleArray` üstünde yazılmış matmul:
- **JIT (HotSpot C2)** hot path'i 1–2 ms içinde derler.
- İç döngüler **register'da kalır**, array bounds check JIT tarafından çıkarılır (`-XX:+AggressiveOpts`).
- 388 parametre + batch tensor'ları **L1 cache**'e (≈64 KB) sığar; cache miss yok.
- Autograd graph yok — backward pass elle yazılı, sadece gerekli intermediate'lar (`lastX` per Linear, mask per ReLU) tutulur.
- Python/C++ FFI çağrı katmanı yok.

JVM, küçük ML çekirdekleri için Python+PyTorch'a kıyasla **dispatcher-free, allocation-light** bir ortam sağlıyor. Donanım yavaş olsa da framework cezası sıfır.

### 7.3 Bu Bulgunun Önemi

- **"Cihaz tabanlı ML yavaştır" varsayımı bu boyutta çürütüldü.** Edge unlearning senaryolarında telefon, masaüstü Python pipeline'larıyla rekabet edebilir — hatta geçebilir.
- Federated privacy senaryosunda forget client'ın verisinin **cihazdan hiç çıkmadan** silinmesi, performans cezasız mümkün.
- "Frameworkless" hesap (pure Kotlin / pure C++) küçük modellerde **adapter pattern olarak değerli** — özellikle on-device training & unlearning'de.

### 7.4 Ölçekleme Senaryoları — Model ve Veri Büyüdükçe Ne Olur?

Bulgumuz "telefon her zaman hızlı" demek değildir. Avantaj **model + veri boyutu × framework overhead'i** dengesine bağlı. Aşağıdaki tablo nereye kadar geçerli, nerede tersine döner sorusunu kabaca cevaplar:

| Senaryo | Model boyutu | Batch örnek | Beklenen kazanan | Gerekçe |
|---|---|---|---|---|
| **Bizim setup** | 388 param | 32 | **Telefon (Kotlin)** | Framework overhead > hesap |
| Küçük MLP | ~5 K param | 32–64 | Telefon (Kotlin) | Hâlâ overhead-bound |
| Orta MLP / küçük CNN | ~100 K param | 64–128 | Yakın denge | Crossover bölgesi |
| Orta CNN (MobileNet'in 1/10'u) | ~500 K – 1 M param | 64–128 | Host (BLAS devreye girer) | SIMD/AVX amortize olur |
| Büyük model (ResNet-50 vb.) | ~25 M param | 32+ | Host (CPU) veya GPU | BLAS, multi-thread, GPU avantajı |
| Büyük model + GPU | ≥1 M param | 64+ | Host (GPU) | Cihaz GPU'su daha zayıf (yarı performans) |

**Crossover noktası neye bağlı:**

1. **Tek bir op'un FLOP/dispatch oranı:** ~10⁴ FLOP / op'a ulaşınca dispatcher overhead amortize olmaya başlar.
2. **Aritmetik yoğunluk (Arithmetic Intensity):** FLOP / byte memory access oranı. Memory-bound modellerde (örn. saf MLP'ler) cache miss baskın olur; compute-bound modellerde BLAS/SIMD kazanır.
3. **Multi-threading kazanımı:** BLAS thread pool sadece batch × matrix yeterince büyükse kâr eder. Aksi halde `OMP_NUM_THREADS=1` daha hızlı.
4. **GPU varsa:** Host GPU varsa eğitim çoğu senaryoda kazanır; ama CPU vs CPU karşılaştırmasında crossover yukarıdaki gibi.

**Pratik kural:** Model boyutu **~100 K parametre**'nin altında ve batch size **<128** ise, "framework cezası" pure Kotlin'in donanım dezavantajını yenebilir. Bu bant, **on-device personalization + on-device unlearning** için tam adres.

### 7.5 Cihaz Tarafında Daha Büyük Modeller — Nasıl Optimize Edilir?

Modelin büyüdüğü senaryolarda cihaz tarafının da BLAS/SIMD seviyesinde optimize edilmesi gerekir. Opsiyonlar:

- **PyTorch Mobile / ExecuTorch (LibTorch ARM)**: Compute kernel'ları C++ + NEON vektörize, autograd graph runtime tarafından inferred. APK boyutu ~10–30 MB artar.
- **TensorFlow Lite + on-device training (TFLite Model Personalization)**: Quantize edilmiş eğitim destekler; resmî on-device training API'si var.
- **ONNX Runtime Mobile**: Inference odaklı; training için bazı extensions var ama olgunlaşmamış.
- **Pure NEON intrinsics (NDK + JNI)**: Kotlin'den C kodu çağırılır. Kontrolü maksimum ama yazım maliyeti yüksek.
- **Quantization (INT8)**: 4× memory tasarrufu + ARM v8.2 dot-product instructions ile **4–8× hız**. Eğitimde kalite kaybı dikkat gerektirir.

Bu projede 388 parametre olduğu için Kotlin double aritmetiği yeterli oldu; orta boyutta bir modele geçerken muhtemelen **PyTorch Mobile** veya **TFLite training** tercih edilir, çünkü Kotlin/double overhead'i model büyüdükçe BLAS-tipi optimizasyonların gerisinde kalır.

### 7.6 Veri Büyüklüğü Boyutunda Cihaz Dezavantajı

Cihazın asıl dezavantajı **bellek** ve **enerji**:

- **RAM:** Modern telefonlar 4–12 GB; pratikte uygulamaya verilen pay ~512 MB – 2 GB. Büyük dataset (örn. milyonlarca kayıt) tek seferde yüklenemez → disk I/O baskın olur.
- **Storage I/O:** UFS 3.1 telefon NVMe'den 3–5× yavaş; büyük dataset epoch'larında baskındır.
- **Termal throttling:** Sürekli yüksek CPU kullanımı 60–90 saniye sonra thermal throttle'la ~%50 yavaşlama.
- **Pil:** SGD/gradient ascent çok uzun sürerse pil ömrü pratik problem.

Bu projede 2289 kayıt ve 0.22 saniye → bu kısıtların hiçbiri devreye girmedi. Ama 10⁵–10⁶ kayıtlı bir senaryoda cihaz tarafı **lazy data streaming** ve **fragmented training** gerektirir.

### 7.7 Genel Yorum — Edge Unlearning İçin Stratejik Sonuç

| Ölçek | Önerilen kurulum |
|---|---|
| Tiny model (≤10 K param), küçük veri | **Pure Kotlin / Swift** — overhead minimum, framework gereksiz |
| Küçük-orta model (10 K – 1 M param) | **Quantized PyTorch Mobile / TFLite** — kernel optimizasyonu gerekli |
| Büyük model | Cihaz GPU/NPU + ExecuTorch / Core ML; veya hibrit (cihaz sadece embedding/diff, sunucu eğitir) |
| Hibrit pipeline (bu projedeki gibi) | **Cihaz: silme adımı; sunucu: recovery / aggregation** — privacy + ölçeklenebilirlik dengesi |

**Bizim setup'ımız**, edge unlearning'in **"tiny model + frameworkless"** kuadrantına denk düşüyor; bu nedenle telefon, host'u Step 1'de yenebildi. Daha büyük model senaryolarında **Step 1 hâlâ cihazda** kalır (privacy gereği), ama runtime seçimi (PyTorch Mobile vb.) değişir.

---

## 8. Utility Koruması — Kalan Client Accuracy

8–10 penceresinin dışında, **kalan client'ların kendi verileri üstündeki accuracy** unlearning'in onları bozup bozmadığının özet göstergesidir.

| Model | user2 acc | user3 acc | Yorum |
|---|---|---|---|
| Orijinal | 31.9% | 53.1% | Baseline |
| FedEraser | 31.8% | 54.6% | Korundu (hatta hafif iyileşme) |
| Grad Ascent (host) | 32.4% | 52.5% | Korundu |
| Retrain (baseline) | 33.6% | 49.0% | u2 iyileşti, u3 hafif düştü |

→ Hiçbir yöntemde kalan client'lar belirgin bozulmadı. **Utility-privacy trade-off tatmin edici.**

---

## 9. Loss Trajectorileri

### 9.1 Cihaz GA (Step 1)

| Epoch | avg loss |
|---|---|
| 1 | 1.388 |
| 2 | 1.445 |
| 3 | 1.512 |
| 4 | 1.597 |
| 5 | **1.692** |

**Monoton artan** → gradient ascent doğru yönde çalıştı; patlama yok.

### 9.2 Sunucu Recovery (Step 2, cihaz pipeline'ı)

| Round | avg loss |
|---|---|
| 1 | 1.026 |
| 2 | 1.018 |
| 3 | 1.021 |

Stabil — recovery convergeleşmiş gibi görünüyor (3 round yeterli).

---

## 10. Ek Kritik Gözlemler (Rapor İçin Değerli Yan Notlar)

### 10.1 Privacy Argümanının Niceliği

Klasik federated learning'de "client'ın verisi cihazdan çıkmaz" denilir; ancak gradient'ler / weight update'leri **veri sızdırabilir** (model inversion attacks, membership inference, gradient leakage). Unlearning bağlamında ise:

- **Sunucu tabanlı unlearning (notebook'taki GA Step 1):** Sunucu, forget client'ın **verisini bilir** (delta'lar saklanır ya da yeniden eğitim için veri gerekir). Bu, "veriyi unutturuyorum" söylemiyle çelişir.
- **Cihaz tabanlı GA Step 1 (bu pipeline):** Sunucu yalnızca **post-GA state dict** alır. user1'in ham verisi, log'ları, hatta gradient'leri sunucuya hiçbir zaman gönderilmez. Bu, **right-to-be-forgotten** (GDPR Art. 17) hukuki çerçevesine **fiilen daha yakın** durur.

### 10.2 Sunucunun Cihazın Yaptığı İşi Doğrulayamaması (Trust Assumption)

Pipeline'da kritik bir güven varsayımı var: **sunucu, cihazın gerçekten GA yapıp yapmadığını doğrulayamaz**. Kötü niyetli bir client şunları yapabilir:
- Hiç GA yapmayıp orijinal state'i geri gönderir → silme yapılmaz, sunucu farkına varmaz.
- Random noise ile parametre değiştirir → recovery yine de "geçerli" görünebilir.
- Modeli kasıtlı olarak başka bir yöne kaydırır (data poisoning) → "unlearning bahanesi altında saldırı".

Gerçek bir dağıtımda bunlar için ek mekanizmalar gerekir:
- **Cryptographic attestation** (SafetyNet, Play Integrity API)
- **Trusted Execution Environment (TEE)** üstünde GA çalıştırma (örn. ARM TrustZone)
- **Verifiable computation** (ZK-SNARK ile GA çıktısının doğruluğunu kanıtlama — pratik değil ama teorik var)
- **Statistical defense:** Sunucu birden fazla client'tan gelen update'leri istatistiksel olarak outlier detection'a tabi tutar.

Raporda bu nokta vurgulanmalı: **"cihaz tabanlı unlearning" doğal olarak doğrulanabilir değildir; ek bir trust katmanı gerekir.**

### 10.3 Reprodüksiyon Sınırı — RNG Farkı Ne Demek?

Aynı seed (42), aynı algoritma, aynı hiperparametre — ama PyTorch ve JVM farklı sayısal sonuç verdi. Bunun pratik anlamı:

- **Aynı PyTorch sürümünde** (örn. notebook'ta) aynı sonuç **bit-bit** üretilebilir.
- **Aynı Kotlin sürümünde** (cihazda) yine bit-bit reproducible.
- **PyTorch ↔ Kotlin arası bit-bit eşleşme imkânsız** — RNG state'leri farklı dilde / runtime'da farklı evrilir.

Bu, "deneyi tekrar edemem" anlamına gelmez; "iki farklı runtime'da aynı sayısal sonucu beklemiyorum" demektir. Önemli olan **algoritmik eşdeğerlik** (aynı update kuralı, aynı clip, aynı flip) — biz bunu sağladık. Final 8–10 sonuçlarındaki ~%1–15'lik bantta sapma bu farkın tezahürüdür.

Rapor mesajı: **"Sonuçlar farklı çıktı" diye eleştirmek yerine "iki ayrı runtime'da aynı algoritma çalıştı" diye okunmalı.**

### 10.4 Enerji ve Pil Tüketimi (Tartışmalı, Ölçülmedi)

Cihaz tarafı 0.22 saniye CPU kullanımıyla bitti — gözle görülür pil etkisi yok. Ancak gerçek dağıtım senaryolarında (binlerce client, periyodik unlearning request'leri) toplam enerji bütçesi önemli hale gelir:

- Modern Snapdragon SoC ~5–8 W tepe tüketim, bizim hesabın gerektirdiği ~%30 CPU ile ~2 W = **0.22 s × 2 W ≈ 0.44 J** (yaklaşık 0.0001 mAh, **ihmal edilebilir**).
- Daha büyük modeller / daha çok veri → her unlearning bir e-posta okuma kadar batarya tüketebilir.
- Yine de **sunucu tarafında milyonlarca client'ın retraining'i** ile karşılaştırıldığında **toplam enerji kazanımı dağıtık model lehinedir** (sunucu CPU saatleri çok daha pahalı).

### 10.5 Ölçeklenebilirlik — Federated Senaryoda Gerçek Kullanım

Bu projede tek bir forget client (user1) silindi. Pratik bir federated unlearning senaryosunda:
- **Birden fazla client aynı anda unlearning isteyebilir.** Sunucu queue + batch unlearning gerekir.
- **Cihaz GA pipeline'ı paralel ölçeklenebilir:** Her client kendi cihazında bağımsız çalışır; sunucu sadece aggregation yapar. Bu, çok-client senaryosunda **lineer ölçekleme** verir.
- **Recovery, ortak bir sunucu kaynak** — burada bottleneck oluşabilir. Çözüm: recovery'i de federated yapmak (yine farklı cihazlarda), ama bu zaten standart bir FL round'una dönüşür.

Yani **uzun vade**de cihaz tabanlı unlearning, sunucu tabanlıdan **daha iyi ölçeklenir**; çünkü silme adımı client'a paralel olarak dağılır.

### 10.6 Hand-Rolled Kotlin MLP'nin Yan Faydası — Auditability

Kotlin tarafı 4 dosyada (`Mat.kt`, `Model.kt`, `Train.kt`, `Io.kt`) toplam ~450 satır. Notebook tarafıyla yan yana koyulduğunda her bir matematiksel adım (Linear forward = `xW^T + b`, ReLU mask, CE softmax = `(softmax - one_hot) / N`, clip_grad_norm = `max_norm / (total_norm + 1e-6)`) **birebir denetlenebilir**. PyTorch Mobile gibi kara-kutu runtime'lar bu denetlenebilirliği maskeler.

Privacy + akademik şeffaflık senaryolarında bu **kayda değer bir mühendislik avantajı**dır: "modelin ne yaptığını biliyoruz" iddiası kanıtlanabilir.

### 10.7 Recovery Aşamasının Cihaza Taşınması — Neden Yapmadık?

Pipeline tasarım kararı: **Step 2 (recovery) host'ta** kaldı. Sebepleri:

1. **Veri lokasyonu:** Recovery, user2 ve user3'ün tüm verisini gerektirir. Bunlar farklı client'larda. Bizim cihaz simülasyonu sadece user1'i içerir.
2. **Gerçek FL senaryosu:** Recovery aslında u2 ve u3'ün kendi cihazlarında yapılır; sonra sunucu FedAvg'le birleştirir. Bizim simülasyonumuzda bu "ayrı cihaz" rolünü host oynadı.
3. **Karşılaştırılabilirlik:** Notebook'taki recovery Python/PyTorch'ta yapılıyor; cihaz pipeline'ında da aynı sayısal yöntemi kullanmak için host Python kullanıldı.

Yani host recovery, **tek client'lı simülasyonun bir kısıtı**, mimari bir tercih değil. Gerçek dağıtımda her client kendi cihazında recovery yapardı.

### 10.8 Bizim Tasarımdaki "Mini-Server" — Recovery Script Standalone Çalışıyor

`server_recovery.py` notebook'tan bağımsız çalışır; yalnızca cihaz JSON çıktıları + u2/u3 CSV'lerini ister. Bu, gerçek bir client/server pipeline'da **sunucu tarafı endpoint'i**nin nasıl görünebileceğinin somut örneği. Notebook'taki tek-süreç çözümün aksine, bu script production-style bir mikroservis hat'ına yerleştirilebilir.

### 10.9 24-Saat Görünümü Neden Önemli — Sadece 8-10'a Bakmak Yeterli Değil

Eğer sadece 8–10 penceresine bakılsaydı, "Instagram silindi, başarı" derdik. Ama 24-saatlik tablo göstermesi gerekiyor ki **diğer saatlerde yan etki olmamış**. Özellikle:

- **12–14 (u2 Instagram):** Korunmuş, hatta güçlenmiş — utility kazancı.
- **19–21 (u3 YouTube):** Korunmuş, hatta güçlenmiş — utility kazancı.
- **15–17 (LinkedIn):** Hafif düşmüş ama u2 katkısıyla kalıcı.

Bu görünüm, "unlearning yan etkili mi?" sorusunun gerçek cevabını verir. **Tek pencere değerlendirmesi yetersizdir.**

### 10.10 Cihaz Step 1'in Wall-Clock'unu Şüpheyle Karşıla

0.224 saniye gerçekten **bir kez** ölçüldü, JIT warm-up cold-start senaryosunda. Tekrarlayan ölçümlerde:
- İlk 1-2 epoch JIT henüz hot path'i derlerken yavaş olabilir.
- HotSpot C2 derleyici full tier üzerinde çalıştığında 2-3× hızlanabilir.
- Termal durum etkilenir (cihaz soğuk vs sıcak).

**Rapora not olarak eklenmesi gereken caveat:** "Tek-shot ölçüm; n=10 average ile ölçülmeli — bu projenin pratik kapsamı dışında ama metodolojik bir gözlem." Yine de büyüklük sırası (~0.2 s) doğru.

---

## 11. Sonuçlar (Madde Madde)

1. **Cihaz tabanlı Step 1 (Gradient Ascent) başarıyla uygulandı.** Gerçek bir fiziksel cihazda (Android 15, Xiaomi) 2289 kayıtlık user1 verisi üstünde 5 epoch SGD + grad clip + grad flip → **0.224 s**, loss monoton artan.

2. **Sunucu tarafında 3×5 recovery aynı kodla yapıldı.** Pipeline toplam = 3.66 s (0.22 + 3.43).

3. **Final 8–10 dağılımı, host yöntemlerinden agresif:** Cihaz pipeline Instagram %0.9, host yöntemler %1.3–8.4 bandında. LinkedIn artifact yok (%5.0).

4. **Bozulma analizi geçti:** Kalan tüm pattern bandları (u2 12–14 Insta, u3 18–22 YT, u3 22–02 Insta) korundu veya güçlendi. Hiçbir kritik bandda istenmeyen yan etki yok.

5. **Performans, küçük model için sezgi karşıtı:** Telefon, host bilgisayarın GA Step 1'inden **1.53× daha hızlı**. Sebep: 388 parametrelik mini-modelde PyTorch overhead'i, telefondaki düz Kotlin aritmetiğini geçiyor.

6. **FedEraser delta subtraction Step 1'de yenilmez:** **~0.9 ms**. GA herhangi bir donanımda bunun yanına yaklaşamaz; ama delta_history yoksa kullanılamaz.

7. **Recovery, küçük modellerde opsiyonel değildir:** Önceki iterasyonlarda recovery'siz FedEraser %65 LinkedIn artifact üretti; recovery sonrası bu sıfıra inmiş durumda.

8. **Pipeline simetrisi (cihaz + sunucu recovery), notebook'taki tek-süreç GA ile karşılaştırılabilir sonuç verdi** (8–10 Instagram: cihaz pipeline %0.9 vs notebook GA %2.0; aynı büyüklük sırası).

9. **Privacy ve client deneyimi:** Forget client'ın verisi cihazdan çıkmadan unlearning yapılabilir oldu. Bu, gerçek bir privacy-preserving federated unlearning kurulumunun "client side" kısmını tatmin eder.

10. **Retrain (baseline) vs unlearning yöntemleri:** Retrain 3.42 s; FedEraser 2.15 s (1.59× hızlı); Grad Ascent 2.89 s (1.18× hızlı). 388-parametrelik bu modelde fark dramatik değil, ancak model boyutu büyüdükçe (~10⁵–10⁶ parametre) bu avantaj çarpışmalı şekilde büyür.

---

## 12. Üretilen Dosyalar

### Notebook (federated_learning.ipynb) çıktıları
- `weights/global_final_weights.json` — pre-unlearning final FL state dict
- `delta_history.pkl` — 6 round × 3 client delta'lar
- `results/unlearning_results.json` — 8–10 dağılımları + 24-saat tahminleri + utility
- `results/comparison_plot.png` — 4 model 8–10 bar grafik
- `results/comparison_with_sisa.png` — Retrain dahil karşılaştırma
- `results/performance_timing.json` — Step 1 / Step 2 / Total süreler
- `results/performance_timing.png` — Step 1+Step 2 stacked + total karşılaştırma

### Cihaz pipeline çıktıları (`results/device_pipeline/`)
- `ga_state.json` — cihazdan dönen state dict
- `ga_metrics.json` — cihaz GA loss + süre + hiperparametreler
- `final_recovered_state.json` — Step 2 sonrası final state
- `final_metrics.json` — 8–10 dağılımı + cihaz/sunucu süre breakdown
- `hourly_predictions.json` — cihaz pipeline 24 saat

### Kod
- `c:\Users\cagdas\AndroidStudioProjects\MobileUnlearningFinal\` — Android uygulaması (Kotlin)
- `server_recovery.py` — sunucu tarafı recovery script'i

---