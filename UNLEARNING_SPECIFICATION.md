# Machine Unlearning in Federated Learning - Implementation Specification

**Document Purpose:** Bu dosya, halihazırda tamamlanmış Federated Learning projesine Machine Unlearning kısımlarını eklemek için gerekli tüm teknik spesifikasyonları içerir.

**Okuyucu:** AI Agent (kod yazacak)  
**Hedef:** Mevcut federated learning koduna unlearning modüllerini entegre etmek  
**İsim:** Machine Unlearning Experiments - Client-Based vs Server-Based Comparison

---

## 📋 PROJE DURUMU

### Halihazırda Tamamlanan (Mevcut Koddaki):

1. **Dataset**: 3 farklı client, her biri belirli saatlerde baskın uygulama kullanımı
   - Client 0: Saat 8-10 arası Instagram baskın
   - Client 1: Saat 8-10 arası Youtube baskın
   - Client 2: Saat 8-10 arası Whatsapp baskın
   - CSV dosyaları: `client_0_data.csv`, `client_1_data.csv`, `client_2_data.csv`

2. **Model Mimarisi**: ANN 2 -> 16 -> 16 -> 4
   - Input: Sin-Cos encoded time (2 feature)
   - Output: 4 app class (Whatsapp, Instagram, Youtube, Linkedin)

3. **Federated Learning**: 6 Round, 10 Epoch/Round
   - ✅ Eğitim tamamlandı
   - ✅ Global model elde edildi
   - ✅ Her client'ın lokal modelleri var
   - ✅ Eğitim sonuçları (loss, accuracy) hesaplandı

### ⚠️ Delta Saklanması - Yüklenim Konusu

**ÖNEMLI:** Mevcut kodda delta saklanıyor mu? **KONTROL GEREK!**

- Eğer **evet**: `delta_history[round][client_id]` mevcut, buradan kullanılacak
- Eğer **hayır**: **AI Agent delta hesaplama kodunu ekleyecek** (Aşama 0)

---

## 🔄 AŞAMA 0 (Ön Koşul) - Delta Saklanması

**SADECE delta saklanmıyorsa yapılmalı!**

### 0.1 Ne İçin?

Unlearning işlemlerinde client'ın tüm etkisini undo etmek için, her round'da ne kadar değiştirdiğini (delta) bilmemiz gerekir.

### 0.2 Nasıl Yapılacak?

**Federated Learning kodu'nda eklenecek:**

```python
# Eğitim öncesi SAKLA
delta_history = {}  # Round başında

for round_t in range(6):
    delta_history[round_t] = {}
    
    for client in clients:
        # SAKLA: Round başında global model state'i
        w_before = deepcopy(global_model.state_dict())
        
        # Eğitim yap
        client.train_epoch(global_model, epochs=10)
        
        # SAKLA: Delta hesapla
        w_after = client.get_weights()
        delta = {param_name: w_after[param_name] - w_before[param_name] 
                 for param_name in w_before}
        
        delta_history[round_t][client.id] = delta
        
        # Sunucuya gönder
        send_to_server(client.weights)
    
    # FedAvg ve güncelle
    global_model = fedavg([...])
```

### 0.3 Çıktı

```python
delta_history = {
    0: {0: {...}, 1: {...}, 2: {...}},
    1: {0: {...}, 1: {...}, 2: {...}},
    ...
    5: {0: {...}, 1: {...}, 2: {...}}
}

# DOSYAYA KAYDET
pickle.dump(delta_history, open('delta_history.pkl', 'wb'))
```

---

## 🎯 YAPILMASI GEREKEN: ÜÇ AŞAMA

### AŞAMA 1: Client-Based Unlearning (FedEraser)

**Algoritma**: FedEraser (Liu et al. 2021)

#### 1.1 Giriş Parametreleri
```
global_model: Son eğitilmiş federated model
delta_history: 6 round × 3 client'ın tüm delta'ları (Aşama 0'dan)
client_id_to_remove: Hangi client silinecek? (0, 1 veya 2)
forget_data: Silinecek client'ın train verisi (CSV'den)
remaining_clients: Kalan 2 client (silinecek hariç)
recovery_rounds: Kaç recovery round? (2-3 öneriliyor)
```

#### 1.2 İş Akışı

**Step 1: Client tarafında eğitim (Gradient Ascent)**

```python
# Client cihazında yapılacak:
# Silinecek client'ın verisiyle, TERS gradientler ile eğitim

for epoch in range(ascent_epochs):  # 5-10 epoch
    for batch in get_batches(forget_data):
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # ÖNEMLI: Gradientleri TERS ÇEVİR
        for param in model.parameters():
            param.grad = -param.grad
        
        optimizer.step()
```

**Step 2: Sunucu tarafında Recovery**

```python
# Sunucu tarafında yapılacak:
# Kalan client'larla 2-3 recovery round

for recovery_round in range(recovery_rounds):
    for client in remaining_clients:  # silinecek client HARIÇ
        # Client eğitim yap (normal eğitim)
        client.train_epoch(global_model, epochs=5)
    
    # Sadece kalan client'ların ağırlıklarından FedAvg
    remaining_weights = [c.get_weights() for c in remaining_clients]
    global_model = fedavg(remaining_weights)
```

#### 1.3 Çıktı

```python
{
    'unlearned_model': model object,
    'ascent_loss_history': [loss_epoch_0, loss_epoch_1, ...],
    'recovery_loss_history': [loss_round_0, loss_round_1, ...],
    'total_time_seconds': float,
    'success': True/False
}
```

#### 1.4 Kod Adı
```python
def client_based_unlearning(
    global_model,
    client_id_to_remove,
    forget_data,  # silinecek client'ın train CSV verisi
    remaining_clients,
    delta_history,  # Aşama 0'dan
    ascent_epochs=5,
    ascent_lr=0.01,
    recovery_rounds=2,
    recovery_epochs=5
) -> Dict:
    """
    Returns: {
        'unlearned_model': model,
        'metrics': {...},
        'loss_history': {...}
    }
    """
```

---

### AŞAMA 2: Server-Based Unlearning (Recovery via Remaining Clients)

**Algoritma**: Server-Standalone Unlearning (Yang & Zhao 2024)

#### 2.1 Giriş Parametreleri
```
global_model: Son eğitilmiş federated model
client_id_to_remove: Hangi client silinecek? (0, 1 veya 2)
remaining_clients: Kalan 2 client (silinecek hariç)
delta_history: Aşama 0'dan (opsiyonel, logging için)
recovery_rounds: Kaç recovery round? (2-3 öneriliyor)
```

#### 2.2 İş Akışı

**Step 1: Delta'lardan tahmini katkıyı göster (opsiyonel)**

```python
# Bilgilendirme amaçlı (gerekli değil, ama iyi haber)
if delta_history:
    total_delta_estimate = sum(
        delta_history[t][client_id_to_remove] 
        for t in delta_history
    )
    print(f"Tahmini delta (silinecek client): {total_delta_estimate}")
```

**Step 2: Recovery round'ları**

```python
# Server tarafında yapılacak:
# Sadece kalan client'larla eğitim

for recovery_round in range(recovery_rounds):
    for client in remaining_clients:  # silinecek client HARIÇ
        # Client eğitim yap (normal eğitim)
        client.train_epoch(global_model, epochs=recovery_epochs)
    
    # Sadece kalan client'ların ağırlıklarından FedAvg
    remaining_weights = [c.get_weights() for c in remaining_clients]
    global_model = fedavg(remaining_weights)
```

#### 2.3 Çıktı

```python
{
    'unlearned_model': model object,
    'recovery_loss_history': [loss_round_0, loss_round_1, ...],
    'total_time_seconds': float,
    'estimated_delta': {...} if delta_history else None,
    'success': True/False
}
```

#### 2.4 Kod Adı
```python
def server_based_unlearning(
    global_model,
    client_id_to_remove,
    remaining_clients,
    delta_history=None,  # Optional, logging için
    recovery_rounds=2,
    recovery_epochs=5
) -> Dict:
    """
    Returns: {
        'unlearned_model': model,
        'metrics': {...},
        'loss_history': {...}
    }
    """
```

---

### AŞAMA 3: Karşılaştırma ve Analiz

#### 3.1 Test Verisi Hazırla

```python
# Silinecek client'ın test verisi
test_data_client_0 = load_csv('client_0_data.csv', split='test')

# Kalan client'ların test verisi
test_data_client_1 = load_csv('client_1_data.csv', split='test')
test_data_client_2 = load_csv('client_2_data.csv', split='test')
```

#### 3.2 Metrikleri Hesapla

**Metric 1: Saat 8-10 Arasındaki Tahminler**

```python
def analyze_8_to_10_predictions(model, test_data):
    """
    Saat 8-10 arasındaki verileri filtrele
    Her uygulama için ortalama olasılık hesapla
    
    Returns:
    {
        'Whatsapp': 0.XX,
        'Instagram': 0.XX,
        'Youtube': 0.XX,
        'Linkedin': 0.XX,
        'sample_count': N  # 8-10 arasında kaç veri noktası
    }
    """
```

**Metric 2: Reduction Yüzdesi**

```python
reduction_percent = (
    (original_prob - unlearned_prob) / original_prob
) * 100

# Açıklama: Eğer pozitifse, tahmini başarılı şekilde azalmıştır
#           Eğer negatifse, tahmini artmıştır (istenmeyen)
```

**Metric 3: Model Doğruluğu (Kalan Client Verisi)**

```python
def calculate_accuracy(model, test_data, client_id):
    """
    Kalan client'ların test verisi üzerinde accuracy
    
    Amaç: Unlearning sonrası doğruluk düşüp düşmediğini kontrol
    
    Returns: accuracy (0.0 - 1.0)
    """
```

**Metric 4: Entropy (İstatistiksel)**

```python
def calculate_entropy(probabilities):
    """
    entropy = -sum(p * log(p))
    
    Yüksek entropy = belirsiz tahmin (iyi, client silinmiş demek)
    Düşük entropy = kesin tahmin (kötü, client'ın etkisi hala var)
    
    Returns: entropy value
    """
```

**Metric 5: Execution Time**

```python
import time

start = time.time()
# unlearning işlemi
end = time.time()

execution_time = end - start  # saniye cinsinden
```

#### 3.3 Karşılaştırma Yap

```python
def compare_unlearning_methods(
    original_global_model,
    client_based_model,
    server_based_model,
    client_id_removed,  # Logging için
    test_data_all  # Silinecek client'ın test verisi
) -> Dict:
    """
    Üç modeli (original, client-based, server-based) karşılaştır
    
    Returns: Tüm metrikleri içeren Dict
    """
    
    results = {
        'original': analyze_8_to_10_predictions(original_global_model, test_data),
        'client_based': analyze_8_to_10_predictions(client_based_model, test_data),
        'server_based': analyze_8_to_10_predictions(server_based_model, test_data),
        'client_id_removed': client_id_removed,
        'comparison': {
            'client_based_instagram_reduction_%': ...,
            'server_based_instagram_reduction_%': ...,
            'kalan_clients_accuracy': {...},
            'execution_times': {...}
        }
    }
    
    return results
```

#### 3.4 Sonuç Çıkar

```python
# İnsan okunabilir karşılaştırma
print("KARŞILAŞTIRMA SONUÇLARI (Gerçek Veriler):")
print(f"Instagram tahmini (8-10):")
print(f"  Orijinal: {results['original']['Instagram']:.1%}")
print(f"  Client-Based: {results['client_based']['Instagram']:.1%}")
print(f"  Server-Based: {results['server_based']['Instagram']:.1%}")
print()
print("Azalış yüzdeleri:")
print(f"  Client-Based: {reduction_cb:.1f}%")
print(f"  Server-Based: {reduction_sb:.1f}%")
print()
print(f"SONUÇ: {'Client-Based' if reduction_cb > reduction_sb else 'Server-Based'} daha etkili")
```

---

## 📊 BEKLENEN ÇIKTI (Gerçek Veriler ile)

### Output 1: JSON Report

```python
unlearning_results.json
{
  "experiment_date": "2024-XX-XX",
  "client_removed": 0,
  "models": {
    "original": {
      "instagram_prob_8_10": 0.312,  # GERÇEK VERİDEN
      "youtube_prob_8_10": 0.284,
      "whatsapp_prob_8_10": 0.175,
      "linkedin_prob_8_10": 0.229,
      "entropy": 1.35
    },
    "client_based_unlearned": {
      "instagram_prob_8_10": 0.152,  # GERÇEK VERİDEN
      "youtube_prob_8_10": 0.385,
      "whatsapp_prob_8_10": 0.321,
      "linkedin_prob_8_10": 0.142,
      "entropy": 1.38,
      "reduction_percent": 51.3,
      "execution_time_s": 2.34
    },
    "server_based_unlearned": {
      "instagram_prob_8_10": 0.185,  # GERÇEK VERİDEN
      "youtube_prob_8_10": 0.362,
      "whatsapp_prob_8_10": 0.283,
      "linkedin_prob_8_10": 0.170,
      "entropy": 1.35,
      "reduction_percent": 40.7,
      "execution_time_s": 0.87
    }
  },
  "remaining_clients_accuracy": {
    "client_1": 0.921,
    "client_2": 0.914
  },
  "conclusion": "Client-Based Unlearning daha etkili"
}
```

### Output 2: Visualization (Matplotlib/PNG)

```
comparison_plot.png
Subplot 1: Instagram tahmini karşılaştırması (bar chart)
Subplot 2: Recovery loss history (line chart)
Subplot 3: Metrics tablosu (text)
Subplot 4: Conclusion (text)

NOT: Gerçek veriler kullanılacak!
```

### Output 3: Console Output (Logs)

```
=====================================================
MACHINE UNLEARNING EXPERIMENTS
=====================================================

Experiment Date: 2024-XX-XX
Client to Remove: 0 (Instagram Dominant)

--------- AŞAMA 0: DELTA SAKLANMASI ---------
✅ Delta history oluşturuldu
   Round 0: 3 client × 2 delta = 6 param groups
   Round 1: 3 client × 2 delta = 6 param groups
   ...
   Round 5: 3 client × 2 delta = 6 param groups

--------- AŞAMA 1: CLIENT-BASED UNLEARNING ---------
Step 1: Gradient Ascent (Client cihazında)
  Epoch 1: loss = 1.389
  Epoch 2: loss = 1.385
  Epoch 3: loss = 1.381
  ...
  Epoch 5: loss = 1.365
  ✅ Client-side tamamlandı

Step 2: Recovery Rounds (Server tarafında)
  Recovery Round 1:
    Client 1 eğitim...
    Client 2 eğitim...
    FedAvg yapıldı
  Recovery Round 2:
    Client 1 eğitim...
    Client 2 eğitim...
    FedAvg yapıldı
  ✅ Recovery tamamlandı (0.87s)

Sonuçlar:
  Instagram prob (8-10): 0.312 → 0.152 (↓ 51.3%)
  Entropy: 1.35 → 1.38 (↑ belirsizlik arttı)
  Remaining Acc: 0.92

--------- AŞAMA 2: SERVER-BASED UNLEARNING ---------
Step 1: Recovery Rounds (Server tarafında)
  Recovery Round 1:
    Client 1 eğitim...
    Client 2 eğitim...
    FedAvg yapıldı
  Recovery Round 2:
    Client 1 eğitim...
    Client 2 eğitim...
    FedAvg yapıldı
  ✅ Recovery tamamlandı (0.87s)

Sonuçlar:
  Instagram prob (8-10): 0.312 → 0.185 (↓ 40.7%)
  Entropy: 1.35 → 1.35 (= belirsizlik değişmedi)
  Remaining Acc: 0.91

--------- AŞAMA 3: KARŞILAŞTIRMA ---------
Instagram Azalış:
  Client-Based: 51.3% ✅ (DAHA İYİ)
  Server-Based: 40.7%

Execution Time:
  Client-Based: 2.34s
  Server-Based: 0.87s (DAHA HIZLI)

Remaining Client Accuracy:
  Client 1: 0.92 vs 0.91 (Client-Based biraz iyi)
  Client 2: 0.91 vs 0.91 (Aynı)

SONUÇ:
Client-Based Unlearning DAHA BAŞARILI
(Instagram etkisini %51.3 oranında sildi)

AÇIKLAMA:
Client-Based gradient ascent yöntemi, client'ın 
verisi üzerinde doğrudan loss'ı maksimize ederek 
daha radikal bir silme işlemi yapıyor.
Server-Based recovery yöntemi ise daha temkinli,
kalan client'ların bilgisini koruyarak ilerliyor.

=====================================================
```

---

## 🔧 ENTEGRASYON ÖNERİSİ

### Mevcut Kod Yapısı (Tahmini)
```
project/
├── data/
│   ├── client_0_data.csv
│   ├── client_1_data.csv
│   └── client_2_data.csv
├── federated_learning.py (✅ TAMAMLANDI)
│   ├── def load_data()
│   ├── def create_model()
│   ├── def train_federated_learning()
│   └── [ÇIKTI] global_model, clients
│
└── [YAPILACAK] unlearning_experiments.py (EKLENECEK)
```

### Yeni Yapı (İstenen)
```
project/
├── data/
│   ├── client_0_data.csv
│   ├── client_1_data.csv
│   └── client_2_data.csv
│
├── federated_learning.py (✅ TAMAMLANDI)
│   ├── def load_data()
│   ├── def create_model()
│   ├── def train_federated_learning()
│   ├── [AŞAMA 0] def save_delta_history()  ← NEW (eğer yoksa)
│   └── [ÇIKTI] global_model, clients, delta_history
│
├── unlearning_experiments.py (✨ YENİ - AI AGENT YAZACAK)
│   ├── # AŞAMA 1: CLIENT-BASED
│   ├── def client_based_unlearning(...)
│   │
│   ├── # AŞAMA 2: SERVER-BASED
│   ├── def server_based_unlearning(...)
│   │
│   ├── # AŞAMA 3: KARŞILAŞTIRMA
│   ├── def analyze_8_to_10_predictions(...)
│   ├── def compare_unlearning_methods(...)
│   └── def generate_results(...)
│
├── main.py (GÜNCELLENMİŞ - AI AGENT GÜNCELLEYECEK)
│   ├── global_model, clients, delta_history = federated_learning()
│   ├── client_based_results = client_based_unlearning(...)
│   ├── server_based_results = server_based_unlearning(...)
│   ├── comparison_results = compare_unlearning_methods(...)
│   └── generate_report(...)
│
├── results/
│   ├── unlearning_results.json  ← AI AGENT OLUŞTURACAK
│   └── comparison_plot.png      ← AI AGENT OLUŞTURACAK
│
└── UNLEARNING_SPECIFICATION.md (BU DOSYA)
```

---

## 🎬 ÇALIŞMA AKIŞI (AI Agent'ın Yapacağı)

```
1. KOD İNCELEMESİ
   ✓ federated_learning.py oku
   ✓ Model mimarisi anla
   ✓ Client'lar nasıl çalışıyor anla
   ✓ Mevcut output'lar neler anla

2. AŞAMA 0 KONTROLÜ
   ✓ Delta history mevcut mu? KONTROL ET
   ✓ Yoksa delta hesaplama kodunu ekle

3. AŞAMA 1: CLIENT-BASED YAZ
   ✓ Gradient ascent implement et
   ✓ Recovery rounds implement et
   ✓ Loss history kaydet

4. AŞAMA 2: SERVER-BASED YAZ
   ✓ Recovery rounds implement et
   ✓ Loss history kaydet

5. AŞAMA 3: KARŞILAŞTIRMA YAZ
   ✓ Saat 8-10 analiz fonksiyonu
   ✓ Metrikleri hesapla
   ✓ Karşılaştırma yap

6. main.py'yi GÜNCELLE
   ✓ Tüm fonksiyonları çağır
   ✓ Resultları topla

7. REPORT OLUŞTUR
   ✓ JSON dosyası yaz
   ✓ PNG graph oluştur
   ✓ Console logs bastır

8. ÇALIŞT & TEST ET
   python main.py
```

---

## 📌 KRİTİK NOKTALAR

### Client-Based İçin

**ZORUNLU:**
- ✅ Silinecek client'ın train verisi (CSV'den)
- ✅ Mevcut global_model
- ✅ Kalan 2 client'in local modelleri
- ✅ Delta history (Aşama 0'dan)

**DİKKAT:**
- ⚠️ Gradient ascent'i doğru yap: `param.grad = -param.grad`
- ⚠️ Loss'u MAKSİMİZE etmek istiyorsun, minimize değil
- ⚠️ Recovery rounds SADECE kalan client'larla (0 hariç)
- ⚠️ Recovery FedAvg'de silinecek client OLMAMALI

### Server-Based İçin

**ZORUNLU:**
- ✅ Kalan client'ların train edilen local modelleri
- ✅ Mevcut global_model

**DİKKAT:**
- ⚠️ FedAvg'ye silinecek client EKLEME, sadece kalan client'ları topla
- ⚠️ Recovery rounds zamanı gerekli (2-3 round)

### Test Verisi

- ✅ **Test verisi**: Silinecek client'ın train datası (aynı CSV)
  - Eğitimde kullanılan verinin aynısını test olarak kullan
  - Amaç: Saat 8-10'da ne kadar azaldığını görmek

---

## 📚 REFERANSLAR

1. **FedEraser** - Liu et al. 2021
   - "Federated Unlearning via Client-Side Gradient Ascent"
   - Gradient ascent yöntemi

2. **Recovery-Based** - Yang & Zhao 2024
   - "Federated Unlearning with Gradient Descent and Conflict Mitigation"
   - Server-based recovery yaklaşımı

3. **Gradient Ascent** - Goel et al. 2023
   - "Machine Unlearning via Gradient Ascent"
   - Theoretik temeller

4. **Survey** - Yang et al. 2024
   - "A Survey on Federated Unlearning"
   - Kapsamlı literatür

---

## ✅ SUCCESS CRITERIA

✅ **Client-Based Unlearning başarılı sayılır eğer:**
- Instagram tahmini saat 8-10'da **önemli ölçüde azaldıysa** (gerçek veriler)
- Kalan client'ların accuracy >85%
- Kod çalışır ve hata vermez
- Metrics hesaplanmış ve kaydedilmiş

✅ **Server-Based Unlearning başarılı sayılır eğer:**
- Instagram tahmini saat 8-10'da **bazı ölçüde azaldıysa** (gerçek veriler)
- Kalan client'ların accuracy >85%
- Kod çalışır ve hata vermez
- Metrics hesaplanmış ve kaydedilmiş

✅ **Karşılaştırma başarılı sayılır eğer:**
- JSON report oluşturulmuş
- PNG visualizasyon oluşturulmuş
- Hangi yöntemin daha başarılı olduğu belirtilmiş
- İnsan okunabilir log çıktısı var

---

## 🎯 ÖZET

**AI Agent'ın Görevleri:**

1. ✅ Mevcut federated learning kodunu anla
2. ✅ Delta history kontrolü yap (varsa kullan, yoksa ekle)
3. ✅ Client-based unlearning implement et
4. ✅ Server-based unlearning implement et
5. ✅ Karşılaştırma ve analiz yap
6. ✅ Report oluştur ve kaydet

**ÖNEMLI HATIRLATMA:**
- Tüm sonuçlar **gerçek verilerden** çıkacak
- Örnek sayılar yukarıda ÖRNEK olarak verilmiştir
- Projenin gerçek CSV verilerine ve eğitim sonuçlarına erişe bilecek
- Kod hata yönetimi ekle (try-catch), sonuçları JSON ve PNG olarak kaydet

---

**Bu specification'ı okuyarak, projeyi anlamalı ve gerekli unlearning kodlarını yazabilmelisin! İyi şanslar!** 🚀
