# Machine Unlearning in Federated Learning - CORRECTED Specification

**Önceki Hatalar Düzeltildi!**
- ❌ FedEraser client-based diye sunulmuştu → **Server-based'dir!**
- ❌ Recovery yöntemi önerilmişti → **Saçmalık, unutun!**
- ✅ Net tanımlar yapıldı

---

## 🎯 NET TANIMLAR

### CLIENT-BASED UNLEARNING
Silmek istenen **client'ın kendi cihazında** saklı olan veriler kullanılır:
- Kendi train datası
- Kendi epoch-by-epoch lokal updates
- Kendi model snapshots

**Server'da saklanması gereken:** Hiçbir şey! (Privacy!)

### SERVER-BASED UNLEARNING  
Server'ın **erişebildiği** veriler kullanılır:
- Tüm client'ların round-by-round updates (deltas)
- Global model snapshots
- Agregasyon history

**Client'ın bilmesi gereken:** Hiçbir şey! (Server işler)

---

## 🔄 YAPILMASI GEREKEN: İKİ YÖNTEM

### YÖNTEM 1: SERVER-BASED UNLEARNING (FedEraser - Liu et al. 2021)

#### 1.1 Ne Yapılır?

```
Deltas Saklanması (Eğitim esnasında):
  for round in range(6):
      for client in clients:
          delta = client_final_weights - global_model_start
          delta_history[round][client_id] = delta
      global = fedavg(...)

Unlearning (Silme işlemi):
  client_id = 0  # Silinecek
  total_delta = sum(delta_history[t][0] for all rounds t)
  global_weights -= total_delta
  BITTI! Client silinmiş!
```

#### 1.2 Giriş Parametreleri

```python
- global_model: Son eğitilmiş federated model
- delta_history: Saklanmış deltas (her round, her client)
                 delta_history[round][client_id] = Δw
- client_id_to_remove: Hangi client silinecek (0, 1, 2)
```

#### 1.3 Algoritma Adımları

**Step 1: Deltas saklanıp saklanmadığını kontrol et**
```python
if delta_history exists:
    print("✅ Deltas mevcut, Server-based unlearning yapılabilir")
else:
    print("❌ Deltas yoksa, eğitim kodundan çıkar ve sakla")
```

**Step 2: Silinecek client'ın tüm katkısını hesapla**
```python
total_delta = None
for round_t in delta_history:
    delta_t = delta_history[round_t][client_id_to_remove]
    if total_delta is None:
        total_delta = copy(delta_t)
    else:
        total_delta += delta_t  # Kümülatif topla
```

**Step 3: Ters yönde uygula (unlearn)**
```python
for param_name in global_model.state_dict():
    global_model.state_dict()[param_name] -= total_delta[param_name]

# BİTTİ! Unlearned model hazır!
```

#### 1.4 Kod Şablonu

```python
def server_based_unlearning(
    global_model,
    delta_history,
    client_id_to_remove
):
    """
    Server tarafında yapılır.
    Client'ın katkısını mathematically tersine çevir.
    
    Args:
        global_model: Trained global model
        delta_history: Dict[round][client_id] = delta weights
        client_id_to_remove: int (0, 1, or 2)
    
    Returns:
        unlearned_model: Model with client removed
        metrics: {'total_delta': ..., 'removed_client': ...}
    """
    
    import copy
    
    # 1. Total delta hesapla
    total_delta = {}
    for round_t in delta_history:
        delta_t = delta_history[round_t][client_id_to_remove]
        
        for param_name, param_value in delta_t.items():
            if param_name not in total_delta:
                total_delta[param_name] = copy.deepcopy(param_value)
            else:
                total_delta[param_name] += param_value
    
    # 2. Ters yönde uygula
    unlearned_model = copy.deepcopy(global_model)
    for param_name in unlearned_model.state_dict():
        unlearned_model.state_dict()[param_name] -= total_delta[param_name]
    
    # 3. Return
    metrics = {
        'method': 'FedEraser (Server-Based)',
        'client_removed': client_id_to_remove,
        'total_delta_magnitude': sum(p.norm().item() for p in total_delta.values())
    }
    
    return unlearned_model, metrics
```

#### 1.5 Avantajlar & Dezavantajları

| Avantaj | Dezavantaj |
|---------|-----------|
| ✅ Çok hızlı (matrix subtract) | ⚠️ Deltas saklanmalı |
| ✅ Exact (matematiksel) | ⚠️ Server overhead |
| ✅ Ölçeklenebilir | ⚠️ Deltas yoksa yapılamaz |
| ✅ Basit kod | |

---

### YÖNTEM 2: CLIENT-BASED UNLEARNING (Gradient Ascent)

#### 2.1 Ne Yapılır?

```
Client tarafında (silinecek client'in cihazında):
  1. Kendi train datası D_i yükle
  2. Loss'u MAXIMIZE et (param.grad = -param.grad)
  3. 5-10 epoch eğit
  4. Güncellenmiş weights server'a gönder

Server tarafında:
  5. Normal FedAvg yap (hiçbir unlearning kodu yok)
  6. Bitti!
```

#### 2.2 Giriş Parametreleri

```python
- client_model: Client'ın lokal modeli
- forget_data: Silinecek client'ın train datası (CSV'den)
- learning_rate: 0.01 (önerilen)
- epochs: 5-10 (önerilen)
```

#### 2.3 Algoritma Adımları

**Step 1: Client'in datası yükle**
```python
forget_data = load_csv(f'client_{client_id}_data.csv')
# forget_data = [(x1, y1), (x2, y2), ...]
```

**Step 2: Gradient ascent başlat**
```python
for epoch in range(5):
    for batch_x, batch_y in get_batches(forget_data):
        # Forward pass
        outputs = client_model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # ÖNEMLI: Gradients'i TERS ÇEVİR
        for param in client_model.parameters():
            param.grad = -param.grad  # ← KEY STEP!
        
        # Step (ters yönde)
        optimizer.step()
```

**Step 3: Güncellenmiş model server'a gönder**
```python
updated_weights = client_model.state_dict()
send_to_server(updated_weights)
# Server normal FedAvg yapacak
```

#### 2.4 Kod Şablonu

```python
def client_based_unlearning(
    client_model,
    forget_data,  # CSV'den yüklenen data
    learning_rate=0.01,
    epochs=5
):
    """
    Client cihazında yapılır.
    Kendi verisi ile unlearning.
    
    Args:
        client_model: Model (eğitilmiş global model kopyası)
        forget_data: List of (x, y) tuples (client'ın train datası)
        learning_rate: float
        epochs: int
    
    Returns:
        unlearned_model: Unlearned weights
        metrics: {'epochs': ..., 'final_loss': ...}
    """
    
    import torch
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        client_model.parameters(),
        lr=learning_rate
    )
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Mini-batch gradient ascent
        for batch_x, batch_y in get_batches(forget_data, batch_size=32):
            # Forward
            outputs = client_model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # FLIP GRADIENTS (maximize instead of minimize)
            for param in client_model.parameters():
                if param.grad is not None:
                    param.grad = -param.grad
            
            # Step
            optimizer.step()
        
        avg_loss = epoch_loss / len(get_batches(forget_data))
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    # Return unlearned model
    metrics = {
        'method': 'Gradient Ascent (Client-Based)',
        'epochs': epochs,
        'final_loss': loss_history[-1],
        'loss_history': loss_history
    }
    
    return client_model, metrics
```

#### 2.5 Avantajlar & Dezavantajları

| Avantaj | Dezavantaj |
|---------|-----------|
| ✅ Privacy-preserving (client only) | ⚠️ Biraz yavaş (5-10 epoch) |
| ✅ Server hiçbir şey saklamaz | ⚠️ Client'ın datası gerekli |
| ✅ Client tam kontrol | ⚠️ Client cihaz güçlü olmalı |
| ✅ Hiçbir ön koşul yok | ⚠️ Approximate (exact değil) |

---

## 📊 KARŞILAŞTIRMA

| Kriter | Server-Based (FedEraser) | Client-Based (Gradient Ascent) |
|--------|--------------------------|--------------------------------|
| **Nerede yapılır** | Server | Client cihazı |
| **Gereken data** | Saklanan deltas | Kendi train datası |
| **Server overhead** | Deltas depolama | Hiçbir şey |
| **Hız** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐ Orta (epochs) |
| **Privacy** | ⭐⭐⭐⭐ İyi | ⭐⭐⭐⭐⭐ Mükemmel |
| **Komplekslik** | ⭐ Çok basit | ⭐⭐ Basit |
| **Exact/Approximate** | Exact | Approximate |
| **Ölçeklenebilirlik** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🔧 PROJE ENTEGRASYONU

### Mevcut Kod Yapısı
```
project/
├── federated_learning.py (✅ TAMAMLANDI)
├── data/
│   ├── client_0_data.csv
│   ├── client_1_data.csv
│   └── client_2_data.csv
└── [YAPILACAK] unlearning.py
```

### Yeni Yapı
```
project/
├── federated_learning.py
│   ├── def load_data()
│   ├── def train_federated_learning()
│   ├── [KONTROL] delta_history saklanıyor mu?
│   └── [ÇIKTI] global_model, delta_history, clients
│
├── unlearning.py (YENİ - AI YAZACAK)
│   ├── def server_based_unlearning(...)
│   └── def client_based_unlearning(...)
│
├── main.py (GÜNCELLENMIŞ)
│   ├── global_model, delta_history = federated_learning()
│   ├── server_unlearned = server_based_unlearning(...)
│   ├── client_unlearned = client_based_unlearning(...)
│   ├── compare(original, server_unlearned, client_unlearned)
│   └── report.json & plot.png oluştur
│
└── results/
    ├── unlearning_results.json
    └── comparison_plot.png
```

---

## 📋 AI AGENT'A VERICEK İSTEK

```
"UNLEARNING SPECIFICATION dosyasını oku.

Yapılması gerekenler:

1. KONTROL: Delta history mevcut mu?
   - Eğer evet: devam et
   - Eğer hayır: federated_learning.py'dan çıkar ve sakla

2. SERVER-BASED (FedEraser):
   def server_based_unlearning(global_model, delta_history, client_id)
   - Tüm delta'ları topla
   - Ters yönde uygula
   - Bitti!

3. CLIENT-BASED (Gradient Ascent):
   def client_based_unlearning(client_model, forget_data, epochs=5)
   - CSV'den datayı yükle
   - 5 epoch gradient ascent (loss maximize)
   - param.grad = -param.grad
   - Updated model return et

4. KARŞILAŞTIRMA:
   - Saat 8-10'da Instagram tahmini karşılaştır
   - Metrics hesapla (accuracy, entropy, time)
   - JSON ve PNG rapor oluştur

5. SONUÇ:
   Hangi yöntem daha başarılı?
"
```

---

## ✅ SUCCESS CRITERIA

✅ **Server-Based (FedEraser):**
- Kod çalışır
- Instagram tahmini azalırsa iyi
- JSON rapor var

✅ **Client-Based (Gradient Ascent):**
- Kod çalışır
- Instagram tahmini azalırsa iyi
- Loss history var

✅ **Karşılaştırma:**
- İki yöntem karşılaştırılmış
- Hangi biri daha etkili belirtilmiş

---

## 🎯 ÖZET

**Server-Based (FedEraser):**
- Fast (matrix subtract)
- Exact math
- Needs: delta_history

**Client-Based (Gradient Ascent):**
- Privacy-first
- Client-only
- Needs: forget_data

**İkisini de implement et, karşılaştır!**

---

**Bu specification'ı oku ve kod yaz!** 🚀
