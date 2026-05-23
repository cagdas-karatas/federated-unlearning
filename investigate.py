
import numpy as np, torch, torch.nn as nn, copy, math, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device    = torch.device("cpu")
LOCAL_TZ  = "Europe/Istanbul"
APP_TO_IDX = {"com.instagram.android":0,"com.google.android.youtube":1,
               "com.linkedin.android":2,"com.whatsapp":3}
IDX_TO_APP = {0:"Instagram",1:"YouTube",2:"LinkedIn",3:"WhatsApp"}

def load_df(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    df["hour"]     = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    df["sin_h"]    = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_h"]    = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["label"]    = df["packageName"].map(APP_TO_IDX)
    df["date"]     = df["datetime"].dt.date
    return df

users = ["user1", "user2", "user3"]
paths = {u: f"app_usage_data_{u}.csv" for u in users}
dfs   = {u: load_df(p) for u, p in paths.items()}

# ── 1. Her kullanıcının TOPLAM label dağılımı ────────────────
print("=" * 60)
print("  1. TÜM VERİ LABEL DAGILIMI")
print("=" * 60)
for u in users:
    df   = dfs[u]
    n    = len(df)
    print(f"\n{u} (toplam {n} kayıt):")
    for lab in range(4):
        cnt = (df["label"] == lab).sum()
        print(f"  {IDX_TO_APP[lab]:12s}: {cnt:4d}  ({cnt/n*100:.1f}%)")

# ── 2. 8-10 saatlerindeki kayıt sayıları ve oran ───────────────
print()
print("=" * 60)
print("  2. 8-10 SAATLERİ KAYIT DAGILIMI")
print("=" * 60)
for u in users:
    df   = dfs[u]
    win  = df[(df["hour"] >= 8) & (df["hour"] < 10)]
    n_all = len(df)
    n_win = len(win)
    print(f"\n{u}: 8-10 = {n_win} kayıt / toplam = {n_all} (%{n_win/n_all*100:.1f})")
    vc = win["label"].value_counts()
    for lab, cnt in vc.items():
        print(f"  {IDX_TO_APP[lab]:12s}: {cnt:4d}  ({cnt/n_win*100:.0f}%)")

# ── 3. FedAvg round ağırlıkları ───────────────────────────────
print()
print("=" * 60)
print("  3. FedAvg ROUND AĞIRLIKLARI")
print("=" * 60)
all_days = {u: sorted(dfs[u]["date"].unique()) for u in users}
print(f"\n{'Round':>5}  {'user1':>7}  {'user2':>7}  {'user3':>7}  {'n_u1':>5}  {'n_u2':>5}  {'n_u3':>5}")
for r in range(6):
    n = {}
    for u in users:
        rdays = all_days[u][r*5:(r+1)*5]
        n[u]  = len(dfs[u][dfs[u]["date"].isin(rdays)])
    total = sum(n.values())
    w = {u: n[u]/total for u in users}
    print(f"  R{r}    {w['user1']:.3f}    {w['user2']:.3f}    {w['user3']:.3f}   {n['user1']:5d}  {n['user2']:5d}  {n['user3']:5d}")

# ── 4. 8-10'daki FedAvg etkin sinyal analizi ──────────────────
print()
print("=" * 60)
print("  4. 8-10 SAATİ FedAvg ETKİN SİNYAL ANALİZİ")
print("=" * 60)
print("\nHer round, 8-10 saati için effective sinyal katkısı:")
print("(client_8_10_samples / round_total × 100 = modele verilen 'instagram/yt/wa' baskısı)")
print()
for r in range(6):
    n_round = {}
    n_810   = {}
    for u in users:
        rdays       = all_days[u][r*5:(r+1)*5]
        df_r        = dfs[u][dfs[u]["date"].isin(rdays)]
        df_r_win    = df_r[(df_r["hour"] >= 8) & (df_r["hour"] < 10)]
        n_round[u]  = len(df_r)
        n_810[u]    = len(df_r_win)
    total_r = sum(n_round.values())
    print(f"  Round {r}:")
    for u, dominant in [("user1","Instagram"),("user2","YouTube"),("user3","WhatsApp")]:
        effective_pct = n_810[u] / total_r * 100
        print(f"    {u} ({dominant:12s}): {n_810[u]:3d} kayıt -> effective = %{effective_pct:.2f} of round")

# ── 5. Toplam efektif baskı özeti ─────────────────────────────
print()
print("=" * 60)
print("  5. 6 ROUND TOPLAMI EFEKTİF BASKI ÖZET")
print("=" * 60)
effective_total = {u: 0 for u in users}
grand_total = 0
for r in range(6):
    n_round = {}
    n_810   = {}
    for u in users:
        rdays       = all_days[u][r*5:(r+1)*5]
        df_r        = dfs[u][dfs[u]["date"].isin(rdays)]
        df_r_win    = df_r[(df_r["hour"] >= 8) & (df_r["hour"] < 10)]
        n_round[u]  = len(df_r)
        n_810[u]    = len(df_r_win)
    total_r = sum(n_round.values())
    grand_total += total_r
    for u in users:
        effective_total[u] += n_810[u]

print(f"\n{'User':>6}  {'8-10 kayıt':>10}  {'Toplam':>8}  {'Oran':>6}")
for u, dominant in [("user1","Instagram"),("user2","YouTube"),("user3","WhatsApp")]:
    print(f"  {u}  {effective_total[u]:10d}  {grand_total:8d}  {effective_total[u]/grand_total*100:5.2f}%  -> {dominant}")

print()
print("Bu oranlar, modelin 8-10 saatinde her app için 'ne kadar ağırlık gördüğünü' gösterir.")
print("FedAvg, tüm saatleri birlikte değerlendirdiğinden 8-10 sinyali zaten zayıf kalır.")
print("YouTube %100 dominant olmasına rağmen model %31 Instagram görmesinin nedeni budur.")

# ── 6. Model için 8-10 ve diğer saatler arasındaki rekabet ────
print()
print("=" * 60)
print("  6. SAAT ARALIĞINA GÖRE REKABET (user2)")
print("=" * 60)
df2 = dfs["user2"]
for h_s, h_e in [(0,4),(4,8),(8,10),(10,14),(14,18),(18,22),(22,24)]:
    win = df2[(df2["hour"]>=h_s)&(df2["hour"]<h_e)]
    if len(win) == 0: continue
    top_app = IDX_TO_APP[win["label"].mode()[0]]
    pct     = (win["label"] == win["label"].mode()[0]).mean() * 100
    print(f"  {h_s:02d}-{h_e:02d}: {len(win):5d} kayıt | dominant={top_app:12s} ({pct:.0f}%)")
