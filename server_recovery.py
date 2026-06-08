"""
Server-side recovery for the on-device gradient ascent unlearning pipeline.

Takes the GA state dict produced by the Android app (cihaz Step 1 ciktisi) and runs
the EXACT SAME Step 2 (3 recovery rounds x 5 epochs over user2 + user3) that the
notebook's `client_based_gradient_ascent` function applies in-process.

Usage:
    python server_recovery.py \
        --ga-state results/device_pipeline/ga_state.json \
        --ga-metrics results/device_pipeline/ga_metrics.json \
        --output-dir results/device_pipeline/

Outputs (in --output-dir):
    final_recovered_state.json   final state dict
    final_metrics.json           8-10 olasiliklari, recovery loss history, sureler
    hourly_predictions.json      24-saat tahmin tablosu (4 ana app)
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Constants  --- notebook ile birebir aynı
# ============================================================================
SEED = 42
NUM_ROUNDS = 6
DAYS_PER_ROUND = 5

APP_TO_IDX = {
    "com.instagram.android": 0,
    "com.google.android.youtube": 1,
    "com.linkedin.android": 2,
    "com.whatsapp": 3,
}
IDX_TO_LABEL = {0: "Instagram", 1: "YouTube", 2: "LinkedIn", 3: "WhatsApp"}
LOCAL_TZ = "Europe/Istanbul"

USER_FILES = {
    "user1": "app_usage_data_user1.csv",
    "user2": "app_usage_data_user2.csv",
    "user3": "app_usage_data_user3.csv",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Model + helpers  --- notebook'taki tanımların birebir kopyası
# ============================================================================
class AppUsageModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


def local_train(global_state_dict, X, y, local_epochs=20, batch_size=32, lr=0.01):
    model = AppUsageModel().to(device)
    model.load_state_dict(copy.deepcopy(global_state_dict))

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return model.state_dict(), acc, len(X)


def fed_avg(client_state_dicts, client_sizes):
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    avg_state = {}
    for key in client_state_dicts[0]:
        avg_state[key] = sum(
            w * client_state_dicts[i][key].float()
            for i, w in enumerate(weights)
        )
    return avg_state


def load_full_data(csv_path):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["label"] = df["packageName"].map(APP_TO_IDX)
    return df


def split_by_days(df, days_per_round=5):
    all_days = sorted(df["date"].unique())
    windows = []
    for i in range(0, len(all_days), days_per_round):
        window_days = all_days[i:i + days_per_round]
        window_df = df[df["date"].isin(window_days)]
        X = window_df[["sin_hour", "cos_hour"]].values.astype(np.float32)
        y = window_df["label"].values.astype(np.int64)
        windows.append({"X": X, "y": y, "days": window_days, "n_records": len(X)})
    return windows


def analyze_8_to_10(model, csv_path):
    df = load_full_data(csv_path)
    df = df[(df["hour"] >= 8) & (df["hour"] < 10)]
    X = df[["sin_hour", "cos_hour"]].values.astype(np.float32)
    if len(X) == 0:
        return {"Instagram": 0.0, "YouTube": 0.0, "LinkedIn": 0.0, "WhatsApp": 0.0, "sample_count": 0}
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()
    avg = probs.mean(axis=0)
    return {
        "Instagram": float(avg[0]),
        "YouTube":   float(avg[1]),
        "LinkedIn":  float(avg[2]),
        "WhatsApp":  float(avg[3]),
        "sample_count": int(len(X)),
    }


def predict_for_hours(model, hours):
    model.eval()
    rows = []
    for h in hours:
        x = torch.FloatTensor([[
            math.sin(2 * math.pi * h / 24.0),
            math.cos(2 * math.pi * h / 24.0),
        ]]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        rows.append({
            "hour": h,
            "prediction": IDX_TO_LABEL[idx],
            "confidence": float(probs[idx]),
            "probs": {
                "Instagram": float(probs[0]),
                "YouTube":   float(probs[1]),
                "LinkedIn":  float(probs[2]),
                "WhatsApp":  float(probs[3]),
            },
        })
    return rows


# ============================================================================
# Recovery pipeline
# ============================================================================
def load_device_state(path: Path) -> dict:
    """Cihazdan gelen JSON state dict'i torch.Tensor olarak yükle."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        out[k] = torch.tensor(v, dtype=torch.float32)
    return out


def run_recovery(ga_state: dict, recovery_rounds: int = 3, recovery_epochs: int = 5,
                 remaining_users=("user2", "user3")) -> dict:
    """Notebook'taki client_based_gradient_ascent Step 2 ile birebir aynı."""
    user_windows = {}
    for u in remaining_users:
        df = load_full_data(USER_FILES[u])
        user_windows[u] = split_by_days(df, DAYS_PER_ROUND)

    recovery_state = copy.deepcopy(ga_state)
    loss_history = []

    print(f"\nStep 2: Recovery Rounds ({recovery_rounds} round, {recovery_epochs} epoch/round)")
    for rr in range(recovery_rounds):
        print(f"  Recovery Round {rr+1}/{recovery_rounds}:")
        rec_states = []
        rec_sizes = []
        round_losses = []
        for u in remaining_users:
            X_u = np.vstack([user_windows[u][r]["X"] for r in range(NUM_ROUNDS)])
            y_u = np.hstack([user_windows[u][r]["y"] for r in range(NUM_ROUNDS)])
            state, acc, n = local_train(
                recovery_state, X_u, y_u,
                local_epochs=recovery_epochs, batch_size=32, lr=0.01,
            )
            rec_states.append(state)
            rec_sizes.append(n)
            m = AppUsageModel().to(device)
            m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                out = m(torch.FloatTensor(X_u).to(device))
                loss = nn.CrossEntropyLoss()(out, torch.LongTensor(y_u).to(device)).item()
            round_losses.append(loss)
            print(f"    {u}: acc={acc*100:.1f}%, loss={loss:.4f}")
        recovery_state = fed_avg(rec_states, rec_sizes)
        avg_loss = sum(round_losses) / len(round_losses)
        loss_history.append(avg_loss)
        print(f"    FedAvg | avg loss: {avg_loss:.4f}")

    return {"state": recovery_state, "loss_history": loss_history}


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga-state", required=True, type=Path)
    parser.add_argument("--ga-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recovery-rounds", type=int, default=3)
    parser.add_argument("--recovery-epochs", type=int, default=5)
    parser.add_argument("--forget-user", default="user1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determinism for the recovery step
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  SERVER RECOVERY")
    print("=" * 60)
    print(f"  GA state    : {args.ga_state}")
    print(f"  GA metrics  : {args.ga_metrics}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Forget user : {args.forget_user}")
    print(f"  Recovery    : {args.recovery_rounds} round x {args.recovery_epochs} epoch")

    # Load device GA output
    with open(args.ga_metrics, "r", encoding="utf-8") as f:
        ga_metrics = json.load(f)
    ga_state = load_device_state(args.ga_state)
    print(f"\n  Cihaz GA suresi : {ga_metrics['total_time_seconds']:.3f} s")
    print(f"  Cihaz GA n      : {ga_metrics['n_records']}")
    print(f"  Cihaz GA loss   : {ga_metrics['loss_history'][0]:.4f} -> "
          f"{ga_metrics['loss_history'][-1]:.4f}")

    # Step 2 — recovery
    t0 = time.time()
    rec = run_recovery(
        ga_state,
        recovery_rounds=args.recovery_rounds,
        recovery_epochs=args.recovery_epochs,
    )
    recovery_time = time.time() - t0

    # Evaluate on 8-10 window of forget user's data
    final_model = AppUsageModel().to(device)
    final_model.load_state_dict(rec["state"])
    final_model.eval()

    probs_810 = analyze_8_to_10(final_model, USER_FILES[args.forget_user])
    print("\n  8-10 (user1 verisinde) ortalama olasiliklar:")
    for k in ("Instagram", "YouTube", "LinkedIn", "WhatsApp"):
        print(f"    {k:10s}: {probs_810[k]*100:5.1f}%")

    # 24-hour predictions
    hourly = predict_for_hours(final_model, list(range(24)))

    # Persist
    final_state_path = args.output_dir / "final_recovered_state.json"
    with open(final_state_path, "w", encoding="utf-8") as f:
        json.dump({k: v.cpu().tolist() for k, v in rec["state"].items()}, f, indent=2)

    final_metrics = {
        "forget_user": args.forget_user,
        "device_ga": {
            "total_time_seconds": ga_metrics["total_time_seconds"],
            "n_records":          ga_metrics["n_records"],
            "loss_history":       ga_metrics["loss_history"],
            "lr":                 ga_metrics["lr"],
            "epochs":             ga_metrics["epochs"],
            "batch_size":         ga_metrics["batch_size"],
            "max_norm":           ga_metrics["max_norm"],
            "seed":               ga_metrics["seed"],
            "optimizer":          ga_metrics["optimizer"],
        },
        "server_recovery": {
            "rounds":               args.recovery_rounds,
            "epochs_per_round":     args.recovery_epochs,
            "loss_history":         rec["loss_history"],
            "total_time_seconds":   recovery_time,
        },
        "evaluation_8_10": probs_810,
        "pipeline_total_time_seconds": (
            ga_metrics["total_time_seconds"] + recovery_time
        ),
    }
    metrics_path = args.output_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    hourly_path = args.output_dir / "hourly_predictions.json"
    with open(hourly_path, "w", encoding="utf-8") as f:
        json.dump(hourly, f, indent=2, ensure_ascii=False)

    print(f"\n  Yazıldı:")
    print(f"    {final_state_path}")
    print(f"    {metrics_path}")
    print(f"    {hourly_path}")
    print(f"\n  Pipeline toplam suresi: "
          f"{ga_metrics['total_time_seconds'] + recovery_time:.3f} s "
          f"(cihaz: {ga_metrics['total_time_seconds']:.3f} s + "
          f"sunucu: {recovery_time:.3f} s)")


if __name__ == "__main__":
    main()
