
#!/usr/bin/env python3
"""
HE-ML workstation trainer.

Downloads 10 real-world time-series datasets, trains the repository's
NumPy LSTM implementation, evaluates chronologically, saves weights,
and runs a local inference + HE round-trip check.

Run from the HE-ML repository root:
  python workstation_train.py download --all
  python workstation_train.py train --dataset ett_m1 --epochs 5 --max-points 3000
  python workstation_train.py infer --dataset ett_m1 --scheme ckks
  python workstation_train.py benchmark --datasets all --epochs 3 --max-points 1500
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd

from lstm_pkg import LSTMSequencePredictor
from lstm_pkg.datasets.manager import TimeSeriesDatasetManager
from multi_scheme_he.context import MultiSchemeHEContext


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "real_datasets"
MODEL_DIR = ROOT / "models" / "workstation_lstm"
RESULT_DIR = ROOT / "results" / "workstation"

DATASETS: Dict[str, Dict[str, Any]] = {
    "ett_m1": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv",
        "filename": "ETTm1.csv",
        "target": "OT",
        "description": "Electricity Transformer Temperature, 15-minute sampling",
    },
    "ett_m2": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm2.csv",
        "filename": "ETTm2.csv",
        "target": "OT",
        "description": "Electricity Transformer Temperature, 15-minute sampling",
    },
    "h2o": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/h2o.csv",
        "filename": "h2o.csv",
        "target": None,
        "description": "Monthly Australian health-system corticosteroid expenditure",
    },
    "fuel_consumption": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/consumos-combustibles-mensual.csv",
        "filename": "fuel_consumption.csv",
        "target": None,
        "description": "Monthly fuel consumption in Spain",
    },
    "air_quality": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/air_quality_valencia_no_missing.csv",
        "filename": "air_quality_valencia_no_missing.csv",
        "target": "pm2.5",
        "description": "Hourly air-quality measurements in Valencia",
    },
    "website_visits": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/visitas_por_dia_web_cienciadedatos.csv",
        "filename": "website_visits.csv",
        "target": None,
        "description": "Daily website visits",
    },
    "bike_sharing": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/bike_sharing_dataset_clean.csv",
        "filename": "bike_sharing.csv",
        "target": None,
        "description": "Hourly Washington DC bike-sharing demand",
    },
    "australia_tourism": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/australia_tourism.csv",
        "filename": "australia_tourism.csv",
        "target": None,
        "description": "Quarterly Australian tourism demand",
    },
    "uk_daily_flights": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/uk_daily_flights.csv",
        "filename": "uk_daily_flights.csv",
        "target": None,
        "description": "Daily UK flight counts",
    },
    "vic_electricity": {
        "url": "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv",
        "filename": "vic_electricity.csv",
        "target": None,
        "description": "Half-hourly Victoria, Australia electricity demand",
    },
}

TARGET_ALIASES = {
    "h2o": ["y", "value", "x", "h2o"],
    "fuel_consumption": ["Consumo", "consumo", "value"],
    "website_visits": ["visits", "visitas", "value", "y"],
    "bike_sharing": ["users", "cnt", "count", "total_count"],
    "australia_tourism": ["Trips", "trips", "value"],
    "uk_daily_flights": ["flights", "Flights", "value"],
    "vic_electricity": ["Demand", "demand", "value"],
}


def ensure_dirs() -> None:
    for p in (DATA_DIR, MODEL_DIR, RESULT_DIR):
        p.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_id: str, force: bool = False) -> Path:
    cfg = DATASETS[dataset_id]
    ensure_dirs()
    dest = DATA_DIR / cfg["filename"]
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dataset_id}: {dest}")
        return dest
    print(f"[download] {dataset_id} -> {dest}")
    request = urllib.request.Request(
        cfg["url"],
        headers={"User-Agent": "HE-ML-workstation-trainer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"[done] {dataset_id}: {dest.stat().st_size / 1024**2:.2f} MB")
    return dest


def download_all(force: bool = False) -> None:
    for dataset_id in DATASETS:
        download_dataset(dataset_id, force=force)


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def choose_target(dataset_id: str, df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target '{requested}' not found. Columns: {list(df.columns)}")
        return requested

    preferred = DATASETS[dataset_id].get("target")
    if preferred and preferred in df.columns:
        return preferred

    for alias in TARGET_ALIASES.get(dataset_id, []):
        if alias in df.columns and pd.api.types.is_numeric_dtype(df[alias]):
            return alias

    candidates = _numeric_candidates(df)
    if not candidates:
        raise ValueError(f"No numeric target column found in {dataset_id}. Columns: {list(df.columns)}")
    return candidates[-1]


def load_real_series(
    dataset_id: str,
    target: str | None = None,
    max_points: int | None = None,
) -> tuple[pd.DataFrame, str]:
    path = DATA_DIR / DATASETS[dataset_id]["filename"]
    if not path.exists():
        download_dataset(dataset_id)

    df = pd.read_csv(path)
    target_col = choose_target(dataset_id, df, target)

    values = pd.to_numeric(df[target_col], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(limit_direction="both").ffill().bfill()
    values = values.dropna()

    if max_points and len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values.iloc[idx]

    out = pd.DataFrame({"step": np.arange(len(values)), "value": values.to_numpy(dtype=float)})
    return out, target_col


def save_model(model: LSTMSequencePredictor, dataset_id: str, metadata: dict[str, Any]) -> Path:
    ensure_dirs()
    path = MODEL_DIR / f"{dataset_id}.npz"
    np.savez(
        path,
        Wf=model.Wf, bf=model.bf,
        Wi=model.Wi, bi=model.bi,
        Wc=model.Wc, bc=model.bc,
        Wo=model.Wo, bo=model.bo,
        Wy=model.Wy, by=model.by,
        scaler_min=np.array(model.scaler_min),
        scaler_max=np.array(model.scaler_max),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_model(dataset_id: str) -> LSTMSequencePredictor:
    path = MODEL_DIR / f"{dataset_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Train it first.")
    z = np.load(path)
    hidden_dim = int(z["Wf"].shape[0])
    model = LSTMSequencePredictor(hidden_dim=hidden_dim)
    for name in ("Wf", "bf", "Wi", "bi", "Wc", "bc", "Wo", "bo", "Wy", "by"):
        setattr(model, name, z[name])
    model.scaler_min = float(z["scaler_min"])
    model.scaler_max = float(z["scaler_max"])
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def train_one(
    dataset_id: str,
    *,
    seq_len: int = 24,
    epochs: int = 5,
    lr: float = 0.01,
    max_points: int | None = 3000,
    target: str | None = None,
) -> dict[str, Any]:
    df, target_col = load_real_series(dataset_id, target=target, max_points=max_points)
    X, y = TimeSeriesDatasetManager.create_sequence_windows(df["value"].values, seq_len=seq_len)
    if len(X) < 20:
        raise ValueError(f"{dataset_id}: too few windows ({len(X)})")

    split = max(1, int(len(X) * 0.8))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = LSTMSequencePredictor(hidden_dim=16)
    t0 = time.perf_counter()
    history = model.fit(X_train, y_train, epochs=epochs, lr=lr, verbose=False)
    train_seconds = time.perf_counter() - t0

    pred = model.predict(X_test)
    m = metrics(y_test, pred)

    metadata = {
        "dataset_id": dataset_id,
        "target": target_col,
        "points": len(df),
        "seq_len": seq_len,
        "epochs": epochs,
        "lr": lr,
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "train_seconds": train_seconds,
        "metrics": m,
        "final_train_loss": float(history[-1]),
    }
    model_path = save_model(model, dataset_id, metadata)
    metadata["model_path"] = str(model_path)
    ensure_dirs()
    (RESULT_DIR / f"{dataset_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return metadata


def infer_one(
    dataset_id: str,
    *,
    scheme: str = "ckks",
    seq_len: int = 24,
    max_points: int | None = 3000,
    target: str | None = None,
) -> dict[str, Any]:
    model = load_model(dataset_id)
    df, target_col = load_real_series(dataset_id, target=target, max_points=max_points)
    values = df["value"].to_numpy(dtype=float)
    if len(values) < seq_len + 1:
        raise ValueError("Not enough points for inference.")
    window = values[-seq_len:]
    X = window.reshape(1, seq_len, 1)

    plain = float(model.predict(X)[0, 0])

    he_ctx = MultiSchemeHEContext(scheme_id=scheme)
    t0 = time.perf_counter()
    ciphertext = he_ctx.encrypt(window)
    recovered = np.asarray(he_ctx.decrypt(ciphertext), dtype=float)
    he_roundtrip_ms = (time.perf_counter() - t0) * 1000.0

    he_input_pred = float(model.predict(recovered.reshape(1, seq_len, 1))[0, 0])
    result = {
        "dataset_id": dataset_id,
        "target": target_col,
        "scheme": scheme,
        "plain_prediction": plain,
        "he_roundtrip_prediction": he_input_pred,
        "input_roundtrip_mae": float(np.mean(np.abs(window - recovered))),
        "prediction_delta": abs(plain - he_input_pred),
        "he_roundtrip_ms": he_roundtrip_ms,
        "note": "This validates the repository's HE encrypt/decrypt path around the model input; it is not a fully homomorphic LSTM evaluation.",
    }
    print(json.dumps(result, indent=2))
    return result


def benchmark(
    dataset_ids: Iterable[str],
    *,
    epochs: int,
    max_points: int,
    seq_len: int,
    lr: float,
) -> list[dict[str, Any]]:
    results = []
    for dataset_id in dataset_ids:
        try:
            results.append(
                train_one(
                    dataset_id,
                    epochs=epochs,
                    max_points=max_points,
                    seq_len=seq_len,
                    lr=lr,
                )
            )
        except Exception as exc:
            results.append({"dataset_id": dataset_id, "status": "error", "error": repr(exc)})
            print(f"[error] {dataset_id}: {exc}")
    (RESULT_DIR / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HE-ML workstation dataset/training/inference tool")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download")
    d.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    d.add_argument("--force", action="store_true")

    t = sub.add_parser("train")
    t.add_argument("--dataset", choices=list(DATASETS), required=True)
    t.add_argument("--target")
    t.add_argument("--seq-len", type=int, default=24)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--max-points", type=int, default=3000)

    i = sub.add_parser("infer")
    i.add_argument("--dataset", choices=list(DATASETS), required=True)
    i.add_argument("--scheme", choices=["ckks", "bfv", "tfhe", "paillier"], default="ckks")
    i.add_argument("--target")
    i.add_argument("--seq-len", type=int, default=24)
    i.add_argument("--max-points", type=int, default=3000)

    b = sub.add_parser("benchmark")
    b.add_argument("--datasets", choices=list(DATASETS) + ["all"], default="all")
    b.add_argument("--seq-len", type=int, default=24)
    b.add_argument("--epochs", type=int, default=3)
    b.add_argument("--lr", type=float, default=0.01)
    b.add_argument("--max-points", type=int, default=1500)

    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()

    if args.command == "download":
        if args.dataset == "all":
            download_all(force=args.force)
        else:
            download_dataset(args.dataset, force=args.force)

    elif args.command == "train":
        train_one(
            args.dataset,
            target=args.target,
            seq_len=args.seq_len,
            epochs=args.epochs,
            lr=args.lr,
            max_points=args.max_points,
        )

    elif args.command == "infer":
        infer_one(
            args.dataset,
            scheme=args.scheme,
            target=args.target,
            seq_len=args.seq_len,
            max_points=args.max_points,
        )

    elif args.command == "benchmark":
        ids = list(DATASETS) if args.datasets == "all" else [args.datasets]
        benchmark(
            ids,
            epochs=args.epochs,
            max_points=args.max_points,
            seq_len=args.seq_len,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
