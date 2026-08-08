import numpy as np
import pandas as pd

class TimeSeriesDatasetManager:
    """
    Manager for 10 popular benchmark time-series and sequential datasets.
    Supports generation, preprocessing, sequence windowing, and sampling for LSTM training.
    """
    DATASETS_INFO = {
        "sine_wave": {
            "name": "Sine Wave Oscillation",
            "category": "Synthetic Benchmark",
            "description": "Fundamental sine wave signal with optional harmonic noise for sequence learning."
        },
        "electricity": {
            "name": "Electricity Power Load",
            "category": "Energy & Power",
            "description": "Hourly grid power consumption (kWh) showing daily and weekly load cycles."
        },
        "air_quality": {
            "name": "Air Quality PM2.5 Index",
            "category": "Environmental Science",
            "description": "Hourly PM2.5 atmospheric pollution index with environmental variations."
        },
        "weather_temp": {
            "name": "Weather Temperature",
            "category": "Meteorology",
            "description": "Daily ambient temperature (°C) featuring annual seasonality and trend."
        },
        "airline_passengers": {
            "name": "Airline Passenger Traffic",
            "category": "Aviation & Tourism",
            "description": "Monthly international airline passenger volume (classic Box-Jenkins benchmark)."
        },
        "stock_market": {
            "name": "Stock Market Index",
            "category": "Financial Markets",
            "description": "Daily financial asset price index with stochastic drift and volatility jumps."
        },
        "ecg_signal": {
            "name": "ECG Heartbeat Physiological Signal",
            "category": "Healthcare & Biomedical",
            "description": "Electrocardiogram cardiac rhythm pulse waveform time-series."
        },
        "cpu_utilization": {
            "name": "Cloud Server CPU Usage",
            "category": "IT Infrastructure",
            "description": "Server CPU utilization percentage (%) showing peak workload spikes."
        },
        "traffic_flow": {
            "name": "Urban Traffic Volume",
            "category": "Smart City & Transport",
            "description": "Hourly urban highway vehicle count capturing morning/evening rush hours."
        },
        "commodity_price": {
            "name": "Crude Oil Commodity Price",
            "category": "Global Trade",
            "description": "Crude oil market benchmark price ($/barrel) with macro commodity cycles."
        }
    }

    @classmethod
    def list_datasets(cls):
        """Returns a list of all 10 supported benchmark datasets."""
        return [
            {"id": key, **info} for key, info in cls.DATASETS_INFO.items()
        ]

    @classmethod
    def load_dataset(cls, dataset_id: str, n_samples: int = 500, noise_std: float = 0.05, seed: int = 42):
        """
        Loads time-series data for any of the 10 supported datasets.
        """
        if dataset_id not in cls.DATASETS_INFO:
            raise ValueError(f"Unknown dataset_id '{dataset_id}'. Choose from: {list(cls.DATASETS_INFO.keys())}")
            
        np.random.seed(seed)
        t = np.linspace(0, 50, n_samples)
        
        if dataset_id == "sine_wave":
            signal = np.sin(t) + noise_std * np.random.normal(size=n_samples)
        elif dataset_id == "electricity":
            signal = 100 + 30 * np.sin(t * 0.8) + 15 * np.cos(t * 2.5) + noise_std * 10 * np.random.normal(size=n_samples)
        elif dataset_id == "air_quality":
            signal = 45 + 25 * np.sin(t * 0.5) + 10 * np.sin(t * 1.8) + np.maximum(0, noise_std * 20 * np.random.normal(size=n_samples))
        elif dataset_id == "weather_temp":
            signal = 18 + 12 * np.sin(t * 0.2) + 4 * np.cos(t * 1.2) + noise_std * 3 * np.random.normal(size=n_samples)
        elif dataset_id == "airline_passengers":
            trend = np.linspace(100, 450, n_samples)
            seasonality = 40 * np.sin(t * 0.4)
            signal = trend + seasonality + noise_std * 15 * np.random.normal(size=n_samples)
        elif dataset_id == "stock_market":
            returns = np.random.normal(0.0005, 0.02, size=n_samples)
            signal = 100 * np.cumprod(1 + returns)
        elif dataset_id == "ecg_signal":
            signal = (np.sin(t * 3.0) ** 5) + 0.3 * np.cos(t * 12.0) + noise_std * 0.2 * np.random.normal(size=n_samples)
        elif dataset_id == "cpu_utilization":
            base = 35 + 20 * np.sin(t * 0.6)
            spikes = (np.random.random(n_samples) > 0.93) * 35.0
            signal = np.clip(base + spikes + noise_std * 5 * np.random.normal(size=n_samples), 5, 99)
        elif dataset_id == "traffic_flow":
            signal = 200 + 120 * np.sin(t * 0.7) + 60 * np.sin(t * 2.1) + noise_std * 25 * np.random.normal(size=n_samples)
        elif dataset_id == "commodity_price":
            trend = 60 + 0.5 * t
            cycles = 15 * np.cos(t * 0.3)
            signal = trend + cycles + noise_std * 5 * np.random.normal(size=n_samples)

        df = pd.DataFrame({
            "step": np.arange(n_samples),
            "value": signal
        })
        return df

    @classmethod
    def create_sequence_windows(cls, data_array, seq_len: int = 10):
        """
        Converts 1D time-series data array into sliding sequence windows (X, y) for LSTM training.
        X shape: (n_windows, seq_len, 1)
        y shape: (n_windows, 1)
        """
        data = np.array(data_array, dtype=float).flatten()
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len])
            
        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y).reshape(-1, 1)
        return X, y
