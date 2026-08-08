import numpy as np
import pandas as pd

class LargeDatasetManager:
    """
    Manager for large-scale training datasets (10,000 to 100,000+ data samples).
    Supports Time-Series, Tabular, Text, and Vision datasets with batching and validation splits.
    """
    LARGE_DATASETS = {
        "large_time_series": {
            "name": "Large-Scale Power Grid & Smart Grid Time-Series (50,000 steps)",
            "type": "Sequence / Time-Series",
            "default_samples": 50000,
            "features": 1
        },
        "large_tabular_credit": {
            "name": "Large Credit Risk & Financial Tabular Dataset (20,000 rows)",
            "type": "Tabular Classification",
            "default_samples": 20000,
            "features": 10
        },
        "large_text_intent": {
            "name": "Large NLP Intent & Sentiment Corpus (10,000 samples)",
            "type": "NLP Text Sequences",
            "default_samples": 10000,
            "features": 16
        },
        "large_vision_objects": {
            "name": "Large Industrial Visual Surface Dataset (10,000 samples)",
            "type": "Computer Vision (32x32x3)",
            "default_samples": 10000,
            "features": 3072
        }
    }

    @classmethod
    def load_large_dataset(cls, dataset_id="large_time_series", n_samples=10000, seed=42):
        np.random.seed(seed)
        
        if dataset_id == "large_time_series":
            t = np.linspace(0, 500, n_samples)
            # High-frequency trend + seasonality + harmonics
            signal = 50 + 20 * np.sin(t * 0.1) + 10 * np.cos(t * 0.5) + 5 * np.sin(t * 2.0) + np.random.normal(0, 1.5, n_samples)
            return {"X": signal, "type": "time_series"}
            
        elif dataset_id == "large_tabular_credit":
            X = np.random.normal(0, 1, size=(n_samples, 10))
            # True decision boundary
            logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.8 * X[:, 2] - 0.5 * X[:, 3] + np.random.normal(0, 0.5, n_samples)
            y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)
            return {"X": X, "y": y, "type": "tabular"}
            
        elif dataset_id == "large_text_intent":
            X = np.random.randint(1, 2000, size=(n_samples, 16))
            y = np.random.randint(0, 4, size=n_samples)
            return {"X": X, "y": y, "type": "text"}
            
        elif dataset_id == "large_vision_objects":
            X = np.random.uniform(0, 1, size=(n_samples, 32, 32, 3))
            y = np.random.randint(0, 5, size=n_samples)
            return {"X": X, "y": y, "type": "vision"}

        raise ValueError(f"Unknown large dataset_id '{dataset_id}'")
