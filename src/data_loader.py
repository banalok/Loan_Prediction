# src/data_loader.py

import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at path: {path}")

    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

