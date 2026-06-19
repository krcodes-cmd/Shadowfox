"""
T1 Inference Module — Boston Housing Price Prediction
=====================================================
Loads the trained BostonHousingNet model and provides a clean API
for single-sample inference with the exact same preprocessing pipeline
used during training.

Model architecture and preprocessing are faithfully reproduced from:
    T1/boston_housing_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (exact copy from T1/boston_housing_model.py lines 144-192)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN → GELU → Linear → BN → GELU → Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BostonHousingNet(nn.Module):
    """
    Lightweight MLP with residual connections.
    Architecture:  Input → Expand → ResBlock → ResBlock → Compress → Output
    Total params:  ~15-20k (very lightweight)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of raw features as they appear in the training CSV
RAW_FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

# Reasonable percentile-based clipping bounds (1st / 99th percentile
# estimated from the Boston Housing dataset).  Used at inference time
# since we don't ship the full training data.
_CLIP_BOUNDS = {
    "CRIM":    (0.00632,  73.5341),
    "ZN":      (0.0,      95.0),
    "INDUS":   (0.74,     27.74),
    "CHAS":    (0.0,      1.0),
    "NOX":     (0.385,    0.871),
    "RM":      (3.863,    8.398),
    "AGE":     (2.9,      100.0),
    "DIS":     (1.1296,   10.7103),
    "RAD":     (1.0,      24.0),
    "TAX":     (188.0,    711.0),
    "PTRATIO": (12.6,     21.2),
    "B":       (3.5,      396.9),
    "LSTAT":   (1.73,     34.37),
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_t1_model(base_dir: str = "."):
    """
    Load the trained BostonHousingNet checkpoint.

    Parameters
    ----------
    base_dir : str
        Root directory of the project (parent of T1/).

    Returns
    -------
    model : BostonHousingNet
        Model in eval mode on CPU.
    checkpoint : dict
        Full checkpoint dict containing scaler_mean, scaler_scale,
        features, features_engineered, input_dim, hidden_dim, etc.
    """
    model_path = os.path.join(base_dir, "T1", "saved_model", "boston_housing_model.pth")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = BostonHousingNet(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict_price(model, checkpoint, feature_values: dict) -> float:
    """
    Predict the median housing price for a single sample.

    Preprocessing pipeline (mirrors training exactly):
      1. Assemble raw features in checkpoint order
      2. Clip outliers at 1st/99th percentile bounds
      3. Append engineered features: RM^2, LSTAT^2, RM*LSTAT, DIS*NOX
      4. StandardScaler transform (mean/scale from training)
      5. Forward pass → expm1 to reverse log1p target transform

    Parameters
    ----------
    model : BostonHousingNet
        Loaded model in eval mode.
    checkpoint : dict
        Checkpoint containing scaler_mean, scaler_scale, features.
    feature_values : dict
        Dict with keys CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS,
        RAD, TAX, PTRATIO, B, LSTAT → float values.

    Returns
    -------
    float
        Predicted median home value in $1000s.
    """
    features = checkpoint["features"]  # ordered raw feature names

    # 1. Build raw feature vector in the correct order
    raw = np.array([float(feature_values[f]) for f in features], dtype=np.float32)

    # 2. Clip outliers at 1st/99th percentile
    for i, feat_name in enumerate(features):
        if feat_name in _CLIP_BOUNDS:
            lo, hi = _CLIP_BOUNDS[feat_name]
            raw[i] = np.clip(raw[i], lo, hi)

    # 3. Feature engineering (must match training order)
    rm_idx = features.index("RM")
    lstat_idx = features.index("LSTAT")
    dis_idx = features.index("DIS")
    nox_idx = features.index("NOX")

    rm_sq = raw[rm_idx] ** 2
    lstat_sq = raw[lstat_idx] ** 2
    rm_lstat = raw[rm_idx] * raw[lstat_idx]
    dis_nox = raw[dis_idx] * raw[nox_idx]

    x = np.append(raw, [rm_sq, lstat_sq, rm_lstat, dis_nox]).astype(np.float32)

    # 4. StandardScaler transform
    scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
    x = (x - scaler_mean) / scaler_scale

    # 5. Forward pass
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)

    with torch.no_grad():
        pred_log = model(x_tensor).item()

    # 6. Reverse log1p transform
    predicted_price = float(np.expm1(pred_log))

    return predicted_price


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE INFO (for UI)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_info() -> dict:
    """
    Return metadata for each input feature — descriptions, defaults, and
    min/max bounds suitable for building a Streamlit UI.

    Returns
    -------
    dict
        Keyed by feature name. Each value is a dict with keys:
        description, min, max, default, step.
    """
    return {
        "CRIM": {
            "description": "Per capita crime rate by town",
            "min": 0.0,
            "max": 100.0,
            "default": 0.1,
            "step": 0.01,
        },
        "ZN": {
            "description": "Proportion of residential land zoned for lots over 25,000 sq. ft.",
            "min": 0.0,
            "max": 100.0,
            "default": 11.0,
            "step": 0.5,
        },
        "INDUS": {
            "description": "Proportion of non-retail business acres per town",
            "min": 0.0,
            "max": 30.0,
            "default": 11.0,
            "step": 0.1,
        },
        "CHAS": {
            "description": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
            "min": 0,
            "max": 1,
            "default": 0,
            "step": 1,
        },
        "NOX": {
            "description": "Nitric oxide concentration (parts per 10 million)",
            "min": 0.3,
            "max": 0.9,
            "default": 0.55,
            "step": 0.01,
        },
        "RM": {
            "description": "Average number of rooms per dwelling",
            "min": 3.0,
            "max": 9.0,
            "default": 6.3,
            "step": 0.1,
        },
        "AGE": {
            "description": "Proportion of owner-occupied units built prior to 1940",
            "min": 0.0,
            "max": 100.0,
            "default": 68.0,
            "step": 0.5,
        },
        "DIS": {
            "description": "Weighted distances to five Boston employment centres",
            "min": 1.0,
            "max": 13.0,
            "default": 3.8,
            "step": 0.1,
        },
        "RAD": {
            "description": "Index of accessibility to radial highways",
            "min": 1,
            "max": 24,
            "default": 9,
            "step": 1,
        },
        "TAX": {
            "description": "Full-value property-tax rate per $10,000",
            "min": 180,
            "max": 720,
            "default": 408,
            "step": 1,
        },
        "PTRATIO": {
            "description": "Pupil-teacher ratio by town",
            "min": 12.0,
            "max": 22.0,
            "default": 18.5,
            "step": 0.1,
        },
        "B": {
            "description": "1000(Bk - 0.63)^2 where Bk is the proportion of Black residents",
            "min": 0.0,
            "max": 400.0,
            "default": 356.0,
            "step": 1.0,
        },
        "LSTAT": {
            "description": "Percentage of lower status population",
            "min": 1.0,
            "max": 40.0,
            "default": 12.6,
            "step": 0.1,
        },
    }
