"""
T2 Inference Module — NEXUS-RT v1 (Superstore Multi-Task Prediction)
====================================================================
Loads the trained NexusRT model and provides a clean API for
single-transaction inference with the exact same preprocessing
pipeline used during training.

Model architecture, feature engineering, and data pipeline are
faithfully reproduced from:
    T2/model.py

IMPORTANT: The StandardScaler and LabelEncoders are NOT saved in
the checkpoint. They are refit at load time from the original
training data using the exact same pipeline (StratifiedShuffleSplit
with seed=42, test_size=0.20, scaler fit on X_train only).
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (exact copy from T2/model.py lines 300-415)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> GELU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.20):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, out_dim)

        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.fc(self.drop(self.act(self.bn(x))))
        return out + residual


class NexusRT(nn.Module):
    """
    NEXUS-RT v1 -- Neural EXpert for Unified Sales-analytics (Residual Tabular)

    A lightweight dual-head residual MLP designed for multi-task tabular
    classification on retail transaction data.

    Architecture:
        Input (46-d) -> Linear Projection (256-d) -> BN -> GELU -> Dropout(0.10)
            -> ResidualBlock(256 -> 192) -> ResidualBlock(192 -> 128)
            -> Head A: [BN -> GELU -> FC(64) -> BN -> GELU -> Drop -> FC(32)
                        -> BN -> GELU -> Drop -> FC(2)]    # Profitability
            -> Head B: [BN -> GELU -> Drop -> FC(17)]       # Sub-Category

    Total parameters: ~174K (lightweight, CPU-trainable in <8s)
    """

    def __init__(
        self,
        num_features: int,
        num_classes_multi: int,
        hidden_dims: list = None,
        dropout: float = 0.20,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 192, 128]

        # -- Input projection --------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # -- Shared backbone ---------------------------------------------------
        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout)
            )
        self.backbone = nn.Sequential(*blocks)

        final_dim = hidden_dims[-1]

        # -- Head A: Binary (profitable?) -- deeper for harder task ------------
        self.head_binary = nn.Sequential(
            nn.BatchNorm1d(final_dim),
            nn.GELU(),
            nn.Linear(final_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 2),
        )

        # -- Head B: Multi-class (sub-category) --------------------------------
        self.head_multi = nn.Sequential(
            nn.BatchNorm1d(final_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(final_dim, num_classes_multi),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.backbone(h)
        logits_bin = self.head_binary(h)
        logits_multi = self.head_multi(h)
        return logits_bin, logits_multi


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (exact copy from T2/model.py lines 83-168)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expert-level feature engineering pipeline.
    Extracts temporal, interaction, ratio, and aggregate features
    from the raw Superstore dataset.
    """
    data = df.copy()

    # -- Parse dates -----------------------------------------------------------
    data["Order Date"] = pd.to_datetime(data["Order Date"], format="mixed", dayfirst=False)
    data["Ship Date"] = pd.to_datetime(data["Ship Date"], format="mixed", dayfirst=False)

    # -- Temporal features -----------------------------------------------------
    data["Order_Year"] = data["Order Date"].dt.year
    data["Order_Month"] = data["Order Date"].dt.month
    data["Order_DayOfWeek"] = data["Order Date"].dt.dayofweek
    data["Order_Quarter"] = data["Order Date"].dt.quarter
    data["Order_DayOfYear"] = data["Order Date"].dt.dayofyear
    data["Order_WeekOfYear"] = data["Order Date"].dt.isocalendar().week.astype(int)
    data["Is_Weekend"] = (data["Order_DayOfWeek"] >= 5).astype(int)
    data["Is_MonthEnd"] = data["Order Date"].dt.is_month_end.astype(int)
    data["Is_MonthStart"] = data["Order Date"].dt.is_month_start.astype(int)
    data["Is_QuarterEnd"] = data["Order Date"].dt.is_quarter_end.astype(int)

    # -- Shipping features -----------------------------------------------------
    data["Shipping_Days"] = (data["Ship Date"] - data["Order Date"]).dt.days
    data["Shipping_Days"] = data["Shipping_Days"].clip(lower=0)

    # -- Interaction / ratio features ------------------------------------------
    data["Sales_Per_Quantity"] = data["Sales"] / data["Quantity"].clip(lower=1)
    data["Revenue_After_Discount"] = data["Sales"] * (1 - data["Discount"])
    data["Discount_Amount"] = data["Sales"] * data["Discount"]
    data["Has_Discount"] = (data["Discount"] > 0).astype(int)
    data["High_Discount"] = (data["Discount"] >= 0.3).astype(int)
    data["Very_High_Discount"] = (data["Discount"] >= 0.5).astype(int)

    # Profit-signal features (no leakage -- derived from Sales/Discount/Quantity)
    data["Log_Sales"] = np.log1p(data["Sales"])
    data["Log_SalesPerQty"] = np.log1p(data["Sales_Per_Quantity"])
    data["Discount_Sq"] = data["Discount"] ** 2
    data["Sales_x_Discount"] = data["Sales"] * data["Discount"]
    data["Qty_x_Discount"] = data["Quantity"] * data["Discount"]
    data["Net_Revenue_Ratio"] = 1 - data["Discount"]
    data["Margin_Proxy"] = data["Sales_Per_Quantity"] * (1 - data["Discount"])

    # Quantity buckets
    data["Quantity_Bucket"] = pd.cut(
        data["Quantity"],
        bins=[0, 2, 5, 9, 15],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # -- Cyclical encoding for month/day ---------------------------------------
    data["Month_Sin"] = np.sin(2 * np.pi * data["Order_Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Order_Month"] / 12)
    data["DayOfWeek_Sin"] = np.sin(2 * np.pi * data["Order_DayOfWeek"] / 7)
    data["DayOfWeek_Cos"] = np.cos(2 * np.pi * data["Order_DayOfWeek"] / 7)

    # -- Aggregate / statistical features (target-free) ------------------------
    subcat_sales_mean = data.groupby("Sub-Category")["Sales"].transform("mean")
    data["SubCat_Sales_Mean"] = subcat_sales_mean

    subcat_sales_std = data.groupby("Sub-Category")["Sales"].transform("std").fillna(0)
    data["SubCat_Sales_Std"] = subcat_sales_std

    state_sales_mean = data.groupby("State")["Sales"].transform("mean")
    data["State_Sales_Mean"] = state_sales_mean

    cat_discount_mean = data.groupby("Category")["Discount"].transform("mean")
    data["Cat_Discount_Mean"] = cat_discount_mean

    subcat_discount_mean = data.groupby("Sub-Category")["Discount"].transform("mean")
    data["SubCat_Discount_Mean"] = subcat_discount_mean

    cust_freq = data.groupby("Customer ID")["Row ID"].transform("count")
    data["Customer_Order_Freq"] = cust_freq

    # Discount deviation from sub-category average (key profit signal)
    data["Discount_Dev_SubCat"] = data["Discount"] - data["SubCat_Discount_Mean"]

    # Sales deviation from sub-category mean (z-score proxy)
    data["Sales_ZScore_SubCat"] = (
        (data["Sales"] - subcat_sales_mean) / subcat_sales_std.clip(lower=1e-6)
    )

    return data


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Replicate the full prepare_data pipeline for scaler/encoders
# ─────────────────────────────────────────────────────────────────────────────

def _replicate_prepare_data(data_path: str):
    """
    Replay the exact training pipeline from T2/model.py prepare_data()
    to recover the StandardScaler (fit on X_train) and LabelEncoders.

    Returns
    -------
    scaler : StandardScaler  (fit on X_train only)
    label_encoders : dict     {col_name: fitted LabelEncoder}
    feature_cols : list[str]  ordered feature column names
    subcat_names : list[str]  ordered sub-category class names
    states : list[str]        sorted unique state names
    full_data : pd.DataFrame  the engineered dataframe (for aggregate lookups)
    """
    raw = pd.read_csv(data_path, encoding="latin1")
    data = engineer_features(raw)

    # Binary target
    data["Is_Profitable"] = (data["Profit"] > 0).astype(int)

    # Multi-class target
    le_subcat = LabelEncoder()
    data["SubCat_Label"] = le_subcat.fit_transform(data["Sub-Category"])
    subcat_names = le_subcat.classes_.tolist()

    # Categorical encoding
    cat_columns = ["Ship Mode", "Segment", "Category", "Sub-Category", "Region", "State"]
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        data[f"{col}_enc"] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Feature selection (exclude IDs, names, dates, raw targets)
    exclude_cols = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Customer ID", "Customer Name", "Country", "City",
        "Postal Code", "Product ID", "Product Name",
        "Profit", "Is_Profitable", "SubCat_Label",
        "Ship Mode", "Segment", "Category", "Sub-Category", "Region", "State",
    ]
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    X = data[feature_cols].values.astype(np.float32)
    y_multi = data["SubCat_Label"].values.astype(np.int64)

    # Handle NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified split — exact same seed and test_size as training
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, _test_idx = next(sss.split(X, y_multi))

    X_train = X[train_idx]

    # Fit scaler on training split only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Sorted unique states
    states = sorted(raw["State"].dropna().unique().tolist())

    return scaler, label_encoders, feature_cols, subcat_names, states, data


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_t2_model(base_dir: str = "."):
    """
    Load the trained NexusRT checkpoint and reconstruct the full
    inference pipeline (scaler, label encoders, feature columns).

    Parameters
    ----------
    base_dir : str
        Root directory of the project (parent of T2/).

    Returns
    -------
    model : NexusRT
        Model in eval mode on CPU.
    scaler : StandardScaler
        Fitted on training split.
    label_encoders : dict
        {column_name: fitted LabelEncoder} for categorical columns.
    subcat_names : list[str]
        Ordered sub-category class names.
    feature_cols : list[str]
        Ordered feature column names expected by the model.
    """
    # Resolve project root from this file's location (utils/ -> project root)
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- Locate model checkpoint ---
    model_candidates = [
        os.path.join(base_dir, "T2", "nexus_rt_v1.pth"),
        os.path.join(_project_root, "T2", "nexus_rt_v1.pth"),
    ]
    model_path = None
    for path in model_candidates:
        if os.path.isfile(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Cannot find nexus_rt_v1.pth.\n"
            f"Searched:\n  " + "\n  ".join(model_candidates) + "\n"
            f"Project root detected: {_project_root}\n"
            f"Contents of project root: {os.listdir(_project_root)}"
        )

    # --- Locate dataset CSV ---
    data_candidates = [
        os.path.join(base_dir, "T2", "Dataset", "Sample - Superstore.csv"),
        os.path.join(_project_root, "T2", "Dataset", "Sample - Superstore.csv"),
    ]
    data_path = None
    for path in data_candidates:
        if os.path.isfile(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Cannot find Sample - Superstore.csv.\n"
            f"Searched:\n  " + "\n  ".join(data_candidates) + "\n"
            f"Project root detected: {_project_root}"
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Replicate the data pipeline to recover scaler & encoders
    scaler, label_encoders, feature_cols, subcat_names, states, _full_data = (
        _replicate_prepare_data(data_path)
    )

    # Reconstruct model
    info = checkpoint["info"]
    config = checkpoint["config"]
    model = NexusRT(
        num_features=info["num_features"],
        num_classes_multi=info["num_subcat_classes"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Stash states and full_data reference on the function for helpers
    load_t2_model._states = states
    load_t2_model._full_data = _full_data
    load_t2_model._project_root = _project_root

    return model, scaler, label_encoders, subcat_names, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict_transaction(
    model,
    scaler,
    label_encoders,
    feature_cols,
    subcat_names,
    raw_input: dict,
) -> dict:
    """
    Predict profitability and sub-category for a single transaction.

    Parameters
    ----------
    model : NexusRT
        Loaded model in eval mode.
    scaler : StandardScaler
        Fitted scaler from training pipeline.
    label_encoders : dict
        Fitted LabelEncoders for categorical columns.
    feature_cols : list[str]
        Ordered feature column names.
    subcat_names : list[str]
        Ordered sub-category class names.
    raw_input : dict
        Transaction data with keys:
            Ship Mode, Segment, Category, Sub-Category, Region, State,
            Sales, Quantity, Discount, Order Date, Ship Date,
            Row ID (optional), Order ID (optional), Customer ID (optional),
            Customer Name (optional), Product ID (optional),
            Product Name (optional), Country (optional), City (optional),
            Postal Code (optional), Profit (optional, set 0 for inference).

    Returns
    -------
    dict with keys:
        is_profitable : bool
        profit_confidence : float  (probability of predicted class)
        sub_category : str
        subcat_confidence : float  (probability of predicted sub-category)
        profit_probs : list[float] (probabilities for [Not Profitable, Profitable])
        subcat_probs : dict[str, float]  (sub-category name → probability)
    """
    # -- Build a single-row DataFrame matching the raw training format ---------
    # Fill in defaults for optional columns
    defaults = {
        "Row ID": raw_input.get("Row ID", 1),
        "Order ID": raw_input.get("Order ID", "INF-0001"),
        "Customer ID": raw_input.get("Customer ID", "INF-00001"),
        "Customer Name": raw_input.get("Customer Name", "Inference User"),
        "Country": raw_input.get("Country", "United States"),
        "City": raw_input.get("City", "New York"),
        "Postal Code": raw_input.get("Postal Code", 10001),
        "Product ID": raw_input.get("Product ID", "INF-PROD-001"),
        "Product Name": raw_input.get("Product Name", "Inference Product"),
        "Profit": raw_input.get("Profit", 0.0),
    }

    row = {**defaults, **raw_input}

    # Ensure numeric types
    row["Sales"] = float(row["Sales"])
    row["Quantity"] = int(row["Quantity"])
    row["Discount"] = float(row["Discount"])
    row["Profit"] = float(row.get("Profit", 0.0))
    row["Row ID"] = int(row.get("Row ID", 1))

    single_df = pd.DataFrame([row])

    # -- Need the full training data for aggregate features --------------------
    # Load the training dataset to compute group-level aggregates
    # (SubCat_Sales_Mean, State_Sales_Mean, etc.) correctly
    if hasattr(load_t2_model, "_full_data") and load_t2_model._full_data is not None:
        # Append the inference row to full training data so groupby
        # transforms produce correct aggregate values for this row
        full_data = load_t2_model._full_data
        # Ensure the single row has the same base columns
        base_cols = [
            "Row ID", "Order ID", "Order Date", "Ship Date",
            "Customer ID", "Customer Name", "Segment", "Country", "City",
            "State", "Postal Code", "Region", "Product ID", "Product Name",
            "Category", "Sub-Category", "Sales", "Quantity", "Discount",
            "Profit", "Ship Mode",
        ]
        # Build combined dataframe for feature engineering
        combined_rows = []
        for _, r in full_data.head(0).iterrows():
            pass  # just need the structure

        # Re-read raw data and append inference row
        _project_root = getattr(load_t2_model, "_project_root",
                                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_candidates = [
            os.path.join(_project_root, "T2", "Dataset", "Sample - Superstore.csv"),
        ]
        raw_df = None
        for dp in data_candidates:
            if os.path.exists(dp):
                raw_df = pd.read_csv(dp, encoding="latin1")
                break
        if raw_df is not None:
            combined_df = pd.concat([raw_df, single_df], ignore_index=True)
        else:
            combined_df = single_df
    else:
        combined_df = single_df

    # -- Apply feature engineering on the combined dataset ---------------------
    engineered = engineer_features(combined_df)

    # Binary target (dummy for inference)
    engineered["Is_Profitable"] = (engineered["Profit"] > 0).astype(int)

    # Sub-category label
    le_subcat_temp = LabelEncoder()
    le_subcat_temp.fit(subcat_names)
    engineered["SubCat_Label"] = le_subcat_temp.transform(engineered["Sub-Category"])

    # Categorical encoding — use the fitted label_encoders
    cat_columns = ["Ship Mode", "Segment", "Category", "Sub-Category", "Region", "State"]
    for col in cat_columns:
        le = label_encoders[col]
        col_values = engineered[col].astype(str)
        # Handle unseen labels gracefully
        encoded = []
        for val in col_values:
            if val in le.classes_:
                encoded.append(le.transform([val])[0])
            else:
                encoded.append(0)  # fallback to 0 for unseen labels
        engineered[f"{col}_enc"] = encoded

    # -- Select the last row (our inference sample) ----------------------------
    inference_row = engineered.iloc[[-1]]

    # -- Select feature columns and scale --------------------------------------
    exclude_cols = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Customer ID", "Customer Name", "Country", "City",
        "Postal Code", "Product ID", "Product Name",
        "Profit", "Is_Profitable", "SubCat_Label",
        "Ship Mode", "Segment", "Category", "Sub-Category", "Region", "State",
    ]
    available_feature_cols = [c for c in engineered.columns if c not in exclude_cols]

    # Use the exact feature_cols ordering from training
    x_values = []
    for col in feature_cols:
        if col in inference_row.columns:
            x_values.append(float(inference_row[col].values[0]))
        else:
            x_values.append(0.0)

    x = np.array([x_values], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    x_scaled = scaler.transform(x)

    # -- Forward pass ----------------------------------------------------------
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits_bin, logits_multi = model(x_tensor)

    # -- Process outputs -------------------------------------------------------
    probs_bin = torch.softmax(logits_bin, dim=1).squeeze(0).numpy()
    probs_multi = torch.softmax(logits_multi, dim=1).squeeze(0).numpy()

    pred_bin = int(logits_bin.argmax(dim=1).item())
    pred_multi = int(logits_multi.argmax(dim=1).item())

    is_profitable = bool(pred_bin == 1)
    profit_confidence = float(probs_bin[pred_bin])

    sub_category = subcat_names[pred_multi] if pred_multi < len(subcat_names) else "Unknown"
    subcat_confidence = float(probs_multi[pred_multi])

    subcat_probs = {
        name: float(probs_multi[i])
        for i, name in enumerate(subcat_names)
    }

    return {
        "is_profitable": is_profitable,
        "profit_confidence": profit_confidence,
        "sub_category": sub_category,
        "subcat_confidence": subcat_confidence,
        "profit_probs": probs_bin.tolist(),
        "subcat_probs": subcat_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (for UI dropdowns)
# ─────────────────────────────────────────────────────────────────────────────

def get_ship_modes() -> list:
    """Return valid Ship Mode values."""
    return ["First Class", "Same Day", "Second Class", "Standard Class"]


def get_segments() -> list:
    """Return valid Segment values."""
    return ["Consumer", "Corporate", "Home Office"]


def get_categories() -> list:
    """Return valid Category values."""
    return ["Furniture", "Office Supplies", "Technology"]


def get_subcategories() -> list:
    """Return the 17 sub-category names (alphabetical, matching LabelEncoder order)."""
    return [
        "Accessories", "Appliances", "Art", "Binders", "Bookcases",
        "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings",
        "Labels", "Machines", "Paper", "Phones", "Storage",
        "Supplies", "Tables",
    ]


def get_regions() -> list:
    """Return valid Region values."""
    return ["Central", "East", "South", "West"]


def get_states() -> list:
    """Return US state names from the dataset."""
    if hasattr(load_t2_model, "_states") and load_t2_model._states:
        return load_t2_model._states
    # Fallback: comprehensive list of US states present in Superstore data
    return [
        "Alabama", "Arizona", "Arkansas", "California", "Colorado",
        "Connecticut", "Delaware", "District of Columbia", "Florida",
        "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
        "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
        "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
        "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
        "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    ]
