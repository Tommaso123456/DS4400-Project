"""
Data loading, cleaning, scaling, and resampling for credit card fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train/test and scale Amount and Time.
    V1-V28 are already PCA-transformed and do not need scaling.
    Returns X_train, X_test, y_train, y_test (no SMOTE applied yet).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Oversample the minority (fraud) class on training data only.
    Never apply SMOTE to test data.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
