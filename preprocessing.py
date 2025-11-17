"""
preprocessing.py

Utilities for cleaning and preprocessing the Arvato demographics datasets.

This module is designed so that the SAME functions can be applied to:
- General population data (AZDIAS)
- Customer data (CUSTOMERS)

You will call these from your notebook.
"""

from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd


def encode_missing_values(
    df: pd.DataFrame,
    missing_value_map: Optional[Dict[str, List[int]]] = None
) -> pd.DataFrame:
    """
    Replace special 'unknown' encodings in the dataframe with np.nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing_value_map : dict, optional
        Dictionary mapping column names (or 'ALL') to a list of values
        that should be treated as missing.
        Example:
            {
                "ALL": [-1, -2, -3, -9],
                "CAMEO_INTL_2015": [-1, 0]
            }

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with all specified missing encodings replaced by np.nan.
    """
    df_clean = df.copy()

    if missing_value_map is None:
        # TODO: customize this based on AZDIAS_Feature_Summary.csv
        missing_value_map = {"ALL": [-1, -2, -3, -9]}

    # Apply global missing encodings
    if "ALL" in missing_value_map:
        df_clean.replace(missing_value_map["ALL"], np.nan, inplace=True)

    # Apply column-specific missing encodings
    for col, values in missing_value_map.items():
        if col == "ALL":
            continue
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(values, np.nan)

    return df_clean


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with a fraction of missing values above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float
        Maximum allowed fraction of missing values.
        e.g. 0.3 means: drop columns with > 30% missing.

    Returns
    -------
    df_reduced : pd.DataFrame
        Dataframe with high-missing columns removed.
    dropped_cols : list of str
        Names of dropped columns.
    """
    missing_frac = df.isna().mean()
    dropped_cols = missing_frac[missing_frac > threshold].index.tolist()

    df_reduced = df.drop(columns=dropped_cols)

    return df_reduced, dropped_cols


def split_by_row_missingness(
    df: pd.DataFrame,
    threshold: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows into two sets based on row-wise missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float
        Fraction of missing allowed.
        Rows with missing fraction <= threshold go to `main_df`,
        others go to `high_missing_df`.

    Returns
    -------
    main_df : pd.DataFrame
        Rows with acceptable missingness.
    high_missing_df : pd.DataFrame
        Rows with high missingness (potentially excluded or investigated separately).
    """
    row_missing_frac = df.isna().mean(axis=1)

    main_df = df[row_missing_frac <= threshold].copy()
    high_missing_df = df[row_missing_frac > threshold].copy()

    return main_df, high_missing_df


def convert_categorical(
    df: pd.DataFrame,
    binary_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle categorical features:
    - Convert selected binary categorical columns to 0/1.
    - One-hot encode other categorical columns.
    - Optionally drop some columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    binary_cols : list of str, optional
        Columns that are categorical but can be treated as binary.
        These may already be coded as 0/1 or 1/2, etc. You can recode manually here.
    drop_cols : list of str, optional
        Columns to drop entirely (e.g. IDs, free text, or unhelpful categories).

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with categorical features properly encoded.
    """
    df_encoded = df.copy()

    # Drop irrelevant / ID columns if provided
    if drop_cols:
        to_drop = [c for c in drop_cols if c in df_encoded.columns]
        df_encoded.drop(columns=to_drop, inplace=True)

    # Handle binary categorical columns (customize as needed)
    if binary_cols:
        for col in binary_cols:
            if col not in df_encoded.columns:
                continue
            # Example: if values are {1, 2}, map to {0, 1}
            # Adjust mapping based on data dictionary
            df_encoded[col] = df_encoded[col].map({1: 0, 2: 1}).astype("float")

    # One-hot encode remaining object / categorical columns
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    return df_encoded


def clean_demographic_data(
    df: pd.DataFrame,
    missing_value_map: Optional[Dict[str, List[int]]] = None,
    col_missing_threshold: float = 0.3,
    row_missing_threshold: float = 0.3,
    binary_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full cleaning pipeline for demographic data.

    Steps:
    1. Standardize missing value encodings -> np.nan
    2. Drop high-missing columns
    3. Split rows based on missingness, keep "main" subset
    4. Process categorical features

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    missing_value_map : dict, optional
        See `encode_missing_values`.
    col_missing_threshold : float
        Maximum allowed missing fraction per column.
    row_missing_threshold : float
        Maximum allowed missing fraction per row.
    binary_cols : list of str, optional
        See `convert_categorical`.
    drop_cols : list of str, optional
        See `convert_categorical`.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned, encoded dataframe ready for scaling/PCA.
    metadata : dict
        Dictionary with information useful for later:
        - 'dropped_columns'
        - 'high_missing_rows_count'
        - 'original_shape'
        - 'cleaned_shape'
    """
    metadata = {
        "original_shape": df.shape
    }

    # 1) Encode missing values consistently
    df_mv = encode_missing_values(df, missing_value_map)

    # 2) Drop columns with too many missing values
    df_cols, dropped_cols = drop_high_missing_columns(df_mv, threshold=col_missing_threshold)
    metadata["dropped_columns"] = dropped_cols

    # 3) Split by row-wise missingness
    main_df, high_missing_df = split_by_row_missingness(df_cols, threshold=row_missing_threshold)
    metadata["high_missing_rows_count"] = high_missing_df.shape[0]

    # 4) Convert categorical features
    cleaned_df = convert_categorical(
        main_df,
        binary_cols=binary_cols,
        drop_cols=drop_cols
    )

    metadata["cleaned_shape"] = cleaned_df.shape

    return cleaned_df, metadata
