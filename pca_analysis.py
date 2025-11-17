"""
pca_analysis.py

Helpers for scaling and applying PCA to the cleaned demographic data.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_features(
    df: pd.DataFrame
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply standard scaling to features (mean=0, std=1).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned numeric dataframe.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature array.
    scaler : StandardScaler
        Fitted scaler (to re-use on customers data).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    return X_scaled, scaler


def transform_with_scaler(
    df: pd.DataFrame, scaler: StandardScaler
) -> np.ndarray:
    """
    Use an existing scaler to transform a new dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        New cleaned dataframe.
    scaler : StandardScaler
        Fitted scaler from general population data.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature array.
    """
    return scaler.transform(df.values)


def fit_pca(
    X: np.ndarray,
    n_components: int
) -> PCA:
    """
    Fit PCA on scaled data.

    Parameters
    ----------
    X : np.ndarray
        Scaled data.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return pca


def apply_pca(
    X: np.ndarray,
    pca: PCA
) -> np.ndarray:
    """
    Project data into PCA space using a fitted PCA model.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature array.
    pca : PCA
        Fitted PCA object.

    Returns
    -------
    X_pca : np.ndarray
        PCA-transformed data.
    """
    return pca.transform(X)


def get_explained_variance(pca: PCA) -> np.ndarray:
    """
    Get variance ratios for each principal component.

    Returns
    -------
    explained_variance_ratio_ : np.ndarray
    """
    return pca.explained_variance_ratio_


def get_component_loadings(
    pca: PCA,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Create a DataFrame of PCA component loadings to interpret components.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    feature_names : list of str
        Names of original features.

    Returns
    -------
    loadings_df : pd.DataFrame
        Dataframe with components as rows and feature loadings as columns.
    """
    loadings = pca.components_
    loadings_df = pd.DataFrame(
        loadings,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(loadings.shape[0])]
    )
    return loadings_df
