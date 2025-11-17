"""
clustering.py

Helpers for running KMeans clustering on PCA-transformed data.
"""

from typing import Tuple, List, Dict

import numpy as np
from sklearn.cluster import KMeans


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> KMeans:
    """
    Fit KMeans clustering on feature matrix X.

    Parameters
    ----------
    X : np.ndarray
        Input data (e.g., PCA-transformed).
    n_clusters : int
        Number of clusters to form.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    kmeans : KMeans
        Fitted KMeans model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto"
    )
    kmeans.fit(X)
    return kmeans


def compute_inertia_over_k(
    X: np.ndarray,
    k_range: List[int],
    random_state: int = 42
) -> Dict[int, float]:
    """
    Compute KMeans inertia (within-cluster sum of squares) for different k.

    Useful for elbow method.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    k_range : list of int
        List of k values to evaluate.
    random_state : int
        Random seed.

    Returns
    -------
    inertia_dict : dict
        Mapping from k -> inertia.
    """
    inertia_dict = {}
    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto"
        )
        model.fit(X)
        inertia_dict[k] = model.inertia_
    return inertia_dict


def assign_clusters(
    X: np.ndarray,
    model: KMeans
) -> np.ndarray:
    """
    Assign data points to clusters using a fitted KMeans model.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    model : KMeans
        Fitted KMeans model.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each data point.
    """
    return model.predict(X)
