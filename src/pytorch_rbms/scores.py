# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:57:30 2023

@author: COND04848
"""

from typing import Union, List
import torch
import numpy as np
from scipy.stats import wasserstein_distance


def calc_KL_divergence(
    Q: Union[List, np.array], P: Union[List, np.array, torch.tensor]
):
    """Get the Kullback Leibler divergence for a discrete probability function.
    Args:
        P, Q: Vector of probabilities between which to calculate the divergence.
            Q is the reference distribution
    Returns:
        KL-divergence"""

    # Convert to lists
    if torch.is_tensor(P) or isinstance(P, np.ndarray):
        P = np.array(P.tolist())
    if torch.is_tensor(Q) or isinstance(Q, np.ndarray):
        Q = np.array(Q.tolist())

    # Check if P and Q have the same length
    if not len(P) == len(Q):
        raise Exception("Probability distributions should have same length")

    logp = np.log(P)
    logp[logp == -float("inf")] = 0
    logq = np.log(Q)

    KL = P * logp - P * logq

    return KL


def calc_wasserstein_distance(
    Q: Union[List, np.array], P: Union[List, np.array, torch.tensor]
):
    """Get the Wasserstein distance for a discrete probability function.
    Args:
        P, Q: Vector of probabilities between which to calculate the distance.
            Q is the reference distribution
    Returns:
        Wasserstein distance"""

    # Convert to lists
    if torch.is_tensor(P) or isinstance(P, np.ndarray):
        P = np.array(P.tolist())
    if torch.is_tensor(Q) or isinstance(Q, np.ndarray):
        Q = np.array(Q.tolist())

    # Check if P and Q have the same length
    if not len(P) == len(Q):
        raise Exception("Probability distributions should have same length")

    WS_dist = wasserstein_distance(Q, P)

    return WS_dist


def calc_squared_error(
    Q: Union[List, np.array], P: Union[List, np.array, torch.tensor]
):
    """Get the squared error between two discrete probability functions.
    Args:
        P, Q: Vector of probabilities between which to calculate the error.
            Q is the reference distribution
    Returns:
        Squared error"""

    # Convert to lists
    if torch.is_tensor(P) or isinstance(P, np.ndarray):
        P = np.array(P.tolist())
    if torch.is_tensor(Q) or isinstance(Q, np.ndarray):
        Q = np.array(Q.tolist())

    # Check if P and Q have the same length
    if not len(P) == len(Q):
        raise Exception("Probability distributions should have same length")

    squared_error = (P - Q) ** 2

    return squared_error
