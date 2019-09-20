"""
This module provides utility functions that are not tied to specific
classes or concepts, but still perform specific and important roles
across many of the packages modules.

"""
import numpy as np


def get_indices_and_probabilities(interval_lengths, indices):
    """
    Given a list of different interval lengths and the indices of
    interest in that list, weight the probability that we will sample
    one of the indices in `indices` based on the interval lengths in
    that sublist.

    Parameters
    ----------
    interval_lengths : list(int)
        The list of lengths of intervals that we will draw from. This is
        used to weight the indices proportionally to interval length.
    indices : list(int)
        The list of interval length indices to draw from.

    Returns
    -------
    indices, weights : tuple(list(int), list(float)) \
        Tuple of interval indices to sample from and the corresponding
        weights of those intervals.

    """
    select_interval_lens = np.array(interval_lengths)[indices]
    weights = select_interval_lens / float(np.sum(select_interval_lens))

    keep_indices = []
    for index, weight in enumerate(weights):
        if weight > 1e-10:
            keep_indices.append(indices[index])
    if len(keep_indices) == len(indices):
        return indices, weights.tolist()
    else:
        return get_indices_and_probabilities(
            interval_lengths, keep_indices)
