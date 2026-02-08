"""
Numerically stable softmax implementation used inside attention to convert
raw similarity scores into probability distributions.
"""

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply a numerically stable softmax along a specified dimension.

    x:   Input tensor of arbitrary shape
    (dim) Dimension over which probabilities are computed
    Returns: Tensor of same shape with values summing to 1 along dim
    """
    # Shift values by the maximum to prevent exponential overflow
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max

    # Exponentiate the stabilized values
    exp_x = torch.exp(x_stable)

    # Normalize so probabilities sum to 1 along the specified dimension
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x