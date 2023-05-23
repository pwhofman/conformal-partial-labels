import numpy as np
import torch

def compute_quantile(scores, alpha):
    """compute quantile from the scores
    Args:
        scores (list or np.array): scores of calibration data
        alpha (float): error rate in conformal prediction
    """
    n = scores.shape[0]
    return np.quantile(scores.detach().numpy(), np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")

def generate_sets(outputs, q):
    """Return sets including softmax outputs larger than 1 - the quantile"""
    sets = outputs >= (1-q)
    return sets

def efficiency(sets):
    """Return mean size of sets"""
    set_sizes = torch.sum(sets, dim=1)
    return torch.mean(set_sizes.float())

def coverage(sets, targets):
    """Return coverage of sets, i.e. the proportion of targets in sets"""
    return torch.sum(sets[torch.arange(targets.shape[0]),targets]) / sets.shape[0]