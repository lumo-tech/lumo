import numpy as np


def mixup(major, minor=None, lam=None, alpha=2):
    if lam is None:
        lam = beta(alpha)
    lam = max(lam, 1 - lam)
    if minor is None:
        reid = permutation(len(major))
        minor = major[reid]
    return major * lam + minor * (1 - lam), lam


def permutation(size, nonoverlap=True):
    if nonoverlap:
        half = size // 2
        reid = np.concatenate(
            [
                np.random.permutation(half) + half,
                np.random.permutation(half),
            ]
        )
    else:
        reid = np.random.permutation(size)
    return reid


def beta(alpha=2.0):
    """
    Return lambda
    Args:
        alpha:

    Returns:

    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam
