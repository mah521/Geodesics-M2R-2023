"""Stage 2 - Compute the (Approximated) Length of a curve on the torus:"""

import numpy as np

'''
def EuclideanDistance(A, B):
    """Euclidean distance A to B."""
    try:
        assert len(A) == len(B)
    except AssertionError:
        raise ValueError(f"{A} and {B} must be the same dimension")
    else:
        return sum([(A[p] - B[p])**2 for p in range(len(A))])**0.5
'''


def GetGammaLength(gamma):
    """Approximate path length along gamma (a list of 2D points.)"""
    N = len(gamma)
    total_length = 0
    for n in range(0, N-1):
        point1 = gamma[n]
        point2 = gamma[n+1]
        segment_length = np.linalg.norm(point1 - point2)

        total_length += segment_length

    return total_length
