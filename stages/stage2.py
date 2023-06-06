"""Stage 2 - Compute the (Approximated) Length of a curve on the torus:"""


def GetGammaLength(gamma):
    N = len(gamma)
    total_length = 0
    for n in range(0, N-1):
        point1 = gamma[n]
        point2 = gamma[n+1]
        segment_length = ((point1[0]-point2[0])**2 +
                          (point1[1]-point2[1])**2) ** 0.5

        total_length += segment_length

    return total_length
