# Main code for computing the minimising curve.

import numpy as np
import math

# Just an example; R >> r is what matters.
R = 2000
r = 0.1


def f(x, y):  # R**2 * R**2 -> R**3
    # The following is an example for f.
    return np.array([R*math.cos(x) + r*math.cos(y)*math.cos(x),
                     R*math.sin(x) + r*math.cos(y)*math.sin(x), r*math.sin(y)])

# Surface is represented by f


PeriodsRange = [p for p in range(-1000, 2001)]


def CheckPeriodicity(f):
    # Need to generate random x's
    # Need to generate random y's
    for m in PeriodsRange:
        for n in PeriodsRange:
            assert np.allclose(f(x, y), f(x + n, y + m))


def D(f):  # Find derivative of f
    pass


# Define the curve in terms of points q0, q1, ... qN


N = 2000  # Example of a number that is large for us but small for a computer.

q = np.empty(shape=(N, 1))  # Each entry is a point on the surface.

assert np.isclose(f(q[0]), f(q[N]))  # q[N] - q[0] = (n,m)

# f(q) returns an array with all points mapped.
