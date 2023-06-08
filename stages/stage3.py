# Stage 3 - Compute the gradient of the (Approximated Length)

def Length(Gamma, f):
    """Function to find the length of the curve"""
    """approximated by points Gamma."""

    # Gamma is a numpy array of length n+1 {q_0, ... q_n+1}
    # Where q_k is a 2D vector
    # q_n+1 = q_1 + (a, b) for integer a, b

    n = len(Gamma) - 1

    abs_sum = 0

    for k in range(1, n + 1, 1):
        abs_sum += abs(f(Gamma[k - 1]) - f(Gamma[k]))

    return abs_sum


def G(f, x, y, w, z):
    return abs(f(x, y) - f(w, z))


def Length_G(Gamma, f):
    """Same as Length but computes using G."""

    n = len(Gamma) - 1

    G_sum = 0

    x, y = zip(*Gamma)

    a, b = x[n] - x[0], y[n] - y[0]

    for k in range(1, n, 1):
        G_sum += G(x[k - 1], y[k - 1], x[k], y[k])

    G_sum += G(x[n - 1], y[n - 1], x[0] + a, y[0] + b)
