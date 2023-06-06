# Stage 1: Stage 1 - Visualisation of piecewise linear curves on the plane and
# torus.

import plotly.graph_objects as go
import numpy as np


gamma = []  # List of N points in 2D here:
N = len(gamma)

# Plot them on the plane

x_coords, y_coords = zip(*gamma)

line = go.scatter3d(
    x=x_coords,
    y=y_coords,
    z=np.zeroes(N),
    mode="lines",)

Line_fig = go.Figure(data=[line])
Line_fig.show()
