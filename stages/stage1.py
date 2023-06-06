# Stage 1: Stage 1 - Visualisation of piecewise linear curves on the plane and
# torus.

import plotly.graph_objects as go
import numpy as np


gamma = [(0, 0), (1, 1), (4, 5),
         (9, 10), (-10, 3.14)]  # List of N points in 2D here:
N = len(gamma)

# Plot them on the plane

x_coords, y_coords = zip(*gamma)

line = go.Scatter3d(
    x=x_coords,
    y=y_coords,
    z=np.zeros(N),
    mode="lines",)

Line_fig = go.Figure(data=[line])
Line_fig.show()

# Plot them on a torus


def f(x, y):  # This is the function to map from x, y to the surface.
    pass

#  Copied from PlottingTest:


R = 5
r = 2
theta = np.linspace(0, 2.*np.pi, 100)
phi = np.linspace(0, 2.*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
torus_x = (R + r * np.cos(theta)) * np.cos(phi)
torus_y = (R + r * np.cos(theta)) * np.sin(phi)
torus_z = r * np.sin(theta)

torus = go.Surface(x=torus_x, y=torus_y, z=torus_z, colorscale='Viridis')

f_x, f_y, f_z = f(x_coords, y_coords)

torus_line = go.Scatter3d(x=f_x, y=f_y, z=f_z, mode="lines")

Torus_fig = go.Figure(data=[torus, torus_line])
Torus_fig.show()
