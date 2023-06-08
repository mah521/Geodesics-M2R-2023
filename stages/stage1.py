# Stage 1: Stage 1 - Visualisation of piecewise linear curves on the plane and
# torus.

import plotly.graph_objects as go
import numpy as np
# import matplotlib.pyplot as plt

"""
gamma = np.array([(0, 0), (1, 1), (4, 5),
                 (9, 10), (-10, 3.14)])  # List of N points in 2D here:
N = len(gamma)

# Plot them on the plane

x_coords, y_coords = zip(*gamma)

plt.plot(x_coords, y_coords)
plt.title("Plot of linear curve in the plane.")
plt.show()

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
"""


def plot_plane_and_torus(f, point_2d_list):
    # f is a function from R^2 to R^3
    # point_2d_list is a list of points in R^2

    # Desired output:
    # 1. A graphic of the line in R^2
    # 2. A graphic of the line in R^3 and the torus in R^3

    x, y = zip(*point_2d_list)

    x, y = np.array(x), np.array(y)

    line_trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Line'
    )

    # Create the layout
    layout = go.Layout(
        title='Line Graph',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y')
    )

    # Create the figure
    figure = go.Figure(data=[line_trace], layout=layout)

    # Display the figure
    figure.show()

    R = 5
    r = 2
    theta = np.linspace(0, 2.*np.pi, 100)
    phi = np.linspace(0, 2.*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    torus_x = (R + r * np.cos(theta)) * np.cos(phi)
    torus_y = (R + r * np.cos(theta)) * np.sin(phi)
    torus_z = r * np.sin(theta)

    torus = go.Surface(x=torus_x, y=torus_y, z=torus_z, colorscale='Viridis')

    f_x, f_y, f_z = f(x, y, R, r)

    torus_line = go.Scatter3d(x=f_x, y=f_y, z=f_z, mode="lines")

    Torus_fig = go.Figure(data=[torus, torus_line])
    Torus_fig.show()


# Example tryout:


def torus_transform(x, y, R, r):
    theta = 2 * np.pi * x
    phi = 2 * np.pi * y

    torus_x = (R + r * np.cos(theta)) * np.cos(phi)
    torus_y = (R + r * np.cos(theta)) * np.sin(phi)
    torus_z = r * np.sin(theta)

    return torus_x, torus_y, torus_z


gamma = np.array([[1, 1], [2, 3], [4, 7], [10, -20]])

plot_plane_and_torus(torus_transform, gamma)
