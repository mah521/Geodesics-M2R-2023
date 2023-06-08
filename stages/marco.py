"""A version of the code suggested by the supervisor."""

import plotly.graph_objects as go
import numpy as np


def plot_plane_and_torus(f, point_2d_list):

    # Create the line in R^2

    x_line = [point[0] for point in point_2d_list]

    y_line = [point[1] for point in point_2d_list]

    # Create the line in R^3

    line_3d = [f(point[0], point[1]) for point in point_2d_list]

    x_line_3d = [point[0] for point in line_3d]

    y_line_3d = [point[1] for point in line_3d]

    z_line_3d = [point[2] for point in line_3d]

    # Create the torus in R^3

    u = np.linspace(0, 1, 100)

    v = np.linspace(0, 1, 100)

    u, v = np.meshgrid(u, v)

    x_torus, y_torus, z_torus = f(u, v)

    # Plot the line in R^2

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=x_line, y=y_line,
                              mode='lines', line=dict(width=4)))

    fig1.update_layout(scene=dict(xaxis_title='', yaxis_title='',
                                  xaxis_showgrid=False, yaxis_showgrid=False))

    fig1.show()

    # Plot the line in R^3 and the torus

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter3d(x=x_line_3d, y=y_line_3d, z=z_line_3d,
                                mode='lines', line=dict(width=8, color='red')))

    fig2.add_trace(go.Surface(x=x_torus, y=y_torus, z=z_torus, opacity=1,
                              showscale=False))

    fig2.update_layout(scene=dict(xaxis_title='', yaxis_title='',
                                  zaxis_title='', xaxis_showgrid=False,
                                  yaxis_showgrid=False, zaxis_showgrid=False))

    fig2.show()


# Example:


f = lambda u, v: ((2 + np.cos(2 * np.pi * v)) *  # NOQA:E731
                  np.cos(2 * np.pi * u),

                  (2 + np.cos(2 * np.pi * v)) * np.sin(2 * np.pi * u),

                  np.sin(2 * np.pi * v))

point_2d_list = np.array([[i/10, i/10] for i in range(11)])

plot_plane_and_torus(f, point_2d_list)
