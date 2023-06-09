"""A version of stage1.py that has implemented""" + \
    """suggestions/corrections by ChatGPT."""
 
import plotly.graph_objects as go
import numpy as np


def plot_plane_and_torus(f, point_2d_list):
    # f is a function from R^2 to R^3
    # point_2d_list is a list of points in R^2

    # Desired output:
    # 1. A graphic of the line in R^2
    # 2. A graphic of the line in R^3 and the torus in R^3

    x, y = zip(*point_2d_list)
    x, y = np.array(x), np.array(y)

    # Create the 2D line trace
    line_trace_2d = go.Scatter(
        x=x / (2 * np.pi),
        y=y / (2 * np.pi),
        mode='lines',
        name='Line in R^2'
    )

    # Create the layout for the 2D plot
    layout_2d = go.Layout(
        title='Line in R^2',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y')
    )

    # Create the figure for the 2D plot
    figure_2d = go.Figure(data=[line_trace_2d], layout=layout_2d)

    # Display the 2D plot
    figure_2d.show()

    R = 5
    r = 2
    theta = np.linspace(0, 2. * np.pi, 100)
    phi = np.linspace(0, 2. * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    torus_x = (R + r * np.cos(theta)) * np.cos(phi)
    torus_y = (R + r * np.cos(theta)) * np.sin(phi)
    torus_z = r * np.sin(theta)

    torus = go.Surface(x=torus_x, y=torus_y, z=torus_z, colorscale='Viridis')

    f_x, f_y, f_z = f(x, y, R, r)
    # print(f_x, f_y, f_z)

    torus_line = go.Scatter3d(x=f_x, y=f_y, z=f_z,
                              mode="lines", name='Line in R^3')

    # Create the layout for the 3D plot
    layout_3d = go.Layout(
        title='Line and Torus in R^3',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # Create the figure for the 3D plot
    figure_3d = go.Figure(data=[torus, torus_line], layout=layout_3d)

    # Display the 3D plot
    figure_3d.show()


def torus_transform(x, y, R, r):
    theta = 2 * np.pi * x
    phi = 2 * np.pi * y

    torus_x = (R + r * np.cos(theta)) * np.cos(phi)
    torus_y = (R + r * np.cos(theta)) * np.sin(phi)
    torus_z = r * np.sin(theta)

    return torus_x, torus_y, torus_z


gamma = np.array([[1, 1], [2, 3], [4, 7], [10, -20]])

# Example tryout:
gamma = np.array([[1, 1], [2, 3], [4, 7], [10, -20]])

plot_plane_and_torus(torus_transform, gamma)
