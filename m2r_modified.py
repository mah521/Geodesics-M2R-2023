import plotly.graph_objects as go
import numpy as np
import sympy as sp


# PLOTTING STAGES 1 & 2

def plot_plane_and_torus(f, normal_f, point_2d_list, epsilon):
    # Create the line in R^2
    x_line = [point[0] for point in point_2d_list]
    y_line = [point[1] for point in point_2d_list]

    # Create the line in R^3
    line_3d = [np.array(f(point[0], point[1])) + epsilon *
               np.array(normal_f(point[0], point[1]))
               for point in point_2d_list]
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
    # Uncomment to plot in R^2
    # fig1.show()

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


# Example using a sympy expression:
# Define your variables
x, y = sp.symbols('x y')

# Define your function f as a sympy expression
f_sympy = ((2 + sp.cos(2 * sp.pi * y)) * sp.cos(2 * sp.pi * x),
           (2 + sp.cos(2 * sp.pi * y)) * sp.sin(2 * sp.pi * x),
           sp.sin(2 * sp.pi * y))

# Lambdify the function f
f = sp.lambdify((x, y), f_sympy)

# Compute the Jacobian as a lambda function
jacobian_f = sp.lambdify((x, y), sp.Matrix(f_sympy).jacobian((x, y)))

# Compute the normalized cross product of the gradients of f with respect
# to x and y
normal_f = lambda u, v: np.cross(jacobian_f(u, v)[:, 0], # NOQA E731
                                 jacobian_f(u, v)[:, 1])\
                                      / np.linalg.norm(np.cross(
                                          jacobian_f(u, v)[:, 0],
                                          jacobian_f(u, v)[:, 1]))

# Test the function
epsilon = 0.022
# point_2d_list = np.array([[i/30, i/30] for i in range(31)])
# Uncomment to plot
# plot_plane_and_torus(f, normal_f, point_2d_list, epsilon)


def generate_perturbed_line(n, m, N, lambda_val):
    # Generate evenly spaced points
    x_points = np.linspace(0, n, N+2)
    y_points = np.linspace(0, m, N+2)

    # Add random perturbation
    x_points += np.random.uniform(-lambda_val, lambda_val, size=N+2)
    y_points += np.random.uniform(-lambda_val, lambda_val, size=N+2)

    x_points[-1] = n + x_points[0]
    y_points[-1] = m + y_points[0]

    # Combine x and y coordinates
    points = list(zip(x_points, y_points))

    return np.array(points)


n = 1
m = 1
N = 100
lambda_val = 0.005

point_2d_list = generate_perturbed_line(n, m, N, lambda_val)

plot_plane_and_torus(f, normal_f, point_2d_list, epsilon)


# GRADIENT DESCENT PART - STAGES 3 & 4

# Function to calculate normalized vector difference
def normalized_diff(a, b, c, d):
    diff_vec = np.array(f(a, b)) - np.array(f(c, d))
    return diff_vec / np.linalg.norm(diff_vec)


# Define H
def H(point1, point2):
    a, b = point1
    c, d = point2

    # Compute normalized difference
    norm_diff = normalized_diff(a, b, c, d)

    # Calculate H1 and H2
    H1_val = np.dot(jacobian_f(a, b)[:, 0], norm_diff)
    H2_val = np.dot(jacobian_f(a, b)[:, 1], norm_diff)

    return [H1_val, H2_val]

# Define the gradient of Length function


def gradient_L(point_2d_list):
    n = len(point_2d_list) - 1
    new_list = []

    for i in range(len(point_2d_list)):
        if i == 0:  # First point
            new_val = np.add(H(point_2d_list[i], point_2d_list[i+1]),
                             H(point_2d_list[n], point_2d_list[n-1]))
        elif i == n:  # Last point
            new_val = np.add(H(point_2d_list[0], point_2d_list[1]),
                             H(point_2d_list[n], point_2d_list[n-1]))
        else:  # Other points
            new_val = np.add(H(point_2d_list[i], point_2d_list[i-1]),
                             H(point_2d_list[i], point_2d_list[i+1]))

        new_list.append(new_val)

    return new_list


# gradient descent process

def gradient_descent(N, alpha, point_2d_list):
    steps = [point_2d_list]

    for _ in range(N):
        last_step = steps[-1]
        new_step = last_step - alpha * np.array(gradient_L(last_step))
        steps.append(new_step.tolist())

    return steps


def animate_gradient_descent(steps):
    frames = []

    # Create the torus in R^3
    u = np.linspace(0, 1, 100)
    v = np.linspace(0, 1, 100)
    u, v = np.meshgrid(u, v)
    x_torus, y_torus, z_torus = f(u, v)

    for i, step in enumerate(steps):
        line_3d = [np.array(f(point[0], point[1])) + epsilon *
                   np.array(normal_f(point[0], point[1])) for point in step]
        x_line_3d = [point[0] for point in line_3d]
        y_line_3d = [point[1] for point in line_3d]
        z_line_3d = [point[2] for point in line_3d]

        frames.append(go.Frame(data=[go.Scatter3d(x=x_line_3d, y=y_line_3d,
                                                  z=z_line_3d, mode='lines',
                                                  line=dict(width=8,
                                                            color='red')),
                                     go.Surface(x=x_torus, y=y_torus,
                                                z=z_torus, opacity=1,
                                                showscale=False)]))

    # Plot the line in R^3 and the torus
    fig = go.Figure(
        data=[go.Scatter3d(x=x_line_3d, y=y_line_3d, z=z_line_3d,
                           mode='lines', line=dict(width=8, color='red')),
              go.Surface(x=x_torus, y=y_torus, z=z_torus,
                         opacity=1, showscale=False)],
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                              showactive=False,
                              y=1,
                              x=0.5,
                              xanchor='left',
                              yanchor='bottom',
                              pad=dict(t=45, r=10),
                              buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None,
                                                  dict(frame=dict(duration=500,
                                                                  redraw=True),
                                                       transition= # NOQA E251
                                                       dict(duration=0),
                                                       fromcurrent=True,
                                                       mode='immediate')])])]),
        frames=frames
    )
    fig.update_layout(scene=dict(xaxis_title='', yaxis_title='',
                                 zaxis_title='', xaxis_showgrid=False,
                                 yaxis_showgrid=False, zaxis_showgrid=False))
    fig.show()


# Generate the gradient descent steps
K = 300
alpha = 0.0001
steps = gradient_descent(K, alpha, point_2d_list)

# Animate the steps
animate_gradient_descent(steps)
