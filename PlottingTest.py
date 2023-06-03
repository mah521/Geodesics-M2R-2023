# Let's try these libraries to try to make 
# an interactive animation

import numpy as np
import plotly.graph_objects as go

# This is the same parametrisation of the
# torus that we discussed in my office 

# we can choose different radi
R = 5
r = 2

# For the parameter space it is enough to use just one
# representing square
# make sure to a square of size 1
theta = np.linspace(0, 2.*np.pi, 100)
phi = np.linspace(0, 2.*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Finally, these are the coordinates of the torus
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)

# With this library we can create a plot object,
# that means this is an absctract plot which is not
# displayed yet

torus = go.Surface(x=x, y=y, z=z, colorscale='Viridis')

# To display the plot we need to choose how to display the 
# plot object, for example, let's go with a figure

# Create a figure based on the plot:
fig = go.Figure(data=[torus])

# Display the figure:
fig.show()

# We can also use the library to make line plot 
# this is when we give the computer a sequence of points
# and it plots the line curve joining those points 
# consecutively 

# For example, this is a list of points in 3D space:
points = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 1, 2), (3, 2, 2)]

# We can use zip() function to separate the x, y and z coordinates:
x_coordinates, y_coordinates, z_coordinates = zip(*points)

# And we use the Scatter3d function to make a line plot:
line = go.Scatter3d(
    x=x_coordinates, 
    y=y_coordinates, 
    z=z_coordinates, 
    mode='lines')

# Just as above we can visualise what we did using figure and show:
fig = go.Figure(data=[line])
fig.show()

# When we want to visualise both the torus and the line 
# we can do something like:

fig = go.Figure(data=[torus, line])
fig.show()

# Because the nature of our project involves a minimisation
# procedure, it is desirable to present an animation of what is
# happening. Additionally, having an visualisation of all the
# steps of our process will also allow us to debug quickly
# any problems that appear along the way

# We can make animations by simply passing an array of 
# figure objects. Here is an example which consists of 
# a plane, which is fixed, and a circle line that starts 
# as contained in the plane and rotates around an axis 
# contained in the plane

# Define the plane (a 2D grid in the x-z plane)
x_plane = np.linspace(-5, 5, 10)
z_plane = np.linspace(-5, 5, 10)
x_plane, z_plane = np.meshgrid(x_plane, z_plane)
y_plane = np.zeros_like(x_plane)

plane = go.Surface(x=x_plane, y=y_plane, 
                   z=z_plane, opacity=0.5, showscale=False)

# Define the circle (centered at the origin, in the x-z plane)
# This is an example of a smooth curve in this library
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
z_circle = np.sin(theta)

# Now to the important bit
# we create a collection of frames for the 
# rotating circle:

# we begin with an empty array: 
frames = []

# then we generate a frame for each rotation
# and append it to the list frame:
for i in range(10):
    # Each frame rotates the circle by an
    # additional 10 degrees around the x-axis
    rotation_angle = i * np.pi / 18
    y_circle = np.sin(rotation_angle) * z_circle
    z_circle_rotated = np.cos(rotation_angle) * z_circle
    circle = go.Scatter3d(x=x_circle, y=y_circle, 
                          z=z_circle_rotated, mode='lines')
    frames.append(go.Frame(data=[plane, circle]))

# Create then pass the array to the go.Figure:
fig = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        scene=dict(xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]),
                   zaxis=dict(range=[-5, 5])),
        updatemenus=[dict(type="buttons",
                          showactive=False,
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])]),
    frames=frames
)

# And then we show our animation!
fig.show()

# Note that before you start the animation below, you can
# rotate the view, this interactivity will come in handy
# Also note that in this case we are making the surface
# more transparent so that we can clearly visualise the curve
# please play aroun with those features in your project
# so that we can create good visualisations