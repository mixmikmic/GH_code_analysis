from fenics import *

get_ipython().magic('matplotlib inline')

mesh = UnitCubeMesh(10, 10, 10)

plot(mesh)

from mshr import *

# Create list of polygonal domain vertices
domain_vertices = [Point(0.0, 0.0),
                   Point(10.0, 0.0),
                   Point(10.0, 2.0),
                   Point(8.0, 2.0),
                   Point(7.5, 1.0),
                   Point(2.5, 1.0),
                   Point(2.0, 4.0),
                   Point(0.0, 4.0),
                   Point(0.0, 0.0)]

pg = Polygon(domain_vertices)

# Generate mesh and plot
mesh = generate_mesh(pg, 20);
plot(mesh, interactive=True)

r = Rectangle(Point(0.5, 0.5), Point(1.5, 1.5))

c = Circle(Point(1, 1), 1)

g2d = c - r

# Generate and plot mesh
mesh2d = generate_mesh(g2d, 10)
plot(mesh2d, title="2D mesh")

box = Box(Point(0, 0, 0), Point(1, 1, 1))

sphere = Sphere(Point(0, 0, 0), 0.3)

cone1 = Cone(Point(0., 0., -1.), Point(0., 0., 3.), 1.)

cone2 = Cone(Point(0., 0., 1.), Point(0., 0., 3.), 0.5)

g3d = box + cone1 - cone2 - sphere

mesh3d = generate_mesh(g3d, 32)
info(mesh3d)
plot(mesh3d, "3D mesh")

