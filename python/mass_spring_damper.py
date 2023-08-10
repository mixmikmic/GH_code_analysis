import sympy as sym
import sympy.physics.mechanics as me

from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax')

x, v = me.dynamicsymbols('x v')

m, c, k, g, t = sym.symbols('m c k g t')

ceiling = me.ReferenceFrame('C')

O = me.Point('O')
P = me.Point('P')

O.set_vel(ceiling, 0)

P.set_pos(O, x * ceiling.x)
P.set_vel(ceiling, v * ceiling.x)
P.vel(ceiling)

damping = -c * P.vel(ceiling)
stiffness = -k * P.pos_from(O)
gravity = m * g * ceiling.x
forces = damping + stiffness + gravity
forces

zero = me.dot(forces - m * P.acc(ceiling), ceiling.x)
zero

dv_by_dt = sym.solve(zero, v.diff(t))[0]
dx_by_dt = v
dv_by_dt, dx_by_dt

mass = me.Particle('mass', P, m)

kane = me.KanesMethod(ceiling, q_ind=[x], u_ind=[v], kd_eqs=[v - x.diff(t)])

fr, frstar = kane.kanes_equations([(P, forces)], [mass])
fr, frstar

M = kane.mass_matrix_full
f = kane.forcing_full
M, f

M.inv() * f

from pydy.system import System

sys = System(kane)

sys.constants = {m:10.0, g:9.8, c:5.0, k:10.0}
sys.initial_conditions = {x:0.0, v:0.0}

from numpy import linspace
sys.times = linspace(0.0, 10.0, 100)

x_trajectory = sys.integrate()

from pydy.viz import *

bob = Sphere(2.0, color="red", material="metal")
bob_vframe = VisualizationFrame(ceiling, P, bob)

ceiling_circle = Circle(radius=10, color="white", material="metal")
from numpy import pi
rotated = ceiling.orientnew("C_R", 'Axis', [pi / 2, ceiling.z])
ceiling_vframe = VisualizationFrame(rotated, O, ceiling_circle)

scene = Scene(ceiling, O, system=sys)

scene.visualization_frames = [bob_vframe, ceiling_vframe]

camera_frame = ceiling.orientnew('Camera Frame','Axis', [pi / 2, ceiling.z])
camera_point = O.locatenew('Camera Location', 100 * camera_frame.z)
primary_camera = PerspectiveCamera(camera_frame, camera_point)
scene.cameras = [primary_camera]

scene.display_ipython()

get_ipython().magic('load_ext version_information')

get_ipython().magic('version_information pydy, numpy, scipy')

