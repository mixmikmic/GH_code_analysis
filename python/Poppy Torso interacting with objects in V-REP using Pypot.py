from pypot.creatures import PoppyTorso

poppy = PoppyTorso(simulator='vrep')

io = poppy._controllers[0].io

name = 'cube'
position = [0.2, 0, 0.8] # X, Y, Z
sizes = [0.1, 0.1, 0.1] # in meters
mass = 0.5 # in kg
io.add_cube(name, position, sizes, mass)

io.set_object_position('cube', [0.15,0.05,0.8])

{m.name: m.present_position for m in poppy.motors}

poppy.l_arm_z.goal_position = -30

io.get_object_position('cube')

io.get_object_orientation('cube')

Z_orientation = io.get_object_orientation('cube')[2] * 180./3.14
Z_orientation

io.add_sphere('ball1', [0.1, 0, 0.8], [0.1, 0.1, 0.2], 2)

io.add_cylinder('cylinder1', [0.1, 0, 0.8], [0.1, 0.1, 0.2], 2, [1000,1000])

io.add_cone('cone1', [0.1, 0, 0.8], [0.1, 0.1, 0.2], 2, [1000,1000])

from IPython.display import VimeoVideo
VimeoVideo(127023576)

