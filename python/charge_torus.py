#NAME: Charged Ring
#DESCRIPTION: Charged particle above a oppositely charged fixed ring.

import numpy as np
import charge_module as cm

from pycav import display

def a(R,ring_r,ring_steps,q):
	# Function to calculate the acceleration of the particle(s) at position R
	# R - position vector of the the particle(s)
	# ring_r -  radius of the charged ring
	# q - ratio of the charge on the particle to that on the ring
	# ring_steps - Number of sections of ring considered

	# Charge density of the ring
	rho = q/(2*np.pi*ring_r)

	# Azimuthal angle steps
	phi_h = 2*np.pi/float(ring_steps)
	
	total_acc = np.array((0.0,0.0,0.0))
	for i in range(ring_steps):
		# Azimuthal angle
		phi = i*phi_h
		# Relative position vector of particle to element of ring dq
		P = R-np.array((ring_r*np.cos(phi),ring_r*np.sin(phi),0))
	
		# Magnitude of the relative position vector
		P_mag = np.linalg.norm(P)

		acc = -ring_r*phi_h*rho*P/(4*np.pi*P_mag**3)

		total_acc = total_acc + acc

	return total_acc

def shm_freq(ring_r,q):
	return (q/(4*np.pi*ring_r**3))**0.5

# initial conditions
# Particle position and velocity
R = np.array((0.0,0.0,0.01))
V = np.array((0.0,0.0,0.0))

# Ring radius
ring_r = 1.0
# Ring sections
ring_steps = 25
# Charge factor = qQ/epsilon_0
q = 10000.0

# Time step
h = 10.0**-3
N = 1000

# Animation frame change
N_output = 10

Anim = cm.Animation(a,q,ring_r,ring_steps,h,N,N_output)
Anim.create_sliders(R,V)

animate = display.create_animation(Anim.animate, temp = True)
display.display_animation(animate)

Anim2 = cm.Animation(a,q,ring_r,ring_steps,h,N,N_output)

Anim2.R_i = np.array((0.0,0.0,0.01))
Anim2.ring_steps = 2
Anim2.create_sliders(R,V)

animate_2 = display.create_animation(Anim2.animate, temp = True)
display.display_animation(animate_2)

# Distance over which to plot in units of ring_r
z_lim = 4.0
# Step size used in plotting
z_step = 0.05

get_ipython().magic('matplotlib inline')

cm.E_field(z_lim,z_step,q,ring_r)



