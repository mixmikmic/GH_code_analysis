get_ipython().magic('pylab inline')
figsize(9,6)

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight
print flashlight.__version__

from pylab import *; import scipy.integrate

import flashlight.interpolate_utils as interpolate_utils
import flashlight.quadrotor_2d      as quadrotor_2d

# Define a simple position trajectory in 2D.
num_samples = 200
t_begin     = 0
t_end       = pi
dt          = (t_end - t_begin) / (num_samples - 1)

t = linspace(t_begin, t_end, num_samples)
p = c_[ sin(2*t) + t, t**2 ]

# Compute the corresponding state space trajectory and control trajectories for a 2D quadrotor.
q_qdot_qdotdot = quadrotor_2d.compute_state_space_trajectory_and_derivatives(p, dt)
u              = quadrotor_2d.compute_control_trajectory(q_qdot_qdotdot)

# Define a function that interpolates the control trajectory in between time samples.
u_interp_func = interpolate_utils.interp1d_vector_wrt_scalar(t, u, kind="cubic")

# Define a simulation loop.
def compute_x_dot(x_t, t):

    # Get the current control vector.
    u_t = u_interp_func(clip(t, t_begin, t_end))
    
    # Compute the state derivative from the current state and current control vectors.
    x_dot_t = quadrotor_2d.compute_x_dot(x_t, u_t).A1

    return x_dot_t

# Simulate.
x_nominal, _, _, _ = quadrotor_2d.pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot)
x_0                = x_nominal[0]
x_sim              = scipy.integrate.odeint(compute_x_dot, x_0, t)

# Plot the results.
quadrotor_2d.draw(t, x_sim, t_nominal=t, x_nominal=x_nominal, inline=True)

