get_ipython().magic('pylab inline')
from pylab import *

import control
import scipy.integrate

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.ipython_display_utils as ipython_display_utils
import flashlight.quadrotor_3d          as quadrotor_3d

m = quadrotor_3d.m
g = quadrotor_3d.g

x_star = matrix([0,0,0,0,0,0,0,0,0,0,0,0]).T
u_star = matrix([m*g/4.0,m*g/4.0,m*g/4.0,m*g/4.0]).T

Q = diag([1,1,1,1,1,1,1,1,1,1,1,1])
R = diag([1,1,1,1])

A, B    = quadrotor_3d.compute_df_dx_and_df_du(x_star, u_star)
K, S, E = control.lqr(A, B, Q, R)

x_disturbance = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0*pi, 0.0, 0.0]).T
x_0           = (x_star + x_disturbance).A1

def compute_x_dot(x_t, t):

    x_t     = matrix(x_t).T    
    x_bar_t = x_t - x_star
    u_bar_t = -K*x_bar_t
    u_t     = u_bar_t + u_star
    x_dot_t = quadrotor_3d.compute_x_dot(x_t, u_t).A1
    
    return x_dot_t

num_timesteps_sim = 200

t_begin = 0.0
t_end   = 10.0
t_sim   = linspace(t_begin, t_end, num_timesteps_sim)
x_sim   = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

quadrotor_3d.draw(t_sim, x_sim)
# quadrotor_3d.draw(t_sim, x_sim, savefig=True, out_dir="data/09", out_file="00.mp4")
ipython_display_utils.get_inline_video("data/09/00.mp4")

