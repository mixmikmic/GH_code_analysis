get_ipython().magic('pylab inline')
from pylab import *

import control
import scipy.integrate

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.ipython_display_utils as ipython_display_utils
import flashlight.quadrotor_3d          as quadrotor_3d
import flashlight.trig_utils            as trig_utils

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
# quadrotor_3d.draw(t_sim, x_sim, savefig=True, out_dir="data/11", out_file="00.mp4")
ipython_display_utils.get_inline_video("data/11/00.mp4")

x_disturbance = matrix([0.0, 0.0, 0.0, 2*pi, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0*pi, 0.0, 0.0]).T
x_0           = (x_star + x_disturbance).A1
x_sim         = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

quadrotor_3d.draw(t_sim, x_sim)
# quadrotor_3d.draw(t_sim, x_sim, savefig=True, out_dir="data/11", out_file="01.mp4")
ipython_display_utils.get_inline_video("data/11/01.mp4")

x_disturbance = matrix([0.0, 0.0, 0.0, 2*pi, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0*pi, 0.0, 0.0]).T
x_0           = (x_star + x_disturbance).A1

p_star, theta_star, psi_star, phi_star, p_dot_star, theta_dot_star, psi_dot_star, phi_dot_star, q_star, q_dot_star =     quadrotor_3d.unpack_state(x_star)

def compute_x_dot(x_t, t):

    x_t = matrix(x_t).T
    p_t, theta_t, psi_t, phi_t, p_dot_t, theta_dot_t, psi_dot_t, phi_dot_t, q_t, q_dot_t =         quadrotor_3d.unpack_state(x_t)

    theta_star_hat_t = theta_t + trig_utils.compute_smallest_angular_diff(theta_t, theta_star)    
    psi_star_hat_t   = psi_t   + trig_utils.compute_smallest_angular_diff(psi_t,   psi_star)
    phi_star_hat_t   = phi_t   + trig_utils.compute_smallest_angular_diff(phi_t,   phi_star)    
    x_star_hat_t, _, _ =         quadrotor_3d.pack_state(p_star,     theta_star_hat_t, psi_star_hat_t, phi_star_hat_t,
                                p_dot_star, theta_dot_star,   psi_dot_star,   phi_dot_star)

    x_bar_t = x_t - x_star_hat_t
    u_bar_t = -K*x_bar_t
    u_t     = u_bar_t + u_star
    x_dot_t = quadrotor_3d.compute_x_dot(x_t, u_t).A1
    
    return x_dot_t

x_sim = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

quadrotor_3d.draw(t_sim, x_sim)
# quadrotor_3d.draw(t_sim, x_sim, savefig=True, out_dir="data/11", out_file="02.mp4")
ipython_display_utils.get_inline_video("data/11/02.mp4")

