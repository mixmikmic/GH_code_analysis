get_ipython().magic('pylab inline')
from pylab import *

import control
import matplotlib.animation
import scipy.integrate

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.curve_utils       as curve_utils
import flashlight.quadrotor_2d      as quadrotor_2d
import flashlight.spline_utils      as spline_utils
import flashlight.trig_utils        as trig_utils

m = quadrotor_2d.m
g = quadrotor_2d.g

x_star = matrix([0,0,0,0,0,0]).T
u_star = matrix([m*g/2.0,m*g/2.0]).T

Q = diag([1,1,1,1,1,1])
R = diag([1,1])

A, B    = quadrotor_2d.compute_df_dx_and_df_du(x_star, u_star)
K, S, E = control.lqr(A, B, Q, R)

x_disturbance = matrix([0.0, 0.0, 0.0, 5.0, 5.0, -4.0*pi]).T
x_0           = (x_star + x_disturbance).A1

def compute_x_dot(x_t, t):

    x_t     = matrix(x_t).T
    x_bar_t = x_t - x_star
    u_bar_t = -K*x_bar_t
    u_t     = u_bar_t + u_star
    x_dot_t = quadrotor_2d.compute_x_dot(x_t, u_t).A1
    
    return x_dot_t

num_timesteps_sim = 200

t_begin = 0.0
t_end   = 10.0
t_sim   = linspace(t_begin, t_end, num_timesteps_sim)
x_sim   = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

figsize(6,4)
quadrotor_2d.draw(t_sim, x_sim, inline=True)

x_disturbance = matrix([0.0, 0.0, 8.0*pi, 5.0, 5.0, -4.0*pi]).T
x_0           = (x_star + x_disturbance).A1
x_sim         = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

figsize(6,4)
quadrotor_2d.draw(t_sim, x_sim, inline=True)

x_disturbance = matrix([0.0, 0.0, 8.0*pi, 5.0, 5.0, -4.0*pi]).T
x_0           = (x_star + x_disturbance).A1

p_star, theta_star, p_dot_star, theta_dot_star, q_star, q_dot_star = quadrotor_2d.unpack_state(x_star)

def compute_x_dot(x_t, t):

    x_t = matrix(x_t).T
    p_t, theta_t, p_dot_t, theta_dot_t, q_t, q_dot_t = quadrotor_2d.unpack_state(x_t)

    theta_star_hat_t   = theta_t + trig_utils.compute_smallest_angular_diff(theta_t, theta_star)    
    x_star_hat_t, _, _ = quadrotor_2d.pack_state(p_star, theta_star_hat_t, p_dot_star, theta_dot_star)
    
    x_bar_t = x_t - x_star_hat_t
    u_bar_t = -K*x_bar_t
    u_t     = u_bar_t + u_star
    x_dot_t = quadrotor_2d.compute_x_dot(x_t, u_t).A1
    
    return x_dot_t

x_sim = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

figsize(6,4)
quadrotor_2d.draw(t_sim, x_sim, inline=True)

