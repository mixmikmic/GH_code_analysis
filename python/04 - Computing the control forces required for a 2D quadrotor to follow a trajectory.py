get_ipython().magic('pylab inline')
from pylab import *

import matplotlib.animation
import scipy.integrate

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.curve_utils       as curve_utils
import flashlight.interpolate_utils as interpolate_utils
import flashlight.quadrotor_2d      as quadrotor_2d
import flashlight.spline_utils      as spline_utils

T_y = matrix([0,1,2,3]).T.A
T_x = matrix([0,1,2,3]).T.A
T   = c_[T_y, T_x]
P_y = matrix([0,3,1,4]).T.A
P_x = matrix([0,3,7,10]).T.A
P   = c_[P_y, P_x]

num_samples = 200

C, T, sd =     spline_utils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(
        P, T, degree=7, lamb=[0,0,0,1,0])
    
P_eval, T_eval, dT =     spline_utils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(
        C, T, sd, num_samples=num_samples)

T_s = matrix([0.0,1.2,1.8,3.0]).T.A
P_s = matrix([0.0,0.3,0.7,1.0]).T.A

C_s, T_s, sd_s =     spline_utils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(
        P_s, T_s, degree=7, lamb=[0,0,0,1,0])
    
P_s_eval, T_s_eval, dT_s =     spline_utils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(
        C_s, T_s, sd_s, num_samples=num_samples)

t = linspace(0.0,10.0,num_samples)
s_spline = P_s_eval

P_eval_spline, t_spline, P_eval_cum_length, t_norm = curve_utils.reparameterize_curve(P_eval, s_spline)

figsize(6,6);
scatter(P_eval_spline[:,1], P_eval_spline[:,0], c=t, s=50);
gca().set_aspect("equal")
title("$\\mathbf{p}(t)$\n", fontsize=20);
ylabel("$\\mathbf{p}_y$", rotation="horizontal", fontsize=20); xlabel("$\\mathbf{p}_x$", fontsize=20);

t_begin       = t[0]
t_end         = t[-1]
num_timesteps = num_samples

p  = P_eval_spline
dt = (t_end-t_begin) / (num_timesteps-1)

q_qdot_qdotdot = quadrotor_2d.compute_state_space_trajectory_and_derivatives(p, dt)

p, p_dot, p_dot_dot, theta, theta_dot, theta_dot_dot = q_qdot_qdotdot

figsize(14,9)

subplot(331); plot(t, p[:,0]);
title("$\\mathbf{p}_y(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(332); plot(t, p[:,1]);
title("$\\mathbf{p}_x(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(333); plot(t, theta);
title("$\\theta(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(334); plot(t, p_dot[:,0]);
title("$\\dot{\\mathbf{p}}_y(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(335); plot(t, p_dot[:,1]);
title("$\\dot{\\mathbf{p}}_x(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(336); plot(t, theta_dot);
title("$\\dot{\\theta}(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(337); plot(t, p_dot_dot[:,0]);
title("$\\ddot{\\mathbf{p}}_y(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(338); plot(t, p_dot_dot[:,1]);
title("$\\ddot{\\mathbf{p}}_x(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(339); plot(t, theta_dot_dot);
title("$\\ddot{\\theta}(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

gcf().tight_layout();

u = quadrotor_2d.compute_control_trajectory(q_qdot_qdotdot)

figsize(5,2)
plot(t, u[:,0]); plot(t, u[:,1]);
title("$\\mathbf{u}(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

x_nominal, q_nominal, qdot_nominal, qdotdot_nominal =     quadrotor_2d.pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot)
    
x_0 = x_nominal[0]

def compute_x_dot(x_t, t):

    x_t     = matrix(x_t).T
    u_t     = u_interp_func(clip(t, t_begin, t_end))
    x_dot_t = quadrotor_2d.compute_x_dot(x_t, u_t).A1
    
    return x_dot_t

num_timesteps_sim = 200
t_sim             = linspace(t_begin, t_end, num_timesteps_sim)
u_interp_func     = interpolate_utils.interp1d_vector_wrt_scalar(t, u, kind="cubic")
x_sim             = scipy.integrate.odeint(compute_x_dot, x_0, t_sim)

figsize(9,4)
quadrotor_2d.draw(t_sim, x_sim, t_nominal=t, x_nominal=x_nominal, inline=True)

