get_ipython().magic('pylab inline')
from pylab import *

import mpl_toolkits.mplot3d

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.spline_utils as spline_utils

T_y = matrix([0,1,2,3]).T.A
T_x = matrix([0,1,2,3]).T.A
T   = c_[T_y, T_x]
P_y = matrix([0,4,7,9]).T.A
P_x = matrix([0,9,1,4]).T.A
P   = c_[P_y, P_x]

print "T = "; print T; print; print "P = "; print P

C, T, sd =     spline_utils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(
        P, T, degree=7, lamb=[0,0,0,1,0])
    
P_eval, T_eval, dT =     spline_utils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(
        C, T, sd, num_samples=100)

t = T_eval[:,0]

figsize(4,4);
scatter(P_eval[:,1], P_eval[:,0], c=t, s=50);
scatter(P[:,1], P[:,0], c=T[:,0], s=200, alpha=0.5);
gca().set_aspect("equal")
title("$\\mathbf{p}(t)$\n", fontsize=20);
ylabel("$\\mathbf{p}_y$", rotation="horizontal", fontsize=20); xlabel("$\\mathbf{p}_x$", fontsize=20);

T_z = matrix([0,1,2,3]).T.A
T_y = matrix([0,1,2,3]).T.A
T_x = matrix([0,1,2,3]).T.A
T   = c_[T_z, T_y, T_x] 
P_z = matrix([0,0,1,2]).T.A
P_y = matrix([0,4,7,9]).T.A
P_x = matrix([0,9,1,4]).T.A
P   = c_[P_z, P_y, P_x]

C, T, sd =     spline_utils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(
        P, T, degree=7, lamb=[0,0,0,1,0])
    
P_eval, T_eval, dT =     spline_utils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(
        C, T, sd,num_samples=100)

figsize(6,6);
fig = plt.figure(); ax = fig.add_subplot(111, projection="3d");
ax.scatter(P_eval[:,2], P_eval[:,0], P_eval[:,1], c=t, s=50);
ax.scatter(P[:,2], P[:,0], P[:,1], c=T[:,0], s=200, alpha=0.5);
title("$\\mathbf{p}(t)$\n", fontsize=20);
ax.set_zlabel("$\\mathbf{p}_y$", fontsize=20);
xlabel("$\\mathbf{p}_x$", fontsize=20); ylabel("$\\mathbf{p}_z$", fontsize=20);

