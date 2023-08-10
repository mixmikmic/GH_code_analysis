get_ipython().magic('pylab inline')
from pylab import *

import path_utils
path_utils.add_relative_to_current_source_file_path_to_sys_path("../../lib")

import flashlight.gradient_utils as gradient_utils
import flashlight.spline_utils   as spline_utils

T = matrix([0,1,4,5]).T.A
P = matrix([0,9,1,4]).T.A

print "T = "; print T; print; print "P = "; print P

C, T, sd =     spline_utils.compute_minimum_variation_nonlocal_interpolating_b_spline_coefficients(
        P, T, degree=7, lamb=[0,0,0,1,0])

P_eval, T_eval, dT =     spline_utils.evaluate_minimum_variation_nonlocal_interpolating_b_spline(
        C, T, sd, num_samples=500)

t = T_eval[:,0]

figsize(5,2);
plot(t, P_eval[:,0]); scatter(T[:,0], P[:,0]);
title("$p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

dpdtN = gradient_utils.gradients_scalar_wrt_scalar_smooth_boundaries(
    P_eval[:,0], dT[0], max_gradient=5, poly_deg=5)

figsize(14,6)

subplot(231); plot(t, dpdtN[0,:]); scatter(T[:,0], P[:,0]); xlims=xlim();
title("$p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(232); plot(t, dpdtN[1,:]); xlim(xlims);
title("$\\frac{d}{dt}p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(233); plot(t, dpdtN[2,:]); xlim(xlims);
title("$\\frac{d^2}{dt^2}p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(234); plot(t, dpdtN[3,:]); xlim(xlims);
title("$\\frac{d^3}{dt^3}p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(235); plot(t, dpdtN[4,:]); xlim(xlims);
title("$\\frac{d^4}{dt^4}p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

subplot(236); plot(t, dpdtN[5,:]); xlim(xlims);
title("$\\frac{d^5}{dt^5}p(t)$\n", fontsize=20); xlabel("$t$", fontsize=20);

gcf().tight_layout();

