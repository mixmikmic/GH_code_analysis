from __future__ import print_function
from classy import Class

p1 = {'tau_reio' : 0.06}
p2 = {'tau_reio' : 0.08}

# add in precision parameters if you want
precs = {'reionization_optical_depth_tol': 1e-06}
p1.update(precs)
p2.update(precs)

# compute cosmology with CLASS
c1, c2 = Class(), Class()
c1.set(p1)
c2.set(p2)
c1.compute()
c2.compute()

# get BAO-relevant quantities (not H_0 because we specify it)
print('rs', (c2.rs_drag() - c1.rs_drag())/c2.rs_drag() )

source_z = 0.5
print('dA', (c2.angular_distance(source_z)-c1.angular_distance(source_z)) /               c1.angular_distance(source_z))

p1 = {'z_reio' : 0.06}
p2 = {'z_reio' : 0.08}

# add in precision parameters if you want
precs = {'reionization_optical_depth_tol': 1e-06}
p1.update(precs)
p2.update(precs)

# compute cosmology with CLASS
c1, c2 = Class(), Class()
c1.set(p1)
c2.set(p2)
c1.compute()
c2.compute()

# get BAO-relevant quantities (not H_0 because we specify it)
print('rs', (c2.rs_drag() - c1.rs_drag())/c2.rs_drag() )

source_z = 0.5
print('dA', (c2.angular_distance(source_z)-c1.angular_distance(source_z)) /               c1.angular_distance(source_z))

p1 = {'cc_dmeff_p' : 0.0}
p2 = {'cc_dmeff_p' : 1e5}

precs = {'omega_dmeff':0.120,
        'omega_cdm':0.,
        'm_dmeff':1,
        'cc_dmeff_op':1,
        'spin_dmeff': 0.5,
        'cc_dmeff_n':0.,
        'use_temperature_dmeff':'yes',
        'use_helium_dmeff':'yes',
        'cc_dmeff_num':1,
        'cc_dmeff_qm2':0,
        'tight_coupling_trigger_tau_c_over_tau_k':0.,
        'tight_coupling_trigger_tau_c_over_tau_h':0.
        }

p1.update(precs)
p2.update(precs)

# compute cosmology with CLASS
c1, c2 = Class(), Class()
c1.set(p1)
c2.set(p2)
c1.compute()
c2.compute()

# get BAO-relevant quantities (not H_0 because we specify it)
print('rs', (c2.rs_drag() - c1.rs_drag())/c2.rs_drag() )

source_z = 0.5
print('dA', (c2.angular_distance(source_z)-c1.angular_distance(source_z)) /               c1.angular_distance(source_z))





