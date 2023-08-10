get_ipython().magic('matplotlib nbagg')
import matplotlib.pyplot as plt
import sys
import sygma as s
reload(s)
import omega as o
reload(o)
import numpy as np
print s.global_path
#%matplotlib inline
import stellab
import read_yields as ry
import matplotlib

table='yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt'

# Initial metallicity 0
#includes pop3_table='yield_tables/popIII_heger10.txt',
iniZ = 0.0

# Original yields
o_NG = o.omega(galaxy='milky_way', table=table,          special_timesteps=60, exp_ml=1.0, mass_frac_SSP=0.35, nb_1a_per_m=1.5e-3, DM_evolution=True, sfe=0.04,           t_sf_z_dep=0.3, mass_loading=1.02, iniZ=iniZ)

# Initial metallicity 0
#includes pop3_table='yield_tables/popIII_heger10.txt',
iniZ = 0.0

#turn on net yield capability
yield_interp='wiersma'

#yield input not net yields
netyields_on=False

#should not matter
wiersmamod=False

Z_trans=1e-20

# Original yields
o_NG_net = o.omega(galaxy='milky_way', table=table,          special_timesteps=60, exp_ml=1.0, mass_frac_SSP=0.35, nb_1a_per_m=1.5e-3, DM_evolution=True, sfe=0.04,           t_sf_z_dep=0.3, mass_loading=1.02, iniZ=iniZ,yield_interp=yield_interp,netyields_on=netyields_on)



