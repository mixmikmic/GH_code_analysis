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

#path to nupycee
paper_path=s.global_path
# two factors are applied
f_extra_source=[1.0,0.5]

# yields come from these 2 tables (the same tables)
extra_source_table=2*[paper_path+'/yield_tables/extra_source.txt']


# in which initial mass range are the factors applied?
extra_source_mass_range = [[8,30],[22.5,30]]

# at which Z (Z from yield input tables) are the factors NOT applied.
extra_source_exclude_Z = [[0.01],[0.0001,0.001,0.006,0.02]] 

# Initial metallicity
iniZ = 0.0

# 10% of all massive stars at all Z have shell merger yields
o_NG_0_1 = o.omega(galaxy='milky_way',          special_timesteps=60, exp_ml=1.0, mass_frac_SSP=0.35, nb_1a_per_m=1.5e-3, DM_evolution=True, sfe=0.04,           t_sf_z_dep=0.3, mass_loading=1.02, iniZ=iniZ,
          extra_source_on=True, extra_source_table=extra_source_table,extra_source_mass_range=extra_source_mass_range,
           f_extra_source=f_extra_source,extra_source_exclude_Z=extra_source_exclude_Z)            



