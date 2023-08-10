import MDAnalysis as mda
from MDAnalysis.analysis.hole import HOLEtraj
from MDAnalysis.analysis.rms import RMSD

from MDAnalysis.tests.datafiles import PDB_HOLE, MULTIPDB_HOLE
   
mda.start_logging()

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

ref = mda.Universe(PDB_HOLE)    # reference structure (needs MDAnalysis 0.17.1)
u = mda.Universe(MULTIPDB_HOLE) # trajectory

# calculate RMSD
R = RMSD(u, reference=ref, select="protein", weights="mass")
R.run()

frame, _, rho = R.rmsd.transpose()

ax = plt.subplot(111)
ax.plot(frame, rho, '.-')
ax.set_xlabel("frame")
ax.set_ylabel(r"RMSD $\rho$ ($\AA$)");

# HOLE analysis with order parameters
H = HOLEtraj(u, orderparameters=R.rmsd[:,2], 
             executable="~/hole2/exe/hole")
H.run()

H.plot3D(rmax=2.5)

ax = H.plot()
ax.set_ylim(1, 3)
ax.set_xlim(-15, 35)
plt.tight_layout()

r_rho = np.array([[rho, profile.radius.min()] for rho, profile in H])

ax = plt.subplot(111)
ax.plot(r_rho[:, 0], r_rho[:, 1], lw=2)
ax.set_xlabel(r"order parameter RMSD $\rho$ ($\AA$)")
ax.set_ylabel(r"minimum HOLE pore radius $r$ ($\AA$)");

zeta0_rho = np.array([[rho, profile.rxncoord[np.argmin(profile.radius)]] 
                      for rho, profile in H])

ax = plt.subplot(111)
ax.plot(zeta0_rho[:, 0], np.abs(zeta0_rho[:, 1]), lw=2)
ax.set_xlabel(r"order parameter RMSD $\rho$ ($\AA$)")
ax.set_ylabel(r"position of constriction $|\zeta_0|$ ($\AA$)");



