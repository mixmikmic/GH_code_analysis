get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import os
from fnmatch import fnmatch
import numpy as np
import torsionfit.TorsionScanSet as Scan

# Load MP2 torsion scan psi4 output files
scan = Scan.parse_psi4_out('MP2_torsion_scan/', '../structure/butane.psf')

# Load parameter files and calculate the MM energies
param = Scan.CharmmParameterSet('../../data/charmm_ff/par_all36_cgenff.prm', '../../data/charmm_ff/top_all36_cgenff.rtf')
scan.compute_energy(param)

scan.to_dataframe()

# Remove structures that didn't converge
optimized = scan.remove_nonoptimized()

optimized.to_dataframe()

plt.plot(scan.steps, scan.qm_energy, '.',  label='QM (MP2) Energy')
plt.plot(scan.steps, scan.mm_energy, '.',  label='MM Energy')
plt.plot(scan.steps[1], scan.qm_energy[1]._value, 'r.', label='Not converged')


plt.legend(prop={'size': 7}, loc='upper center')
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
plt.title('Butane Torsion Scan')
plt.savefig('butane_mp2.png')

optimized.compute_energy(param)
plt.plot(optimized.steps, optimized.qm_energy, '.')
plt.plot(optimized.steps, optimized.mm_energy, '.')

# Load B3LYP scan
scan3 = Scan.parse_psi4_out('B3LYP_torsion_scan/', '../structure/butane.psf')

scan3.compute_energy(param)
scan3.to_dataframe()

plt.plot(scan3.steps, scan3.mm_energy, '.', label='CHARMM')
plt.plot(scan3.steps, scan3.qm_energy, '.', label='B3LYP')
unconverged = []
for i, j in enumerate(scan3.optimized):
    if not j:
        unconverged.append((scan3.steps[i], scan3.qm_energy[i], scan3.mm_energy[i]))
for i in unconverged:
    plt.plot(i[0], i[1]._value, '.', color='red')
    plt.plot(i[0], i[2]._value, '.', color='red')
plt.plot(unconverged[0][0], unconverged[0][1]._value, '.', color='red', label='Not Converged')
plt.legend(prop={'size': 7}, loc='upper center')
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
plt.title('Butane B3LYP/cc-PVDZ torsion scan')
plt.savefig('butane_b3lyp.png')

scan3_optimized = scan3.remove_nonoptimized()

scan3_optimized.compute_energy(param)
plt.plot(scan3_optimized.steps, scan3_optimized.qm_energy, '.')
plt.plot(scan3_optimized.steps, scan3_optimized.mm_energy, '.')

plt.plot(scan.steps, scan.qm_energy, '.',  label='MP2')
plt.plot(scan.steps, scan.mm_energy, '.',  label='MM MP2')
#plt.plot(scan.steps[1], scan.qm_energy[1]._value, 'r.', label='Not converged')
#plt.plot(scan.steps[2], scan.qm_energy[2]._value, 'r.', label='Not converged')
#plt.plot(scan.steps[-1], scan.qm_energy[-1]._value, 'r.', label='Not converged')
plt.plot(scan3.steps, scan3.mm_energy, '.', label='MM B3LYP')
plt.plot(scan3.steps, scan3.qm_energy, '.', label='B3LYP')
unconverged = []
for i, j in enumerate(scan3.optimized):
    if not j:
        unconverged.append((scan3.steps[i], scan3.qm_energy[i], scan3.mm_energy[i]))
for i in unconverged:
    plt.plot(i[0], i[1]._value, '.', color='red')
    plt.plot(i[0], i[2]._value, '.', color='red')
#plt.plot(unconverged[0][0], unconverged[0][1]._value, '.', color='red', label='Not Converged')
plt.legend(prop={'size':7}, loc='upper center')
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
#plt.title('Butane B3LYP/cc-PVDZ torsion scan')
#plt.savefig('butane_b3lyp.png')

#plt.legend()
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
plt.title('Butane Torsion Scan')
plt.savefig('butane_qm_comparison.png')



