get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import simtk.openmm as mm
from torsionfit.database import qmdatabase as ScanSet
import torsionfit.TorsionFitModel as Model
from torsionfit import sqlite_plus
from pymc import MCMC
from parmed.charmm import CharmmParameterSet
from torsionfit import plots
from matplotlib.backends.backend_pdf import PdfPages

param_to_opt=[('CG331', 'CG321', 'CG321', 'CG331')]
param = CharmmParameterSet('../../../../data/charmm_ff/top_all36_cgenff.rtf', 
                           '../../../../data/charmm_ff/par_all36_cgenff.prm')
structure = '../../../structure/butane.psf'
scan = '../../../torsion_scans/MP2_torsion_scan/'
# Print initial guess
print param.dihedral_types[param_to_opt[0]]

# Create a butane scan with torsions on (to compare for difference potential)
butane_scan = ScanSet.parse_psi4_out(scan, structure)
butane_scan.compute_energy(param)
optimized = butane_scan.remove_nonoptimized()
optimized.compute_energy(param)

# Turn off torsion
param.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][1].phi_k=0
param.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][0].phi_k=0

# Create butane scan with torsions off
optimized_0 = butane_scan.remove_nonoptimized()
optimized_0.compute_energy(param)

plt.plot(optimized.steps, optimized.qm_energy, 'k.', label='QM')
plt.plot(optimized.steps, optimized.mm_energy, '.', label='MM')
plt.xlabel('Dihedral Angles (Degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
plt.legend()

plt.plot(optimized_0.steps, optimized_0.delta_energy-min(optimized_0.delta_energy), label='Torsion off')
plt.plot(optimized.steps, optimized.delta_energy-min(optimized.delta_energy), label='Torsion on')
plt.legend()
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Relative Energy (KJ/mol)')
plt.title('Butane Difference potential')

platform = mm.Platform.getPlatformByName('Reference')
model = Model.TorsionFitModelEliminatePhase(param, optimized_0, platform=platform, 
                                            param_to_opt=param_to_opt, sample_n5=True)
sampler = MCMC(model.pymc_parameters, db=sqlite_plus, dbname='butane_n5_init_0.db', verbose=5)

db = sqlite_plus.load('butane_n5_init_0.db')

sampler.sample(100000)

import seaborn
plt.plot(optimized.qm_energy, 'ro', label='QM energy')
plt.plot(optimized.qm_energy, color='r')
plt.plot(optimized.mm_energy, 'b.', label='Initial energy')
plt.plot(sampler.trace('mm_energy')[-1], 'k.', label='MM fit')

seaborn.tsplot(sampler.trace('mm_energy')[:])
plt.legend()
plt.xlabel('Dihedral Angle')
plt.ylabel('Relative Energy (KJ/mol)')
plt.title('Butane Energy Fit')
plt.savefig('sample_n5/Energy_fit_n5.pdf')

# reparameterize 
param_new = CharmmParameterSet('../data/charmm_ff/top_all36_cgenff.rtf', '../data/charmm_ff/par_all36_cgenff.prm')
# Turn off torsion
param_new.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][1].phi_k=0
param_new.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][0].phi_k=0

torsion_0 = optimized_0.remove_nonoptimized()
torsion_0.compute_energy(param_new)
plt.plot(optimized_0.delta_energy, label='Fit')
plt.plot(torsion_0.delta_energy, label='CHARMM')
plt.plot(optimized.delta_energy, label='CHARMM torsion')
plt.title('Difference potential (QM-MM)')
plt.xlabel('Dihedral Angle')
plt.ylabel('Energy (KJ/mol)')
plt.legend()
plt.savefig('sample_n5/diff_pot.png')
plt.savefig('sample_n5//diff_pot.pdf')

samples = 100000
burn = 0
multiplicities = (1,2,3,4,5,6)
import numpy as np
mult_bitstring = []
for i in model.pymc_parameters.keys():
    if i.split('_')[-1] == 'bitstring':
        mult_bitstring.append(i)

histogram = np.zeros((len(mult_bitstring), samples, 6))

for m, torsion in enumerate(mult_bitstring):
    for i, j in enumerate(db.trace('%s' % torsion)[burn:]):
        for k, l in enumerate(multiplicities):
            if 2**(l-1) & int(j):
                histogram[m][i][k] = 1

plt.matshow(histogram.sum(1), cmap='jet',  extent=[0, 5, 0, 20]), plt.colorbar()
plt.yticks([])
plt.xlabel('multiplicity term')
plt.ylabel('torsion')
#plt.savefig(filename)

plt.plot(np.exp(db.log_sigma[:]))

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_1_K')[40000:], bins=100)
plt.title('K_1')

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_2_K')[40000:], bins=50)
plt.title('K_2')

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_3_K')[40000:], bins=100)
plt.title('K_3')

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_4_K')[:], bins=100)
plt.title('K_4')

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_5_K')[:])
plt.title('K_5')

plt.plot(sampler.CG331_CG321_CG321_CG331_3_K.trace[40000:])

plt.hist(sampler.trace('CG331_CG321_CG321_CG331_6_K')[:], bins=100)
plt.title('K_6')

alpha=0.5
plt.subplot(2, 1, 1)
plt.plot(db.trace('CG331_CG321_CG321_CG331_1_K')[3000:], alpha=alpha, label='K_1')
plt.plot(db.trace('CG331_CG321_CG321_CG331_2_K')[3000:], alpha=alpha, label='K_2')
plt.plot(db.trace('CG331_CG321_CG321_CG331_3_K')[3000:], alpha=alpha, label='K_3')
#plt.plot(db.trace('CG331_CG321_CG321_CG331_4_K')[4000:], alpha=alpha, label='K_4')
#plt.plot(db.trace('CG331_CG321_CG321_CG331_6_K')[:], alpha=0.4)
lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))



plt.subplot(2, 1, 2)
plt.hist(db.trace('CG331_CG321_CG321_CG331_1_K')[3000:], alpha=alpha, label='K_1')
plt.hist(db.trace('CG331_CG321_CG321_CG331_2_K')[3000:], alpha=alpha, label='K_2')
plt.hist(db.trace('CG331_CG321_CG321_CG331_3_K')[3000:], alpha=alpha, label='K_3')
#plt.hist(db.trace('CG331_CG321_CG321_CG331_4_K')[3000:], alpha=alpha, label='K_4')
lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('trace_hist.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

#plt.plot(sampler.trace('mm_energy')[-1])
#plt.plot(optimized_0.mm_energy)
plt.plot(optimized_0.steps, optimized_0.delta_energy, label='After fit')
plt.plot(optimized.steps, optimized.delta_energy, label='Before fit')

plt.legend()

plt.plot(sampler.BUTA1_offset.trace[4000:])

plots.trace_no_phase_n5('CG331_CG321_CG321_CG331', sampler, equil=False, markersize=1.0, filename='trace_n5_init0.pdf')

plots.get_multiplicity_traces

plt.plot(db.deviance[100:])
plt.xlabel('samples')
plt.title('Deviance')
plt.savefig('deviance.pdf')



