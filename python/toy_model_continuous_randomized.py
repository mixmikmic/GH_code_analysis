get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pymc import MCMC
import seaborn
import numpy as np
from parmed.topologyobjects import DihedralType
from simtk import unit
from torsionfit.toy_model import ToyModel
import torsionfit.backends.sqlite_plus as db

toy = ToyModel(n_increments=18, sample_phase=True, continuous=True)

plt.plot(toy.synthetic_energy, label='True value')
plt.plot(toy.synthetic_energy, 'ko')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
plt.plot(toy.scan_set.mm_energy, color='g', label='initial_mm')
plt.plot(toy.scan_set.initial_mm, 'ko', color='r')
plt.legend()

# sample
sampler = MCMC(toy.model.pymc_parameters, db=db, dbname='continuous_rj_randomized.sqlite')
sampler.sample(iter=10000)

seaborn.tsplot(sampler.trace('mm_energy')[:], interpolate=False)
plt.plot(toy.synthetic_energy, '.', color='r', label='True Value')
plt.title('MM Energy')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
#plt.plot(sampler.trace('mm_energy')[0])
plt.plot(toy.scan_set.initial_mm, '.', color='g', label='Initial Value')
plt.legend()
plt.savefig('energy_fit_cont_rj_randomized.pdf')

import pymbar.timeseries as ts
[t, g, N_eff] = ts.detectEquilibration(sampler.trace('sigma')[:])
print('Start of equilibrated data: {}'.format(t))

plt.plot(np.exp(sampler.trace('sigma')[:]), label='sigma')
plt.axvline(t, color='red', ymin=0)
plt.text(t+200,300000000,r'$t_{equil}$', rotation=0, color='red', fontsize=20)
plt.legend()
plt.xlabel('steps')
plt.ylabel('energy [kJ/mol]')
plt.savefig('sigma_continous_rj_randomized.pdf')

import torsionfit.plots as plots
plots.marg_mult(toy.model, sampler, 10000, t)
plt.savefig('multiplicity_continous_rj_randomized.pdf')

toy.true_value

plots.trace_plots('CG331_CG321_CG321_CG331', sampler, markersize=2.0, continuous=True, 
                  filename='traces_cont_rj_randomized.pdf')

toy.initial_value



