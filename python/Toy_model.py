get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pymc import MCMC
import seaborn
import numpy as np
from parmed.topologyobjects import DihedralType
from simtk import unit
from torsionfit.toy_model import ToyModel
import torsionfit.backends.sqlite_plus as db

toy = ToyModel(true_value=DihedralType(10, 1, 0, 1, 1), initial_value=DihedralType(10, 2, 180, 1, 1), 
               n_increments=18, sample_phase=False)
plt.plot(toy.synthetic_energy, label='True value')
plt.plot(toy.synthetic_energy, 'ko')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
plt.plot(toy.scan_set.mm_energy, color='g', label='initial_mm')
plt.plot(toy.scan_set.initial_mm, 'ko', color='r')
plt.legend()

# sample
sampler = MCMC(toy.model.pymc_parameters, db=db, dbname='eliminate_rj.sqlite')
sampler.sample(iter=10000)

print('true value')
print(toy.true_value)
print('initial value')
print(toy.initial_value)
print(toy.model.pymc_parameters['CG331_CG321_CG321_CG331_1_K'].value)

seaborn.tsplot(sampler.trace('mm_energy')[:], interpolate=False)
plt.plot(toy.synthetic_energy, '.', color='r', label='True Value')
plt.title('MM Energy')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
#plt.plot(sampler.trace('mm_energy')[0])
plt.plot(toy.scan_set.initial_mm, '.', color='g', label='Initial Value')
plt.legend()
plt.savefig('energy_fit_elim_rj.pdf')

import pymbar.timeseries as ts
[t, g, N_eff] = ts.detectEquilibration(sampler.trace('sigma')[:])
print('Start of equilibrated data: {}'.format(t))

plt.plot(np.exp(sampler.trace('sigma')[:]), label='sigma')
plt.axvline(t, color='red')
plt.text(t+100,100000, r'$t_{equil}$', color='red', fontsize=20)
plt.legend()
plt.xlabel('steps')
plt.ylabel('energy [kJ/mol]')
plt.savefig('sigma_elim_rj.pdf')

import torsionfit.plots as plots
plots.marg_mult(toy.model, sampler, 10000, 0)
plt.savefig('multiplicity_elim_rj.pdf')

plt.subplot(2, 1, 1)
plt.title(r'Trace and histogram of $K_1$')
plt.plot(sampler.CG331_CG321_CG321_CG331_1_K.trace[t+20:])
plt.xlabel('samples')
plt.ylabel('K (Energy Kcal/mol)')
plt.subplot(2, 1, 2)
plt.hist(sampler.CG331_CG321_CG321_CG331_1_K.trace[t+20:], label='K_1')
plt.axvline(toy.true_value[0].phi_k, color='red')
plt.text(toy.true_value[0].phi_k+0.005, 1500, 'True value', color='red', fontsize=20)
plt.xlabel('Energy (Kcal/mol)')
plt.ylabel('P(K_1)')
plt.savefig('trace_hist_label_on.pdf')

mean = sampler.CG331_CG321_CG321_CG331_1_K.trace[t:].mean()
var = np.var(sampler.CG331_CG321_CG321_CG331_1_K.trace[t:])
print('Mean of K_1: {}'.format(mean))
print('Variance of K_1: {}'.format(var))

plots.trace_no_phase('CG331_CG321_CG321_CG331', sampler, markersize=2.0, filename='traces_elim_rj.pdf')

alpha = 0.5
plt.subplot(2, 1, 1)
plt.title('Traces and histogram of force constants')
plt.plot(sampler.CG331_CG321_CG321_CG331_2_K.trace[t:], label='K_2', alpha=alpha)
plt.plot(sampler.CG331_CG321_CG321_CG331_3_K.trace[t:], label='K_3', alpha=alpha)
plt.plot(sampler.CG331_CG321_CG321_CG331_4_K.trace[t:], label='K_4', alpha=alpha)
plt.plot(sampler.CG331_CG321_CG321_CG331_6_K.trace[t:], label='K_6', alpha=alpha)
plt.plot(sampler.CG331_CG321_CG321_CG331_1_K.trace[t:], label='K_1', color='black')
plt.xlabel('Samples')
plt.ylabel('K (Kcal/mol)')


lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))


plt.subplot(2, 1, 2)
plt.hist(sampler.CG331_CG321_CG321_CG331_2_K.trace[t:], label='K_2', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_3_K.trace[t:], label='K_3', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_4_K.trace[t:], label='K_4', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_6_K.trace[t:], label='K_6', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_1_K.trace[t:], label='K_1', color='black')

lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))


plt.xlabel('KCal/mol')
plt.ylabel('P(K)')
plt.savefig('trace_hist.pdf')





