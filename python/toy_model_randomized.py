get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pymc import MCMC
import seaborn
import numpy as np
from parmed.topologyobjects import DihedralType
from simtk import unit
from torsionfit.toy_model import ToyModel
import torsionfit.backends.sqlite_plus as db

toy = ToyModel(n_increments=18, sample_phase=True, rj=True)

# Initial and True values are randomized. RJ is on and phases are discrete (0, 180)
toy = ToyModel(n_increments=30, sample_phase=True, rj=True)
plt.plot(toy.synthetic_energy, color='black', label='True value')
plt.plot(toy.synthetic_energy, 'ko')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
plt.plot(toy.scan_set.mm_energy, color='r', label='initial_mm')
plt.plot(toy.scan_set.initial_mm, 'ko', color='r')
plt.legend()

# sample
sampler = MCMC(toy.model.pymc_parameters, db=db, dbname='discrete_phase_rj_randomized.sqlite')
sampler.sample(iter=10000)

seaborn.tsplot(sampler.trace('mm_energy')[:], interpolate=False)
plt.plot(toy.synthetic_energy, '.', color='r', label='True Value')
plt.title('MM Energy')
plt.xlabel('frames')
plt.ylabel('energy [kJ/mol]')
#plt.plot(sampler.trace('mm_energy')[0])
plt.plot(toy.scan_set.initial_mm, '.', color='g', label='Initial Value')
plt.legend()
plt.savefig('energy_fit_discrete_randomized.pdf')

import pymbar.timeseries as ts
[t, g, N_eff] = ts.detectEquilibration(sampler.trace('sigma')[:])
print('Start of equilibrated data: {}'.format(t))

plt.plot(np.exp(sampler.trace('sigma')[:]), label='sigma')
plt.axvline(t, color='red')
plt.text(t+100, 300000000, r'$t_{equil}$', color='red', fontsize=20)
plt.legend()
plt.xlabel('steps')
plt.ylabel('energy [kJ/mol]')
plt.savefig('sigma_discrete_randomized.pdf')

print('True value of model is :')
print(toy.true_value[0])
print(toy.true_value[1])
print(toy.true_value[2])

import torsionfit.plots as plots
plots.marg_mult(toy.model, sampler, 10000, t)
plt.savefig('multiplicity_discrete_random.pdf')

plots.trace_plots('CG331_CG321_CG321_CG331', sampler, markersize=2.0, filename='traces_discrete_rj_random.pdf')

alpha = 0.5
plt.hist(sampler.CG331_CG321_CG321_CG331_1_K.trace[t:], label='K_1', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_2_K.trace[t:], label='K_2', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_3_K.trace[t:], label='K_3', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_4_K.trace[t:], label='K_4', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_6_K.trace[t:], label='K_6', alpha=alpha)

plt.legend()
plt.title('Marginal distribution of force constants')
plt.xlabel('KCal/mol')
plt.ylabel('P(K)')

plt.subplot(1, 3, 1)
plt.hist(sampler.CG331_CG321_CG321_CG331_1_K.trace[t:], label='K_1')
plt.axvline(toy.true_value[1].phi_k, color='red')
plt.text(toy.true_value[1].phi_k+0.005, 1250, 'True Value', color='red', fontsize=20)
plt.title(r'$K_1$ histogram')
plt.xlabel('K (Kcal/mol)')
plt.subplot(1, 3, 2)
plt.hist(sampler.CG331_CG321_CG321_CG331_4_K.trace[t:], label='K_4')
plt.axvline(toy.true_value[2].phi_k, color='red')
plt.text(toy.true_value[2].phi_k+0.005, 1500, 'True Value', color='red', fontsize=20)
plt.xlabel('K (Kcal/mol)')
plt.title(r'$K_2$ histogram')
plt.subplot(1, 3, 3)
plt.hist(sampler.CG331_CG321_CG321_CG331_6_K.trace[t:], label='K_6')
plt.axvline(toy.true_value[0].phi_k, color='red')
plt.text(toy.true_value[0].phi_k+0.005, 1250, 'True Value', color='red', fontsize=20)
plt.title(r'$K_3$ histogram')
plt.xlabel('K (Kcal/mol)')
plt.savefig('hist_randomized.pdf')

alpha = 0.5
plt.hist(sampler.CG331_CG321_CG321_CG331_1_Phase.trace[t:], label='K_1', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_3_Phase.trace[t:], label='K_3', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_4_Phase.trace[t:], label='K_4', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_6_Phase.trace[t:], label='K_6', alpha=alpha)
plt.hist(sampler.CG331_CG321_CG321_CG331_2_Phase.trace[t:], label='K_2')


plt.legend()
plt.title('Marginal distribution of Phase angle')
plt.xlabel('phase angle')
plt.ylabel('P(angle)')



