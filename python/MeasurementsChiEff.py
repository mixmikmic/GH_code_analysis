get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("ticks")
sns.set_palette("colorblind")
sns.set_context("notebook")

chi_eff_constraints = np.genfromtxt('../data/chi_eff_q_measurements.csv', delimiter=',', names=True)

plt.scatter(chi_eff_constraints['q_sim'], chi_eff_constraints['chi_eff_sim'], c=chi_eff_constraints['chi_eff_90cl'],
            alpha=0.3, cmap='viridis_r', lw=0)

plt.xlabel('$q_\mathrm{sim}$')
plt.ylabel('$\chi_\mathrm{eff,sim}$')
cbar = plt.colorbar()
cbar.set_label('width of 90% credible level of $\chi_\mathrm{sim}}$')
cbar.set_alpha(1)
cbar.draw_all()

plt.savefig('../paper/plots/chi-eff-90cl.pdf', dpi=500, bbox_inches='tight')

