get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # This optional package makes plots prettier

import openmc

sp = openmc.StatePoint('statepoint.50.h5')

sp.date_and_time

sp.k_combined

plt.plot(sp.k_generation)

plt.plot(sp.entropy)

sp.tallies.keys()

sp.tallies[10000]

mesh_fiss = sp.get_tally(name='mesh fission')

flux = sp.get_tally(name='flux')
distribcell = sp.get_tally(name='distribcell')

df = mesh_fiss.get_pandas_dataframe()
df.head(10)  # Show the first 10 rows

df = flux.get_pandas_dataframe()
df.head(10)

df = distribcell.get_pandas_dataframe()
df.head(10)

df = mesh_fiss.get_pandas_dataframe()
indices = df['nuclide'] == 'U235'
indices.head(5)

sub_df = df[df['nuclide'] == 'U235']
sub_df.head(5)

# Replace 0's with NaN to eliminate them from average.
sub_df = sub_df.replace(0, np.nan)

# Extract rows corresponding to above-average fission rates.
indices = sub_df['mean'] > sub_df['mean'].mean()
above_avg = sub_df[indices]
above_avg.head(5)

indices = df[('mesh 10001', 'x')] > df[('mesh 10001', 'y')]
lower = df[indices]
lower.head(5)

df = flux.get_pandas_dataframe()
df.head(5)

# Extract the flux mean values array.
fluxes = df['mean'].values

# Extend the flux array for Matplotlib's step plot.
fluxes = np.insert(fluxes, 0, fluxes[0])

# Extract the energy bins from the Tally's EnergyFilter
energy_filter = flux.find_filter(openmc.EnergyFilter)
energies = energy_filter.bins

fig = plt.figure()
plt.loglog(energies, fluxes, drawstyle='steps', c='r')
plt.xlabel('Energy [eV]')
plt.ylabel('Flux')

df = mesh_fiss.get_pandas_dataframe()
df.head(5)

mean = df[df['nuclide'] == 'U235']['mean'].values
rel_err = df[df['nuclide'] == 'U235']['std. dev.'].values / mean

# Reshape the arrays.
mean.shape = (17, 17)
rel_err.shape = (17, 17)

# Transpose them to match the order expected by imshow.
mean = mean.T
rel_err = rel_err.T

# Plot the mean on the left.
fig = plt.subplot(121)
plt.imshow(mean, interpolation='none', cmap='jet')
plt.ylim(plt.ylim()[::-1])  # Invert the y-axis.
plt.title('Mean')
plt.grid(False)

# Plot the uncertainty on the right.
fig2 = plt.subplot(122)
plt.imshow(rel_err, interpolation='none', cmap='jet')
plt.ylim(plt.ylim()[::-1])  # Invert the y-axis.
plt.title('Rel. Unc.')
plt.grid(False)

# Assign a NaN to zero fission rates in guide tubes
# Matplotlib will ignore "bad" values in the colorbar
mean[mean == 0.] = np.nan
cmap = plt.get_cmap('jet')
cmap.set_bad(alpha=0.)

# Plot the mean on the left.
fig = plt.subplot(121)
plt.imshow(mean, interpolation='none', cmap='jet')
plt.ylim(plt.ylim()[::-1])  # Invert the y-axis.
plt.title('Mean')
plt.grid(False)

# Plot the uncertainty on the right.
fig2 = plt.subplot(122)
plt.imshow(rel_err, interpolation='none', cmap='jet')
plt.ylim(plt.ylim()[::-1])  # Invert the y-axis.
plt.title('Rel. Unc.')
plt.grid(False)

