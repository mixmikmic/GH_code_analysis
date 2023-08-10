# Python packages
get_ipython().magic('matplotlib nbagg')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Shared class between SYGMA and OMEGA
import chem_evol
reload(chem_evol)

# One-zone galactic chemical evolution code
import omega
reload(omega)

# Run OMEGA simulations.
# Constant star formation rate, "cte_sfr", of 1 Msun/yr
# with different gas reservoir, "mgal", in units of solar mass.
o_res_1e10 = omega.omega(cte_sfr=1.0, mgal=1e10)
o_res_1e11 = omega.omega(cte_sfr=1.0, mgal=1e11)
o_res_1e12 = omega.omega(cte_sfr=1.0, mgal=1e12)

# Plot the total mass of the gas reservoir as a function of time.
get_ipython().magic('matplotlib nbagg')
o_res_1e12.plot_totmasses(color='r', label='M_res = 1e12 Msun')
o_res_1e11.plot_totmasses(color='g', label='M_res = 1e11 Msun')
o_res_1e10.plot_totmasses(color='b', label='M_res = 1e10 Msun')
plt.ylim(1e9, 2e12)

# Plot the iron concentration of the gas reservoir as a function of time
get_ipython().magic('matplotlib nbagg')
yaxis = '[Fe/H]'
o_res_1e12.plot_spectro(yaxis=yaxis, color='r', label='M_res = 1e12 Msun', shape='-.')
o_res_1e11.plot_spectro(yaxis=yaxis, color='g', label='M_res = 1e11 Msun', shape='--')
o_res_1e10.plot_spectro(yaxis=yaxis, color='b', label='M_res = 1e10 Msun', shape='-')
#plt.xscale('log')

# Plot the Si to Fe abundances of the gas reservoir as a function of time (you can try different elements).
get_ipython().magic('matplotlib nbagg')
xaxis = 'age'
yaxis = '[Si/Fe]'
o_res_1e12.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='r', label='mgal = 1e12 Msun', shape='-.')
o_res_1e11.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='g', label='mgal = 1e11 Msun', shape='--')
o_res_1e10.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='b', label='mgal = 1e10 Msun', shape='-')

# Plot the Si to Fe abundances of the gas reservoir as a function of [Fe/H]
get_ipython().magic('matplotlib nbagg')
yaxis = '[Si/Fe]'
xaxis = '[Fe/H]'
o_res_1e12.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='r', label='mgal = 1e12 Msun', shape='-.')
o_res_1e11.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='g', label='mgal = 1e11 Msun', shape='--')
o_res_1e10.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='b', label='mgal = 1e10 Msun', shape='-')
plt.ylim(-0.2, 2.5)

# Run OMEGA simulations
# Different numbers of SNe Ia per stellar mass, "nb_1a_per_m",
# formed in each simple stellar population (SYGMA). Here, we use
# the same gas reservoir to isolate the impact of "nb_1a_per_m".
o_res_1e10_low_Ia  = omega.omega(cte_sfr=1.0, mgal=1e10, nb_1a_per_m=1.0e-4)
o_res_1e10         = omega.omega(cte_sfr=1.0, mgal=1e10, nb_1a_per_m=1.0e-3)
o_res_1e10_high_Ia = omega.omega(cte_sfr=1.0, mgal=1e10, nb_1a_per_m=1.0e-2)

# Plot the iron concentration of the gas reservoir as a function of time
get_ipython().magic('matplotlib nbagg')
xaxis = '[Fe/H]'
yaxis = '[Si/Fe]'
o_res_1e10_low_Ia.plot_spectro( xaxis=xaxis, yaxis=yaxis, color='g', label='nb_1a_per_m = 1e-4', shape='-.')
o_res_1e10.plot_spectro(        xaxis=xaxis, yaxis=yaxis, color='r', label='nb_1a_per_m = 1e-3', shape='--')
o_res_1e10_high_Ia.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='b', label='nb_1a_per_m = 1e-2', shape='-')

# Plot the mass of Fe present inside the gas reservoir as a function of time.
get_ipython().magic('matplotlib nbagg')
specie = 'Fe'

# Contribution of SNe Ia.
o_res_1e10_low_Ia.plot_mass( specie=specie, color='g', label='SNe Ia, nb_1a_per_m = 1e-4', source='sn1a')
o_res_1e10.plot_mass(        specie=specie, color='r', label='SNe Ia, nb_1a_per_m = 1e-3', source='sn1a')
o_res_1e10_high_Ia.plot_mass(specie=specie, color='b', label='SNe Ia, nb_1a_per_m = 1e-2', source='sn1a')

# Contribution of massive (winds+SNe) and AGB stars.
o_res_1e10.plot_mass(specie=specie, color='k', label='Massive stars', source='massive', shape='-')
o_res_1e10.plot_mass(specie=specie, color='k', label='AGB stars',     source='agb',    shape='--')
# You can drop the 'source' argument to plot the sum the contribution of all stars.

# Plot the evolution of [Fe/H] as a function of time.
get_ipython().magic('matplotlib nbagg')
yaxis = '[Fe/H]'
o_res_1e10_low_Ia.plot_spectro( yaxis=yaxis, color='g', label='nb_1a_per_m = 1e-4', shape='-.')
o_res_1e10.plot_spectro(        yaxis=yaxis, color='r', label='nb_1a_per_m = 1e-3', shape='--')
o_res_1e10_high_Ia.plot_spectro(yaxis=yaxis, color='b', label='nb_1a_per_m = 1e-2', shape='-')
plt.xscale('log')

# OMEGA can receive an input SFH array with the "sfh_array" parameter.
# sfh_array[ number of input times ][ 0 --> time in yr; 1 --> star formation rate in Msun/yr ]

# Time array [Gyr]
t = [0.0, 6.5e9, 13.0e9]

# Build the decreasing star formation history array [Msun/yr]
sfr_dec = [7.0, 4.0, 1.0]
sfh_array_dec = []
for i in range(len(t)):
    sfh_array_dec.append([t[i], sfr_dec[i]])

# Build the increasing star formation history array [Msun/yr]
sfr_inc = [1.0, 4.0, 7.0]
sfh_array_inc = []
for i in range(len(t)):
    sfh_array_inc.append([t[i], sfr_inc[i]])

# Run OMEGA simulations.
# Different star formation histories within the same initial gas reservoir.
o_cte = omega.omega(mgal=5e11, special_timesteps=30, cte_sfr=4.0)
o_dec = omega.omega(mgal=5e11, special_timesteps=30, sfh_array=sfh_array_dec)
o_inc = omega.omega(mgal=5e11, special_timesteps=30, sfh_array=sfh_array_inc)

get_ipython().magic('matplotlib nbagg')
o_cte.plot_star_formation_rate(color='k', shape='-.')
o_dec.plot_star_formation_rate(color='m', shape='--')
o_inc.plot_star_formation_rate(color='c', shape='-')

# Calculate the cumulated stellar mass (integration of the SFH)
print 'Total stellar mass formed (not corrected for stellar mass loss)'
print '  Increasing SFH :', sum(o_inc.history.m_locked), 'Msun'
print '  Constant SFH   :', sum(o_cte.history.m_locked), 'Msun'
print '  Decreasing SFH :', sum(o_dec.history.m_locked), 'Msun'

# Re-run OMEGA simulations with the "sfh_array_norm" parameter.
o_cte = omega.omega(mgal=5e11, special_timesteps=30, cte_sfr=4.0,             sfh_array_norm=5.2e10)
o_dec = omega.omega(mgal=5e11, special_timesteps=30, sfh_array=sfh_array_dec, sfh_array_norm=5.2e10)
o_inc = omega.omega(mgal=5e11, special_timesteps=30, sfh_array=sfh_array_inc, sfh_array_norm=5.2e10)

get_ipython().magic('matplotlib nbagg')
o_cte.plot_star_formation_rate(color='k', shape='-.')
o_dec.plot_star_formation_rate(color='m', shape='--')
o_inc.plot_star_formation_rate(color='c', shape='-')

# Calculate the cumulated stellar mass (integration of the SFH)
print 'Total stellar mass formed (not corrected for stellar mass loss)'
print '  Increasing SFH :', sum(o_inc.history.m_locked), 'Msun'
print '  Constant SFH   :', sum(o_cte.history.m_locked), 'Msun'
print '  Decreasing SFH :', sum(o_dec.history.m_locked), 'Msun'

# Plot the mass of Fe present inside the gas reservoir as a function of time (you can try other elements).
get_ipython().magic('matplotlib nbagg')
specie = 'Fe'

# Increasing SFH
o_inc.plot_mass(specie=specie, color='c', source='massive')
#o_inc.plot_mass(specie=specie, color='c', source='sn1a')
o_inc.plot_mass(specie=specie, color='c', source='agb')

# Constant SFH
o_cte.plot_mass(specie=specie, color='k', source='massive')
#o_cte.plot_mass(specie=specie, color='k', source='sn1a')
o_cte.plot_mass(specie=specie, color='k', source='agb')

# Decreasing SFH
o_dec.plot_mass(specie=specie, color='m', source='massive')
#o_dec.plot_mass(specie=specie, color='m', source='sn1a')
o_dec.plot_mass(specie=specie, color='m', source='agb')

# Add legend directly on the plot
plt.annotate('Decreasing SFH', color='m', xy=(0.6, 0.30), xycoords='axes fraction', fontsize=13)
plt.annotate('Constant SFH',   color='k', xy=(0.6, 0.22), xycoords='axes fraction', fontsize=13)
plt.annotate('Increasing SFH', color='c', xy=(0.6, 0.14), xycoords='axes fraction', fontsize=13)

# Remove the default log scale of the x axis
plt.xscale('linear')

# Plot the evolution of [Fe/H]
get_ipython().magic('matplotlib nbagg')
o_inc.plot_spectro(color='c', shape='-',  label='Increasing SFH')
o_cte.plot_spectro(color='k', shape='-.', label='Constant SFH')
o_dec.plot_spectro(color='m', shape='--', label='Decreasing SFH')
plt.ylim(-6,0)

# Plot the evolution of [Si/Fe].
get_ipython().magic('matplotlib nbagg')
yaxis = '[Si/Fe]'
o_inc.plot_spectro(yaxis=yaxis, color='c', shape='-',  label='Increasing SFH')
o_cte.plot_spectro(yaxis=yaxis, color='k', shape='-.', label='Constant SFH')
o_dec.plot_spectro(yaxis=yaxis, color='m', shape='--', label='Decreasing SFH')

# Plot the predicted chemical evolution.
get_ipython().magic('matplotlib nbagg')
xaxis = '[Fe/H]'
yaxis = '[Si/Fe]'
o_inc.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='c', shape='-',  label='Increasing SFH')
o_cte.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='k', shape='-.', label='Constant SFH')
o_dec.plot_spectro(xaxis=xaxis, yaxis=yaxis, color='m', shape='--', label='Decreasing SFH')



