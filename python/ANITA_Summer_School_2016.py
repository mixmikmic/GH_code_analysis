import matplotlib.pyplot as plt
import sygma
import omega
import stellab

#loading the observational data module STELLAB
stellab = stellab.stellab()

# OMEGA parameters for MW
mass_loading = 1     # How much mass is ejected from the galaxy per stellar mass formed
nb_1a_per_m = 3.0e-3  # Number of SNe Ia per stellar mass formed
sfe = 0.005           # Star formation efficiency, which sets the mass of gas
table = 'yield_tables/isotope_yield_table_MESA_only_ye.txt' # Yields for AGB and massive stars
#milky_way

o_mw = omega.omega(galaxy='milky_way',Z_trans=-1, table=table,sfe=sfe, DM_evolution=True,                  mass_loading=mass_loading, nb_1a_per_m=nb_1a_per_m, special_timesteps=60)

# Choose abundance ratios
get_ipython().magic('matplotlib nbagg')
xaxis = '[Fe/H]'
yaxis = '[C/Fe]'

# Plot observational data points (Stellab)
stellab.plot_spectro(xaxis=xaxis, yaxis=yaxis,norm='Grevesse_Noels_1993',galaxy='milky_way',show_err=False)

# Extract the numerical predictions (OMEGA)
xy_f = o_mw.plot_spectro(fig=3,xaxis=xaxis,yaxis=yaxis,return_x_y=True)

# Overplot the numerical predictions (they are normalized according to Grevesse & Noels 1993)
plt.plot(xy_f[0],xy_f[1],linewidth=4,color='w')
plt.plot(xy_f[0],xy_f[1],linewidth=2,color='k',label='OMEGA')

# Update the existing legend
plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), markerscale=0.8, fontsize=13)

# Choose X and Y limits
plt.xlim(-4.5,0.5)
plt.ylim(-1.4,1.6)

s0p0001=sygma.sygma(iniZ=0.0001)
s0p006=sygma.sygma(iniZ=0.006)

elem='[C/Fe]'
s0p0001.plot_spectro(fig=3,yaxis=elem,marker='D',color='b',label='Z=0.0001')
s0p006.plot_spectro(fig=3,yaxis=elem,label='Z=0.006')

# Plot the ejected mass of a certain element
elem='C'
s0p0001.plot_mass(fig=4,specie=elem,marker='D',color='b',label='Z=0.0001')
s0p006.plot_mass(fig=4,specie=elem,label='Z=0.006')

elem='C'
s0p0001.plot_mass_range_contributions(specie=elem,marker='D',color='b',label='Z=0.0001')
s0p006.plot_mass_range_contributions(specie=elem,label='Z=0.006')

s0p0001.plot_yield_input(fig=6,iniZ=0.0001,table='yield_tables/isotope_yield_table.txt',yaxis='C-12',
                         masses=[1.0, 1.65, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],marker='D',color='b',)
s0p006.plot_yield_input(fig=6,iniZ=0.006,table='yield_tables/isotope_yield_table.txt',yaxis='C-12',
                         masses=[1.0, 1.65, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

import nugridse as mp
import mesa as ms
import matplotlib.pyplot as plt

s=ms.star_log(mass=2,Z=0.0001)

s.kip_cont(ifig=111,ixaxis='model_number')

sefiles=mp.se(mass=2.0,Z=0.0001,output='surf')

cycles=sefiles.se.cycles
cycles1=[cycles[k] for k in range(1,53000,1000)]
c12=sefiles.get(cycles1,'C-12')

plt.figure(33);plt.plot(cycles1,c12)
plt.xlabel('')

