# Load Data and relevant modules
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.signal as sig
from scipy import interpolate
import matplotlib.pyplot as plt
import data_load
import gsw
import oceans as oc
import cmocean
import pandas as pd
import internal_waves_calculations as iwc
import warnings
import seaborn as sns
import ray_tracing as rt

# Kill warnings (they look ugly but use them while testing new code)
warnings.simplefilter("ignore")

# Allow display of pandas data tables

pd.options.display.max_columns = 22


# Load Data
ladcp, ctd = data_load.load_data()
strain = np.genfromtxt('strain.csv', delimiter=',')
wl_max=1200
wl_min=500
ctd_bin_size=1500
ladcp_bin_size=1500
nfft = 2048
rho0 = 1025

# Get Wave parameters using the methods above
PE, KE, omega, m, kh, lambdaH, Etotal,    khi, Uprime, Vprime, b_prime, ctd_bins,    ladcp_bins, KE_grid, PE_grid, ke_peaks,    pe_peaks, dist, depths, KE_psd, eta_psd, N2, N2mean = iwc.wave_components_with_strain(ctd,    ladcp, strain, wl_min=wl_min, wl_max=wl_max, plots=False)


m_plot = np.array([(1)/wl_max,
                   (1)/wl_max, (1)/wl_min,
                   (1)/wl_min])
plt.figure(figsize=[12,6])
plt.loglog(KE_grid, KE_psd.T, linewidth=.6, c='b', alpha=.1)
plt.loglog(KE_grid, np.nanmean(KE_psd, axis=0).T, lw=1.5, c='k')
ylims = plt.gca().get_ylim()
ylim1 = np.array([ylims[0], ylims[1]])
plt.plot(m_plot[2:], ylim1, lw=1.5,
         c='k', alpha=.9,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1.5,
         c='k', alpha=.9,
         linestyle='dotted')
plt.ylim(ylims)
plt.ylabel('Kinetic Energy Density ($J/m^{3}$)')
plt.xlabel('Vertical Wavenumber')
plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)

plt.loglog(PE_grid, .5*np.nanmean(N2)*eta_psd.T,
           lw=.6, c='r', alpha=.1)
plt.loglog(KE_grid, .5*np.nanmean(N2)*np.nanmean(eta_psd, axis=0).T,
           lw=1.5, c='k')
plt.plot(m_plot[2:], ylim1, lw=1.5,
         c='k', alpha=.9,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1.5,
         c='k', alpha=.9,
         linestyle='dotted')
plt.ylim(ylims)
plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)
plt.ylabel('Energy Density ($J/m^{3}$)')
plt.xlabel('Inverse wavelength :$1/\lambda$')
plt.xlim(10**(-3.5), 10**(-1.1))
plt.title('Kinetic and Potential Energy Density')

# Horizontal wave vector decomposition
k = []
l = []

theta = []
dz = 8

for i in ladcp_bins:
    theta.append(iwc.horizontal_azimuth(Uprime[i,:], Vprime[i,:], dz,                                        wl_min=wl_min,
                                        wl_max=wl_max,
                                        nfft=1024))
theta = np.vstack(theta)

k = kh*np.cos(theta)
l = kh*np.sin(theta)

display_table = pd.DataFrame(kh, index=np.squeeze(depths), columns=np.arange(1,22))
cmap = cmap=sns.diverging_palette(250, 5, as_cmap=True)
display_table.style.background_gradient(cmap=cmap, axis=1)    .set_properties(**{'max-width': '300px', 'font-size': '12pt'})    .set_caption("Horizontal Wavenumber")    .set_precision(3)

display_table = pd.DataFrame(k, index=np.squeeze(depths), columns=np.arange(1,22))
display_table.style.background_gradient( axis=1)    .set_properties(**{'max-width': '300px', 'font-size': '12pt'})    .set_caption("Horizontal Wavenumber $k$")    .set_precision(3)

display_table = pd.DataFrame(l, index=np.squeeze(depths), columns=np.arange(1,22))
display_table.style.background_gradient(cmap=cmap, axis=1)    .set_properties(**{'max-width': '300px', 'font-size': '12pt'})    .set_caption("Horizontal Wavenumber $l$")    .set_precision(3)

display_table = pd.DataFrame(omega, index=np.squeeze(depths), columns=np.arange(1,22))
display_table.style.background_gradient(cmap=cmap, axis=1)    .set_properties(**{'max-width': '300px', 'font-size': '12pt'})    .set_caption("Horizontal Wavenumber $l$")    .set_precision(3)

# Generate a wave
l1 = 0.00012
k1 = 0.00012
m1 = -(2*np.pi)/1000
z0 = 1000
w0 = -0.000125
wave1 = rt.wave(k=k1, l=l1, m=m1, w0=w0, z0=z0)

# check that the properties are loaded correctly by using the properties attribute
wave1.properties()

duration = 24
tstep = 10
status = 6 # intervals to give run status
wave1.back3d(duration=duration, tstep=tstep, status=status, print_run_report=True)
wave1.x_m_plot(cmap='Reds', line_colorbar=True)
plt.title('Test Run')

# Frequency Variations
f = -0.00011686983432556936
N = np.sqrt(np.nanmean(N2))
omegas = np.linspace(f, -N, num=50) 
waves = [rt.wave(k=k1, l=l1, m=m1, w0=omega1, z0=z0) for omega1 in omegas]

duration = 48
tstep = 10
status = 6 
seafloor = 4000
for wave in waves:
    wave.back3d(duration=duration, tstep=tstep,
                status=status, seafloor=seafloor, 
                updates=False, print_run_report=False)
    

# plot frequency variation
wave_lines = []
plt.figure(figsize=[10,8])
for wave in waves:
    wave_lines.append(oc.colorline(wave.x_ray.flatten(),
                                   wave.z_ray.flatten(),
                                   wave.m_ray.flatten(),
                                  cmap=cmocean.cm.thermal,
                                  norm=None))
    
# Plot Rays
plt.xlim(0,30)
plt.ylim(500,4000)
plt.gca().invert_yaxis()
cb1 = plt.colorbar(wave_lines[0])
cb1.set_label('Inverse vertical wave numnber ($m^{-1}$)')
plt.title('Frequency Tests $\omega_0$ = $N^2$ to $f$ \n Run Duration: {} Hours'.format(duration))
plt.xlabel('Distance (km)')
plt.ylabel('Depth (m)')


waves1 = [rt.wave(k=k1, l=l1, m=m1, w0=omega1, z0=z0) for omega1 in omegas]
meanU = np.nanmean(waves[0].U)
meandU = np.nanmean(waves[0].dudz)
meanV = np.nanmean(waves[0].V)
meandv = np.nanmean(waves[0].dvdz)
for wave in waves1:
    wave.U = meanU*(wave.U/wave.U)
    wave.dudz = meandU*(wave.dudz/wave.dudz)
    wave.V = meanV*(wave.V/wave.V)
    wave.dvdz = meandv*(wave.dvdz/wave.dvdz)
    wave.back3d(duration=duration, tstep=tstep,
                status=status, seafloor=seafloor,
                print_run_report=False,
                updates=False)

# Plot frequency variation with constant U
wave_lines = []
plt.figure(figsize=[10,8])
for wave in waves1:
    wave_lines.append(oc.colorline(wave.x_ray.flatten(),
                                   wave.z_ray.flatten(),
                                   wave.m_ray.flatten(),
                                  cmap=cmocean.cm.thermal,
                                  norm=None))
    
# Plot Rays
plt.xlim(0,30)
plt.ylim(500,4000)
plt.gca().invert_yaxis()
cb1 = plt.colorbar(wave_lines[0])
cb1.set_label('Inverse vertical wave numnber ($m^{-1}$)')
plt.title('Frequency Tests $\omega_0$ = $N^2$ to $f$ \n Run Duration: {} Hours'.format(duration))
plt.xlabel('Distance (km)')
plt.ylabel('Depth (m)')



# Frequency Variation with constant N2
waves2 = [rt.wave(k=k1, l=l1, m=m1, w0=omega1, z0=z0) for omega1 in omegas]
meanN2 = np.nanmean(waves[0].N2)
for wave in waves2:
    wave.N2 = meanN2*(wave.N2/wave.N2)
    wave.back3d(duration=duration, tstep=tstep,
                status=status, seafloor=seafloor,
                print_run_report=False,
                updates=False)


# Plot with constant buoyancy frequency
wave_lines = []
plt.figure(figsize=[10,8])
for wave in waves2:
    wave_lines.append(oc.colorline(wave.x_ray.flatten(),
                                   wave.z_ray.flatten(),
                                   wave.m_ray.flatten(),
                                  cmap=cmocean.cm.thermal,
                                  norm=None))
    
# Plot Rays
plt.xlim(0,30)
plt.ylim(500,4000)
plt.gca().invert_yaxis()
cb1 = plt.colorbar(wave_lines[0])
cb1.set_label('Inverse vertical wave numnber ($m^{-1}$)')
plt.title('Frequency Tests $\omega_0$ = $N^2$ to $f$ \n Run Duration: {} Hours'.format(duration))
plt.xlabel('Distance (km)')
plt.ylabel('Depth (m)')







