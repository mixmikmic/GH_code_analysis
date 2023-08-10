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

std = np.nanstd(omega.flatten())

mean = np.nanmean(omega.flatten())

S, T, P, lat, lon = oc.loadCTD(ctd)
f = np.nanmean(gsw.f(lat))


plt.hist(omega.flatten(), bins=10, range=(np.nanmin(omega), np.nanmax(omega)))

-0.00015293408269703877 -5.5848191654506425e-05

limits = [mean-std,f]

mask = np.isfinite(omega)
test = np.logical_and(omega[mask].flatten() <= limits[1],  omega[mask].flatten() >= limits[0])

sum(test) / len(omega[mask].flatten())

std/f

f

print(-1e-4)
print(f)



