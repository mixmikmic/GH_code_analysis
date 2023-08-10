# Load Data and relevant modules - this is common to both wayss
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import data_load
import gsw
import oceans as oc
import pandas as pd
import internal_waves_calculations as iwc
import warnings
import cmocean
import matplotlib.colors as colors
import ray_tracing_satGEM as rt
from netCDF4 import Dataset
from datetime import datetime, timedelta


from jupyterthemes import jtplot

# choose which theme to inherit plotting style from
# onedork | grade3 | oceans16 | chesterish | monokai | solarizedl | solarizedd
jtplot.style(theme='chesterish')

# Probably Shouldn't do this but they annoy me
warnings.simplefilter("ignore")

pd.options.display.max_rows = 3000
pd.options.display.max_columns = 22

plt.rcParams.update({'font.size':14})


# load data and cut off bottom (its all nans)
ladcp, ctd = data_load.load_data()
wl_max = 1000
wl_min = 400
ctd_bin_size = 1024
ladcp_bin_size = 1024
nfft = 1024

U, V, p_ladcp = oc.loadLADCP(ladcp)
S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
rho = gsw.density.rho(S, T, p_ctd)
N2, n2grid = gsw.Nsquared(S, T, p_ctd)
maxDepth = 4000
idx_ladcp = p_ladcp[:, -1] <= maxDepth
idx_ctd = p_ctd[:, -1] <= maxDepth


S = S[idx_ctd,:]
T = T[idx_ctd,:]
rho = rho[idx_ctd,:]
p_ctd = p_ctd[idx_ctd, :]
U = U[idx_ladcp, :]
V = V[idx_ladcp, :]
p_ladcp = p_ladcp[idx_ladcp, :]


# Bin CTD data
ctd_bins = oc.binData(S, p_ctd[:, 0], ctd_bin_size)
# Bin Ladcp Data
ladcp_bins = oc.binData(U, p_ladcp[:, 0], ladcp_bin_size)

# Depth and lat/long grids (For plots)
depths = np.vstack([np.nanmean(p_ctd[binIn]) for binIn in ctd_bins])
depths2 = np.tile(depths, (24,1))
dist = gsw.distance(lon, lat)
dist = np.cumsum(dist)/1000
dist = np.append(0,dist)

# Plot map of transect and bathymetry
bathy = Dataset('bathy.nc')

buffer = 0.2
latlims = np.array([np.nanmin(lat)-buffer, np.nanmax(lat)+buffer])
latlims = [np.argmin(np.abs(lat_in - bathy['lat'][:])) for lat_in in latlims]
latlims = np.arange(latlims[0], latlims[1])

lonlims = np.array([np.nanmin(lon)-buffer, np.nanmax(lon)+buffer])
lonlims = [np.argmin(np.abs(lon_in - bathy['lon'][:])) for lon_in in lonlims]
lonlims = np.arange(lonlims[0], lonlims[1])

bathy_rev = bathy['elevation'][latlims, lonlims]
lat_b = bathy['lat'][latlims]
lon_b = bathy['lon'][lonlims]

blevels = np.linspace(np.nanmin(bathy_rev), np.nanmax(bathy_rev), 20)
plt.figure(figsize=(15, 10))
plt.pcolormesh(lon_b, lat_b, bathy_rev, shading='gouraud')
c = plt.colorbar()
c.set_label('Depth (meters)')
plt.plot(lon[:].flatten(), lat[:].flatten(), c='r', marker='+', ms=15)
plt.contour(lon_b, lat_b, bathy_rev, colors='k', alpha=.5, levels=blevels)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# add mean flow vectors
depth = 2000
cutoff = np.argmin(np.abs(p_ladcp[:] - depth))
Umean, Vmean = oc.depthAvgFlow(U, V, 8, depthRange=depth)
plt.quiver(lon[:].flatten(), lat[:].flatten(), Umean, Vmean)


# Plot transect contours of temps and velocities to look at
tlevels = np.linspace(np.nanmin(T), np.nanmax(T), 20)

plt.figure(figsize=(14, 16))
plt.subplot(311)
plt.pcolor(dist.flatten(), p_ctd[:,0].flatten(), T, vmin=0, vmax=5, cmap=cmocean.cm.thermal)
c = plt.colorbar()
plt.contour(dist.flatten(), p_ctd[:,0].flatten(), T, colors='k', linewidths=.5, levels=tlevels)
plt.gca().invert_yaxis()

c.set_label(r'$^{\circ}C$')
plt.title('Temperature')
plt.ylabel('Pressure (dB)')

ulevels = np.linspace(np.nanmin(U), np.nanmax(U), 5)
plt.subplot(312)
plt.pcolor(dist.flatten(), p_ladcp[:,0].flatten(), U,  vmin=-.4, vmax=.4, cmap=cmocean.cm.balance)
plt.gca().invert_yaxis()
c = plt.colorbar()
c.set_label(r'$\frac{m}{s}$')
plt.title('Zonal Velocity')
plt.ylabel('Pressure (dB)')

plt.subplot(313)
plt.pcolor(dist.flatten(), p_ladcp[:,0].flatten(), V, vmin=-.4, vmax=.4, cmap=cmocean.cm.balance)
plt.gca().invert_yaxis()
c = plt.colorbar()
c.set_label(r'$\frac{m}{s}$')
plt.title('Meridional Velocity')
plt.xlabel('Distance Along Transect (km)')
plt.ylabel('Pressure (dB)')



# N2_ref, N2, strain, p_mid, rho_bar = oc.adiabatic_level(S,
#                                                         T,
#                                                         p_ctd,
#                                                         lat,
#                                                         pressure_range=400, # Difference window for leveling
#                                                         order=1, # order of fit
#                                                         axis=0, 
#                                                         )


# # Neutral Densities
rho_neutral =  np.genfromtxt('neutral_densities.csv', delimiter=',') # fix station numbers
rho_n = rho_neutral[idx_ctd,:]

# Poly fit to neutral density to get reference profiles
ref = []

for cast in rho_n.T:
    fitrev = oc.vert_polyFit2(cast, p_ctd[:, 0], 100, deg=2)
    ref.append(fitrev)

ref = np.vstack(ref).T

ref = np.nanmean(ref, axis=1)
ref2 = np.tile(ref,(24,1)).T
eta = oc.isopycnal_displacements(rho_n, ref2, p_ctd, lat)

# set integration limits by vertical wavelength 
wl_max = 1000
wl_min = 500
# Calculate KE spectrums (m2/s2)
z_ladcp = -1*gsw.z_from_p(p_ladcp, lat)
KE, KE_grid, KE_psd, Uprime, Vprime, ke_peaks = iwc.KE_UV(U, V, z_ladcp, ladcp_bins,
                                                    wl_min, wl_max, lc=wl_min-50,
                                                    nfft=1024, detrend='constant')


# Calculate PE spectrum using eta from above (m2/s2)
z_ctd = -1*gsw.z_from_p(p_ctd, lat)
PE, PE_grid, eta_psd, N2mean, pe_peaks = iwc.PE(N2, z_ctd, eta,
                                                wl_min, wl_max,
                                                ctd_bins, nfft=1024,
                                                detrend='constant')

m_plot = np.array([(1)/wl_max,
                       (1)/wl_max, (1)/wl_min,
                       (1)/wl_min])
plt.figure(figsize=[12,6])

plt.loglog(KE_grid, KE_psd.T, linewidth=.6, c='b', marker='x', alpha=.05)
plt.loglog(KE_grid, np.nanmean(KE_psd, axis=0).T, lw=2, c='b')
ylims = plt.gca().get_ylim()
ylim1 = np.array([ylims[0], ylims[1]])
plt.plot(m_plot[2:], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.ylim(ylims)
plt.ylabel('Kinetic Energy Density')
plt.xlabel('Inverse Vertical Wavelength')
plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)

plt.loglog(PE_grid, .5*np.nanmean(N2)*eta_psd.T,
           lw=.6, c='r', marker='x', alpha=.05)
plt.loglog(PE_grid, .5*np.nanmean(N2)*np.nanmean(eta_psd, axis=0).T,
           lw=2, c='r')
plt.plot(m_plot[2:], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.ylim((1e-8, 1e2))
# plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)
plt.ylabel(r'Potential Energy Density $m^2s^2$')
plt.xlabel('Vertical Wavenumber')
plt.title('Kinetic And Potential Energy Spectrum')
# plt.xlim(.0005, .01)
ax = plt.gca()
plt.text(4e-4, 1e-4,r'$\lambda$ = {} m'.format(wl_max))
plt.text(2.8e-3, 1e-7,r'$\lambda$ = {} m'.format(wl_min))
potential = mpatches.Patch(color='red', label='Potential Energy')
kinetic = mpatches.Patch(color='b', label='Kinetic Energy')
plt.legend(handles=[kinetic, potential])


# ENERGY AND WAVE COMPONENT PLOTS
Etotal = 1027*(KE + PE) # Multiply by density to get Joules
# wave components
f = np.nanmean(gsw.f(lat))

# version 2 omega calculation - where did this come from?
omega = f*np.sqrt(((KE+PE)/(KE-PE))) # Waterman et al. 2012 (Ithink)

# m = (2*np.pi)/800

m = (2*np.pi)*np.nanmean(ke_peaks, axis=1)
m = np.reshape(m, KE.shape, order='F') 


kh = m*np.sqrt(((f**2 - omega**2)/(omega**2 - N2mean))) # Waterman et al. 2012
kh2 = (m/np.sqrt(N2mean))*(np.sqrt(omega**2 - f**2)) # Where (meyer i think?)

lambdaH = 1e-3*(2*np.pi)/kh
lambdaH2 = 1e-3*(2*np.pi)/kh2

# version 2 omega calculation
Rw = KE/PE # Unsure what to do with this just yet. 

# Wave vector decomposition
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

Elevels = np.linspace(np.nanmin(Etotal), np.nanmax(Etotal), 10)
mask = Etotal > 8
Etotal2 = Etotal
Etotal2[mask] = np.nan
plt.figure(figsize=[15,22])
plt.subplot(411)
plt.pcolormesh(dist.flatten(), depths.flatten(), np.log10(Etotal), cmap='magma')
c1 = plt.colorbar(extend='max')
c1.set_label(r'log10($J/m^3$)')
plt.gca().invert_yaxis()
plt.xlabel('Distance along transect')
plt.ylabel('Depth')
plt.title(r'Total Internal Wave Energy  $\frac{J}{m^{3}}$  ')

mask = lambdaH > 150
norm=colors.LogNorm(vmin=lambdaH.min(), vmax=lambdaH.max())

# lambdaH[mask] = np.nan
plt.subplot(412)
plt.pcolormesh(dist.flatten(), depths.flatten(), (lambdaH), cmap=cmocean.cm.curl, norm=None)
c2 = plt.colorbar(extend='max')
c2.set_label(r'km')
plt.gca().invert_yaxis()
plt.xlabel('Distance along transect')
plt.title('Horizontal Wavelength Vector Magnitude ')


plt.subplot(413)
plt.pcolormesh(dist.flatten(), p_ladcp.flatten(), (Uprime), 
               cmap=cmocean.cm.balance,
              vmin=-.15, vmax=.15)
c3 = plt.colorbar()
c3.set_label(r'$m/s$')
plt.xlabel('Distance along transect')
plt.ylabel('Depth')
plt.title(r"$u'$")
plt.ylim(0,3000)
plt.gca().invert_yaxis()

plt.subplot(414)
plt.pcolormesh(dist.flatten(), p_ladcp.flatten(), Vprime, 
               cmap=cmocean.cm.balance,
              vmin=-.15, vmax=.15)
c3 = plt.colorbar()
c3.set_label(r'$m/s$')
plt.xlabel('Distance along transect')
plt.ylabel('Depth')
plt.title(r"$v'$")
plt.ylim(0,3000)
plt.gca().invert_yaxis()

plt.tight_layout()

# plt.subplot(615)
# plt.pcolormesh(dist.flatten(), p_ladcp.flatten(), U,
#                vmin = -.3, vmax= 0.3, cmap=cmocean.cm.delta, shading='flat')
# c3 = plt.colorbar(extend='max')
# c3.set_label(r'$m/s$')
# plt.xlabel('Distance along transect')
# plt.ylabel('Depth')
# plt.title(r"$U $")
# plt.ylim(0,3000)
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plt.subplot(616)
# plt.pcolormesh(dist.flatten(), p_ladcp.flatten(), V,
#                vmin = -.3, vmax= 0.3, cmap=cmocean.cm.delta, shading='flat')
# c3 = plt.colorbar(extend='max')
# c3.set_label(r'$m/s$')
# plt.xlabel('Distance along transect')
# plt.ylabel('Depth')
# plt.title(r"$V $")
# plt.ylim(0,3000)
# plt.gca().invert_yaxis()
# plt.tight_layout()

# calculating momentum fluxes in horizontal and vertical components

## Vertical group speed (assume no mean vertical flow)
cgz = rt.CGz(omega, k , l, m, f, N2mean)
Mz = cgz*Etotal
table = oc.display(Mz, index=depths.flatten())
table.style.set_caption('Horizontal Wavelength V1')

table = oc.display(omega, index=depths.flatten())
table.style.set_caption('Horizontal Wavelength V1')

fig = plt.figure(figsize=[10,6])
plt.subplot(121)
plt.hist(lambdaH.flatten(), bins=15, range=(np.nanmin(lambdaH), np.nanmax(lambdaH)))
plt.title(r'$\lambda_h$')
plt.xlabel(r'Horizontal Wavelength Magnitude (km)')
plt.tight_layout()

plt.subplot(122)
plt.hist(omega.flatten(), bins=20, range=(np.nanmin(omega), np.nanmax(omega)))
plt.title(r'$\omega_0$')
plt.xlabel(r'Frequency ($s^{-1}$)')
plt.tight_layout()
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(1,1), axis='x')


table = oc.display(k, index=depths.flatten())
table.style.set_caption(r'Horizontal Wavenumber $k$')

table = oc.display(l, index=depths.flatten())
table.style.set_caption(r'Horizontal Wavenumber $l$')

table = oc.display(m, index=depths.flatten())
table.style.set_caption(r'vertical Wavenumber $m$')

# This is a first attempt at running ray tracing with all the observed wave features.
from datetime import datetime, timedelta
# just use the one date for now (LOAD IN REAL DATETIMES)
towyo_date =  datetime(2012, 2, 28, 21, 33, 44)
plt.figure(figsize=(8,8))
# plt.pcolormesh(lon_b, lat_b, bathy_rev, shading='gouraud')
# plt.contour(lon_b, lat_b, bathy_rev, colors='k')
gem = rt.satGEM_field()
k0 = 0.000379449
l0 = -0.000395896
m0 = -0.0062492
w0 = -0.00014730
wave = rt.Wave(k=k0, l=l0, m=m0, w0=w0, z0=1500,t0=towyo_date,
               lat=lat[:,5], lon=lon[:,5]
              )
              

results = rt.run_tracing(wave, gem, duration=24, tstep=15, time_direction='reverse', status=False)


fig1 = rt.dashboard(results, gem, ms=200, buffer=.1, cls=30)

distance = 1e-3 * np.sqrt(results['x']**2 + results['y']**2)
plt.figure()
plt.subplot(211)
plt.plot(results['elapsed_time'][:-1]/3600, results['cgx'])

plt.subplot(212)
plt.plot(results['elapsed_time'][:-1]/3600, results['cgz'])

# forward testing
k0 = results['k'][-1]
l0 = results['l'][-1]
m0 = results['m'][-1]
w0 = results['omega'][-3]
wave = rt.Wave(k=k0, l=l0, m=m0, w0=w0, z0=1500,
               lat=results['lat'][-1],
               lon=results['lon'][-1],
               t0=results['time'][-1])
gem = rt.satGEM_field()
results1 = rt.run_tracing(wave, gem, duration=7, tstep=30, time_direction='forward', status=False)
fig2 = rt.dashboard(results1, gem, ms=200, buffer=.1, cls=30)



gridI.shape

import importlib as imp
imp.reload(rt)

np.sqrt(np.abs(np.nanmin(np.abs(results['N2']))))

f

t = np.diff(results['elapsed_time'].flatten())
t





