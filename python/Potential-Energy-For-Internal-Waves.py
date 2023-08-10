# Load Data and relevant modules - this is common to both wayss
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import data_load
import gsw
import oceans as oc
import pandas as pd
import internal_waves_calculations as iwc
import warnings
import cmocean
# Probably Shouldn't do this but they annoy me
warnings.simplefilter("ignore")

pd.options.display.max_rows = 3000

pd.options.display.max_columns = 22


# load data and cut off bottom (its all nans)
ladcp, ctd = data_load.load_data()
strain = np.genfromtxt('strain.csv', delimiter=',')
wl_max = 900
wl_min = 400
ctd_bin_size = 1024
ladcp_bin_size = 1024
nfft = 1024
U, V, p_ladcp = oc.loadLADCP(ladcp)
S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
rho = gsw.density.rho(S, T, p_ctd)

maxDepth = 4000
idx_ladcp = p_ladcp[:, -1] <= maxDepth
idx_ctd = p_ctd[:, -1] <= maxDepth

strain = strain[idx_ctd, :]

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
dist = gsw.distance(lon, lat)
dist = np.cumsum(dist)/1000
dist = np.append(0,dist)

# Calculate potential energy density spectrum using adiabatic leveling
# --- This is the part that needs tweaking I think

# Adiabatic leveling following Bray and Fofonoff 1981 -
# the actual code is a python version of Alex Forryan's Matlab code

# Order = order of polynomial fit to use
order = 1

# Pressure window - See Bray and Fofonoff 1981 for details
pressure_range = 400 

# axis = the depth increases on
axis = 0

# Use Adiabtic Leveling from Oceans Library
N2_ref, N2, strain, p_mid, rho_bar = oc.adiabatic_level(S,
                                                        T,
                                                        p_ctd,
                                                        lat,
                                                        pressure_range=pressure_range,
                                                        order=order,
                                                        axis=axis,
                                                        )

rho_ref = np.nanmean(rho_bar, axis=1)

# Stick a nan to the top to make it the same size as the rho array
rho_ref = np.hstack((0, rho_ref))

# set differnece window to 400 meters
win = 400


# since all the data is on a normalized pressure grid use a single vertical vector to make it easier to handle
z = -1*gsw.z_from_p(p_ctd[:,0], lat[:,0])
dz = np.nanmean(np.diff(z))
step = int(np.floor(.5*win/dz))
eta = np.full_like(rho, np.nan)
for i in range(rho.shape[0]):
    
    # If in the TOP half of the profile the window needs to be adjusted
    if i - step < 0:
        lower = 0
        upper = int(2*step)
        
    # If in the BOTTOM half of the profile the window needs to be adjusted
    elif i + step > (rho.shape[0] - 1):
        lower = int(rho.shape[0] - 2*step)
        upper = -1 
        
    else:
        upper = i + step
        lower = i - step
    drefdz = (rho_ref[upper] - rho_ref[lower])/win
    
    eta[i,:] = (rho[i,:] -  rho_ref[i])/drefdz
        

# Calculate KE spectrums (m2/s2)
z_ladcp = -1*gsw.z_from_p(p_ladcp, lat)
KE, KE_grid, KE_psd, Uprime, Vprime, ke_peaks = iwc.KE_UV(U, V, z_ladcp, ladcp_bins,
                                                    wl_min, wl_max, lc=wl_min-50,
                                                    nfft=1024, detrend='constant')
# Calculate PE spectrum using eta from above (m2/s2)
z_ctd = -1*gsw.z_from_p(p_ctd, lat)
PE, PE_grid, eta_psd, N2mean, pe_peaks = iwc.PE(N2, z, eta,
                                                wl_min, wl_max,
                                                ctd_bins, nfft=1024,
                                                detrend=False)



# Plot spectra to see what happened
m_plot = np.array([(1)/wl_max,
                       (1)/wl_max, (1)/wl_min,
                       (1)/wl_min])
plt.figure(figsize=[12,6])

plt.loglog(KE_grid, KE_psd.T, linewidth=.6, c='b', alpha=.05)
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
plt.xlabel('Vertical Wavenumber')
plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)

plt.loglog(PE_grid, .5*np.nanmean(N2)*eta_psd.T,
           lw=.6, c='r', alpha=.05)
plt.loglog(PE_grid, .5*np.nanmean(N2)*np.nanmean(eta_psd, axis=0).T,
           lw=2, c='r')
plt.plot(m_plot[2:], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.ylim(ylims)
# plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)
plt.ylabel('Potential Energy Density (m2/s2)')
plt.xlabel('Vertical Wavenumber')
# plt.xlim(.0005, .01)

Etotal = 1027*(KE + PE) # Multiply by density to get Joules
# wave components
f = np.nanmean(gsw.f(lat))

# version 2 omega calculation - where did this come from?
omega = f*np.sqrt(((KE+PE)/(KE-PE))) # Waterman et al. 2012 (Ithink)

m = (2*np.pi)/800

kh = m*np.sqrt(((f**2 - omega**2)/(omega**2 - N2mean))) # Waterman et al. 2012
kh2 = (m/np.sqrt(N2mean))*(np.sqrt(omega**2 - f**2)) # Where (meyer i think?)

lambdaH = 1e-3*(2*np.pi)/kh
lambdaH2 = 1e-3*(2*np.pi)/kh2

# version 2 omega calculation
Rw = KE/PE # Unsure what to do with this just yet. 

table = oc.display(lambdaH, index=depths.flatten())
table.style.set_caption('Horizontal Wavelength V1')

# Neutral Densities
rho_neutral =  np.genfromtxt('neutral_densities.csv', delimiter=',')
rho_n = rho_neutral[idx_ctd,:]

# Poly fit to neutral density to get reference profiles
ref = []

for cast in rho_n.T:
    fitrev = oc.vert_polyFit2(cast, p_ctd[:, 0], 100, deg=2)
    ref.append(fitrev)

ref = np.vstack(ref).T
eta = oc.isopycnal_displacements(rho_n, ref, p_ctd, lat)

ref = np.nanmean(ref, axis=1)

# recalculate spectrum
z_ctd = -1*gsw.z_from_p(p_ctd, lat)
PE, PE_grid, eta_psd, N2mean, pe_peaks = iwc.PE(N2, z, eta,
                                                wl_min, wl_max,
                                                ctd_bins, nfft=1024,
                                                detrend=False)



# Plot spectra to see what happened
m_plot = np.array([(1)/wl_max,
                       (1)/wl_max, (1)/wl_min,
                       (1)/wl_min])
plt.figure(figsize=[12,6])

plt.loglog(KE_grid, KE_psd.T, linewidth=.6, c='b', alpha=.05)
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
plt.xlabel('Vertical Wavenumber')
plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)

plt.loglog(PE_grid, .5*np.nanmean(N2)*eta_psd.T,
           lw=.6, c='r', alpha=.05)
plt.loglog(PE_grid, .5*np.nanmean(N2)*np.nanmean(eta_psd, axis=0).T,
           lw=2, c='r')
plt.plot(m_plot[2:], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.plot(m_plot[:2], ylim1, lw=1,
         c='k', alpha=.5,
         linestyle='dotted')
plt.ylim(ylims)
# plt.gca().grid(True, which="both", color='k', linestyle='dotted', linewidth=.2)
plt.ylabel('Potential Energy Density (m2/s2)')
plt.xlabel('Vertical Wavenumber')
# plt.xlim(.0005, .01)


Etotal = 1027*(KE + PE) # Multiply by density to get Joules
# wave components
f = np.nanmean(gsw.f(lat))

# version 2 omega calculation - where did this come from?
omega = f*np.sqrt(((KE+PE)/(KE-PE))) # Waterman et al. 2012 (Ithink)

m = (2*np.pi)/800

kh = m*np.sqrt(((f**2 - omega**2)/(omega**2 - N2mean))) # Waterman et al. 2012
kh2 = (m/np.sqrt(N2mean))*(np.sqrt(omega**2 - f**2)) # Where (meyer i think?)

lambdaH = 1e-3*(2*np.pi)/kh
lambdaH2 = 1e-3*(2*np.pi)/kh2

# version 2 omega calculation
Rw = KE/PE # Unsure what to do with this just yet. 

table = oc.display(Etotal, index=depths.flatten())
table.style.set_caption('Horizontal Wavelength V1')

table = oc.display(lambdaH, index=depths.flatten())
table.style.set_caption('Horizontal Wavelength V1')

plt.rcParams.update({'font.size':12})

ref = np.nanmean(ref, axis=1)

ref.shape



