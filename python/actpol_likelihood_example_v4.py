get_ipython().magic('matplotlib inline')
import numpy as np
from scipy.io import FortranFile
import matplotlib.pyplot as plt
import sys
sys.path.append('./actpols2')
import act_like
import camb

data_dir = './actpols2/data/'
act = act_like.ACTPol_s2(data_dir)
act.test()

H0 = 67.5    
ombh2 = 0.022
omch2 = 0.122
tau = 0.06
ns = 0.965
As = 2.5e-09

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
pars.InitPower.set_params(ns=ns, r=0, As=As)
pars.set_for_lmax(6000, lens_potential_accuracy=2);

results = camb.get_results(pars)

powers = results.get_cmb_power_spectra(pars)

totCL = powers['total']

negloglike = act.loglike(totCL[2:6001,0]*1e12*2*np.pi, totCL[2:6001,3]*1e12*2*np.pi, totCL[2:6001,1]*1e12*2*np.pi, 0.98)
print 2*negloglike

# Plot the ACTPol bandpowers and theory spectra

nbin = 132   # total number of bins across TT, TE and EE spectra
nbintt = 42  # multipoles 500-4000
nbinte = 45  # multipoles 250-4000
nbinee = 45  # multipoles 250-4000
like_file = data_dir + '/cl_cmb_aps2.dat'
bval, X_data, X_sig = np.genfromtxt(like_file, max_rows=nbin, delimiter = None, unpack=True)

# Temperature (TT) power spectra

# showing l(l+1)Cl
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), totCL[2:6001,0]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[:nbintt], bval[:nbintt]*(bval[:nbintt]+1)*X_data[:nbintt]*0.5/np.pi,              bval[:nbintt]*(bval[:nbintt]+1)*X_sig[:nbintt]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.ylim([-100, 6000])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'TT power $\ell(\ell+1)C_{\ell}^{TT}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=16)
plt.show()

# showing l^3(l+1)Cl to enchance damping tail
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), np.arange(2,6001)**2*totCL[2:6001,0]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[:nbintt], bval[:nbintt]**3*(bval[:nbintt]+1)*X_data[:nbintt]*0.5/np.pi,              bval[:nbintt]**3*(bval[:nbintt]+1)*X_sig[:nbintt]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'TT power $\ell^3(\ell+1)C_{\ell}^{TT}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=16)
plt.show()

# Temperature-polarization (TE) cross-power spectra

# showing l(l+1)Cl
bins = np.arange(nbintt, nbintt+nbinte)
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), totCL[2:6001,3]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[bins], bval[bins]*(bval[bins]+1)*X_data[bins]*0.5/np.pi,              bval[bins]*(bval[bins]+1)*X_sig[bins]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'TE power $\ell(\ell+1)C_{\ell}^{TE}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=16)
plt.show()

# showing l^3(l+1)Cl to enhance damping tail
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), np.arange(2,6001)**2*totCL[2:6001,3]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[bins], bval[bins]**3*(bval[bins]+1)*X_data[bins]*0.5/np.pi,              bval[bins]**3*(bval[bins]+1)*X_sig[bins]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'TE power $\ell^3(\ell+1)C_{\ell}^{TE}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=16)
plt.show()

# E-mode polarization (EE) power spectra

# showing l(l+1)Cl
bins = np.arange(nbintt+nbinte, nbin)
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), totCL[2:6001,1]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[bins], bval[bins]*(bval[bins]+1)*X_data[bins]*0.5/np.pi,              bval[bins]*(bval[bins]+1)*X_sig[bins]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'EE power $\ell(\ell+1)C_{\ell}^{EE}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=14)
plt.show()

# showing l^3(l+1)Cl to enhance damping tail
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(2,6001), np.arange(2,6001)**2*totCL[2:6001,1]*1e12*2*np.pi, 'k', label='Theory spectrum')
plt.errorbar(bval[bins], bval[bins]**3*(bval[bins]+1)*X_data[bins]*0.5/np.pi,              bval[bins]**3*(bval[bins]+1)*X_sig[bins]*0.5/np.pi, marker='.', ls='None', c='r', lw=1,              label='ACTPol S2')
plt.xlim([0, 4500])
plt.xlabel(r'Multipole $\ell$', size=16)
plt.ylabel(r'EE power $\ell^3(\ell+1)C_{\ell}^{EE}/2\pi$ [$\mu$K$^2$]', size=16)
plt.legend(frameon=0, numpoints=1, fontsize=14)
plt.show()

# show correlation matrices with the diagonal removed

bin_ref = [0, nbintt, nbintt+nbinte, nbin]
label_ref = ['TT', 'TE', 'EE']

# load covariance matrix file
nbin = 132
cov_file = data_dir + '/c_matrix_actpol.dat'
f = FortranFile(cov_file, 'r')
covmat = f.read_reals(dtype=float).reshape((nbin,nbin))
for i_index in range(nbin):
    for j_index in range(i_index, nbin):
        covmat[i_index,j_index] = covmat[j_index,i_index]

# convert from covariance to correlation matrix so we can see off-diagonal structure more easily
temp = np.diag(1/np.sqrt(np.diag(covmat)))
corrmat = np.dot(temp, np.dot(covmat, temp)) - np.eye(nbin)

for i in range(3):
    for j in range(i, 3):
        plt.imshow(corrmat[bin_ref[i]:bin_ref[i+1], bin_ref[j]:bin_ref[j+1]], interpolation='Nearest')
        plt.colorbar()
        plt.xlabel(label_ref[j] + " bandpower number", size=14)
        plt.ylabel(label_ref[i] + " bandpower number", size=14)
        plt.title(label_ref[i] + " x " + label_ref[j] + " correlation coeff.", size=14)
        plt.show()

