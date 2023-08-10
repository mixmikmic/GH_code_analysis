get_ipython().magic('pylab inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import sys
try:
    sys.path.index('.')
except ValueError:
    sys.path.append('.')

import h5py
import vt

chi1s = linspace(-0.99, 0.99, 25)
chi2s = linspace(-0.99, 0.99, 25)
CHI1S, CHI2S = meshgrid(chi1s, chi2s)
M1S = 40.0*ones_like(CHI1S)
M2S = 30.0*ones_like(CHI2S)

VTS = reshape(vt.vts_from_masses_spins(M1S.ravel(), M2S.ravel(), CHI1S.ravel(), CHI2S.ravel(), 8.0, 1.0), CHI1S.shape)

VTS = 0.25*(VTS[1:,1:] + VTS[1:,:-1] + VTS[:-1,1:] + VTS[:-1,:-1])
VTS /= mean(VTS)

pcolormesh(CHI1S, CHI2S, VTS)
colorbar()
xlabel(r'$\chi_1$')
ylabel(r'$\chi_2$')
title(r'$VT\left( \chi_1, \chi_2 \right) / \langle VT \rangle$')
tight_layout()
savefig('GW150914-like-VTs.pdf')

with h5py.File('GW150914-like-VTs.h5','w') as out:
    out.create_dataset('chi1', data=CHI1S, compression='gzip', shuffle=True)
    out.create_dataset('chi2', data=CHI2S, compression='gzip', shuffle=True)
    out.create_dataset('VT', data=VTS, compression='gzip', shuffle=True)
    out.attrs['MUnits'] = 'MSun'
    out.attrs['m1'] = 40.0
    out.attrs['m2'] = 30.0

M1S_GW151226 = 15.0*ones_like(CHI1S)
M2S_GW151226 = 7.5*ones_like(CHI2S)

VTS_GW151226 = reshape(vt.vts_from_masses_spins(M1S_GW151226.ravel(), M2S_GW151226.ravel(), CHI1S.ravel(), CHI2S.ravel(), 8.0, 1.0), CHI1S.shape)

VTS_GW151226 = 0.25*(VTS_GW151226[1:,1:] + VTS_GW151226[1:,:-1] + VTS_GW151226[:-1,1:] + VTS_GW151226[:-1,:-1])
VTS_GW151226 /= mean(VTS_GW151226)

pcolormesh(CHI1S, CHI2S, VTS_GW151226)
colorbar()
xlabel(r'$\chi_1$')
ylabel(r'$\chi_2$')
title(r'$VT\left( \chi_1, \chi_2 \right) / \langle VT \rangle$')
tight_layout()
savefig('GW151226-like-VTs.pdf')

with h5py.File('GW151226-like-VTs.h5','w') as out:
    out.create_dataset('chi1', data=CHI1S, compression='gzip', shuffle=True)
    out.create_dataset('chi2', data=CHI2S, compression='gzip', shuffle=True)
    out.create_dataset('VT', data=VTS_GW151226, compression='gzip', shuffle=True)
    out.attrs['MUnits'] = 'MSun'
    out.attrs['m1'] = 40.0
    out.attrs['m2'] = 30.0



