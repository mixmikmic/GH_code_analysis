import h5py
import math
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#f = h5py.File("../LiH-gamma.pwscf.h5","r")
#f = h5py.File("../LiH-arb.pwscf.h5","r")
f = h5py.File("../../bccH/pwscf.pwscf.h5","r")

version = f.get('application/version')
print 'version = ',version[:]
number_of_kpoints = f.get('electrons/number_of_kpoints')
print 'number of kpoints = ',number_of_kpoints[0]
number_of_electrons = f.get('electrons/number_of_electrons')
print 'number of electrons = ',number_of_electrons[0]

atom_pos = f.get('atoms/positions')
print atom_pos[:]

prim_vectors = f.get('supercell/primitive_vectors')
print prim_vectors[:]

# Reciprocal lattice vectors
def get_kspace_basis(basis):
    # Volume factor for reciprocal lattice
    a1, a2, a3 = basis
    vol = a1.dot(np.cross(a2, a3))

    pre = 2*math.pi
    #pre = 1.0
    b1 = pre*np.cross(a2, a3)/vol
    b2 = pre*np.cross(a3, a1)/vol
    b3 = pre*np.cross(a1, a2)/vol
    return [b1, b2, b3]

kbasis = get_kspace_basis(prim_vectors)
print kbasis

kpoint = f.get('electrons/kpoint_0/reduced_k')
print kpoint[:]

gvectors = f.get('electrons/kpoint_0/gvectors')
print gvectors[0:10,:]

pw_coeffs = f.get('electrons/kpoint_0/spin_0/state_0/psi_g')
print pw_coeffs.shape
print pw_coeffs[0:10,:]

# Compute the orbital value at one point in real-space 
def compute_psi(gvectors, kbasis, coeff, twist, r):
    kp = kbasis[0]*twist[0] + kbasis[1]*twist[1] + kbasis[2]*twist[2]
    total_r = 0.0
    total_i = 0.0
    for idx in range(len(gvectors)):
        G = gvectors[idx]
        c = coeff[idx]
        q = kbasis[0]*G[0] + kbasis[1]*G[1] + kbasis[2]*G[2] + kp
        qr = np.dot(q,r)
        cosqr = math.cos(qr)
        sinqr = math.sin(qr)
        total_r += c[0] * cosqr - c[1] * sinqr
        total_i += c[0] * sinqr + c[1] * cosqr
    #print 'total = ',total_r, total_i
    return complex(total_r, total_i)
        

# Test it out at one point.
r = np.array([0.0, 0.0, 0.0])
compute_psi(gvectors, kbasis, pw_coeffs, kpoint, r)

# Compute a range of values
psi_vals = []
rvals = []
nstep = 10
cell_width = prim_vectors[0,0]
step = cell_width/nstep
for i in range(nstep+1):
    r1 = step*i
    rvals.append(r1)
    r = np.array([r1, 0.0, 0.0])
    pv = compute_psi(gvectors, kbasis, pw_coeffs, kpoint, r)
    print r1, pv
    psi_vals.append(pv)
    

plt.plot(rvals, [p.real for p in psi_vals])

# Find the mesh size
# See EinsplineSetBuilder::ReadGvectors_ESHDF in QMCWavefunctions/EinsplineSetBuilderReadBands_ESHDF.cpp
#  Mesh sizes taken from QMCPACK output.
# BCC H
#meshsize = (52, 52, 52)
# LiH
#meshsize = (68, 68, 68)
MeshFactor = 1.0

max_g = np.zeros(3)
for g in gvectors:
    max_g = np.maximum(max_g, np.abs(g))
    
print 'Maximum G = ',max_g
meshsize = np.ceil(4*max_g*MeshFactor).astype(np.int)
print 'mesh size = ',meshsize

# Plus some more code for mesh sizes larger than 128 than restricts
# sizes to certain allowed values (more efficient FFT?)

# Place points in the box at the right G-vector
# see unpack4fftw in QMCWavefunctions/einspline_helper.h

fftbox = np.zeros(meshsize, dtype=np.complex_)
for c, g in zip(pw_coeffs, gvectors):
    idxs = [(g[i] + meshsize[i])%meshsize[i] for i in range(3)]
    fftbox[idxs[0], idxs[1], idxs[2]] = complex(c[0], c[1])

realbox = scipy.fftpack.fftn(fftbox)

fftvals = np.array([a.real for a in realbox[0:meshsize[0],0,0]])
fftvals

xstep = prim_vectors[0][0]/meshsize[0]
xvals = [xstep * i for i in range(meshsize[0])]

# Compare results of FFT and the compute_psi function
# They don't line up completely because they are on different real-space grids
line1 = plt.plot(rvals, [p.real for p in psi_vals], label="compute_psi")
line2 = plt.plot(xvals, fftvals, label="FFT")
plt.legend()

realbox_kr = np.empty_like(realbox)
for ix in range(meshsize[0]):
    for iy in range(meshsize[1]):
        for iz in range(meshsize[2]):
            tx = kpoint[0]*ix/meshsize[0]
            ty = kpoint[1]*iy/meshsize[1]
            tz = kpoint[2]*iz/meshsize[2]
            tt = -2*np.pi*(tx+ty+tz)
            cos_tt = math.cos(tt)
            sin_tt = math.sin(tt)
            r = realbox[ix, iy, iz]
            realbox_kr[ix,iy,iz] = r*complex(cos_tt, sin_tt)
rNorm = 0.0
iNorm = 0.0
ii = 0
for val in np.nditer(realbox_kr):
#for val in psi_vals:
    rNorm += val.real*val.real
    iNorm += val.imag*val.imag
    ii += 1
print 'real norm, imaginary norm',rNorm,iNorm
arg = math.atan2(iNorm, rNorm)
print 'angle (degrees)',math.degrees(arg)
ang = np.pi/8 - 0.5*arg
sin_ang = math.sin(ang)
cos_ang = math.cos(ang)
rot_psi_vals = []
for val in psi_vals:
    rot = val.real*cos_ang - val.imag*sin_ang
    rot_psi_vals.append(rot)

# These values should be comparable to the output of the spline orbitals
rot_psi_vals

# These are on a different grid than the values above
fft_rot_vals = []
for val in realbox_kr[:,0,0]:
    rot = val.real*cos_ang - val.imag*sin_ang
    fft_rot_vals.append(rot)
fft_rot_vals[0:10]



