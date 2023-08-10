from __future__ import division

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

def ndft(x, f, N):
    """non-equispaced discrete Fourier transform"""
    k = -(N // 2) + np.arange(N)
    return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))

x = -0.5 + np.random.rand(1000)
f = np.sin(10 * 2 * np.pi * x)

k = -20 + np.arange(40)
f_k = ndft(x, f, len(k))

plt.plot(k, f_k.real, label='real')
plt.plot(k, f_k.imag, label='imag')
plt.legend()

# equations C.1 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf

def phi(x, n, m, sigma):
    b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
    return np.exp(-(n * x) ** 2 / b) / np.sqrt(np.pi * b)

def phi_hat(k, n, m, sigma):
    b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
    return np.exp(-b * (np.pi * k / n) ** 2)

from numpy.fft import fft, fftshift, ifftshift

N = 1000
sigma = 1
n = N * sigma
m = 20

# compute phi(x)
x = np.linspace(-0.5, 0.5, N, endpoint=False)
f = phi(x, n, m, sigma)

# compute phi_hat(k)
k = -(N // 2) + np.arange(N)
f_hat = phi_hat(k, n, m, sigma)

# compute the FFT of phi(x)
f_fft = fftshift(fft(ifftshift(f)))

# assure they match
np.allclose(f_fft, f_hat)

import numpy as np


def nfft1(x, f, N, sigma=2):
    """Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
    n = N * sigma  # size of oversampled grid
    m = 20  # magic number: we'll set this more carefully later
    
    # 1. Express f(x) in terms of basis functions phi
    shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
    x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)
    g = np.dot(f, phi(shift_to_range(x[:, None] - x_grid), n, m, sigma))
    
    # 2. Compute the Fourier transform of g on the oversampled grid
    k = -(N // 2) + np.arange(N)
    g_k = np.dot(g, np.exp(2j * np.pi * k * x_grid[:, None]))
    
    # 3. Divide by the Fourier transform of the convolution kernel
    f_k = g_k / phi_hat(k, n, m, sigma)
    
    return f_k

x = -0.5 + np.random.rand(1000)
f = np.sin(10 * 2 * np.pi * x)
N = 100

np.allclose(ndft(x, f, N),
            nfft1(x, f, N))

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift


def nfft2(x, f, N, sigma=2):
    """Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
    n = N * sigma  # size of oversampled grid
    m = 20  # magic number: we'll set this more carefully later
    
    # 1. Express f(x) in terms of basis functions phi
    shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
    x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)
    g = np.dot(f, phi(shift_to_range(x[:, None] - x_grid), n, m, sigma))
    
    # 2. Compute the Fourier transform of g on the oversampled grid
    k = -(N // 2) + np.arange(N)
    g_k_n = fftshift(ifft(ifftshift(g)))
    g_k = n * g_k_n[(n - N) // 2: (n + N) // 2]
    
    # 3. Divide by the Fourier transform of the convolution kernel
    f_k = g_k / phi_hat(k, n, m, sigma)
    
    return f_k

x = -0.5 + np.random.rand(1000)
f = np.sin(10 * 2 * np.pi * x)
N = 100

np.allclose(ndft(x, f, N),
            nfft2(x, f, N))

sigma = 3
n = sigma * N
m = 20
x_grid = np.linspace(-0.5, 0.5, n, endpoint=False)
shift_to_range = lambda x: -0.5 + (x + 0.5) % 1

mat = phi(shift_to_range(x[:, None] - x_grid), n, m, sigma)
plt.imshow(mat, aspect='auto')
plt.colorbar()

from scipy.sparse import csr_matrix

col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
vals = phi(shift_to_range(x[:, None] - col_ind / n), n, m, sigma)
col_ind = (col_ind + n // 2) % n
row_ptr = np.arange(len(x) + 1) * col_ind.shape[1]
spmat = csr_matrix((vals.ravel(), col_ind.ravel(), row_ptr), shape=(len(x), n))

plt.imshow(spmat.toarray(), aspect='auto')
plt.colorbar()
np.allclose(spmat.toarray(), mat)

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift


def nfft3(x, f, N, sigma=2):
    """Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
    n = N * sigma  # size of oversampled grid
    m = 20  # magic number: we'll set this more carefully later
    
    # 1. Express f(x) in terms of basis functions phi
    shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
    col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
    vals = phi(shift_to_range(x[:, None] - col_ind / n), n, m, sigma)
    col_ind = (col_ind + n // 2) % n
    row_ptr = np.arange(len(x) + 1) * col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), row_ptr), shape=(len(x), n))
    g = mat.T.dot(f)
    
    # 2. Compute the Fourier transform of g on the oversampled grid
    k = -(N // 2) + np.arange(N)
    g_k_n = fftshift(ifft(ifftshift(g)))
    g_k = n * g_k_n[(n - N) // 2: (n + N) // 2]
        
    # 3. Divide by the Fourier transform of the convolution kernel
    f_k = g_k / phi_hat(k, n, m, sigma)
    
    return f_k

x = -0.5 + np.random.rand(1000)
f = np.sin(10 * 2 * np.pi * x)
N = 100

np.allclose(ndft(x, f, N),
            nfft3(x, f, N))

def C_phi(m, sigma):
    return 4 * np.exp(-m * np.pi * (1 - 1. / (2 * sigma - 1)))

def m_from_C_phi(C, sigma):
    return np.ceil(-np.log(0.25 * C) / (np.pi * (1 - 1 / (2 * sigma - 1))))

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift


def nfft(x, f, N, sigma=2, tol=1E-8):
    """Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
    n = N * sigma  # size of oversampled grid
    m = m_from_C_phi(tol / N, sigma)
    
    # 1. Express f(x) in terms of basis functions phi
    shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
    col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
    vals = phi(shift_to_range(x[:, None] - col_ind / n), n, m, sigma)
    col_ind = (col_ind + n // 2) % n
    indptr = np.arange(len(x) + 1) * col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    g = mat.T.dot(f)
    
    # 2. Compute the Fourier transform of g on the oversampled grid
    k = -(N // 2) + np.arange(N)
    g_k_n = fftshift(ifft(ifftshift(g)))
    g_k = n * g_k_n[(n - N) // 2: (n + N) // 2]
    
    # 3. Divide by the Fourier transform of the convolution kernel
    f_k = g_k / phi_hat(k, n, m, sigma)
    
    return f_k

x = -0.5 + np.random.rand(1000)
f = np.sin(10 * 2 * np.pi * x)
N = 100

np.allclose(ndft(x, f, N),
            nfft(x, f, N))

from pynfft import NFFT

def cnfft(x, f, N):
    """Compute the nfft with pynfft"""
    plan = NFFT(N, len(x))
    plan.x = x
    plan.precompute()
    plan.f = f
    # need to return a copy because of a
    # reference counting bug in pynfft
    return plan.adjoint().copy()

np.allclose(cnfft(x, f, N),
            nfft(x, f, N))

x = -0.5 + np.random.rand(10000)
f = np.sin(10 * 2 * np.pi * x)
N = 10000

#print("direct ndft:")
#%timeit ndft(x, f, N)
#print()
print("fast nfft:")
get_ipython().magic('timeit nfft(x, f, N)')
print()
print("wrapped C-nfft/pynfft package:")
get_ipython().magic('timeit cnfft(x, f, N)')

