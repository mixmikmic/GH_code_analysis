# load our tools
get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('matplotlib', 'inline')
from numba import njit, jit
from dphplotting import display_grid
# from accelerate import cuda, profiler
import numba
import numba.cuda

# we've changed the following file so that we only have the "matlab" implementation.
# %load ../decon.py
#!/usr/bin/env python
# decon.py
"""
Functions that actually perform the deconvolution.

Copyright (c) 2016, David Hoffman
"""

import numpy as np
try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import (fftshift, ifftshift, fftn, ifftn,
                                             rfftn, irfftn)
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import (fftshift, ifftshift, fftn, ifftn,
                           rfftn, irfftn)
from dphutils import fft_pad
from pyDecon.utils import _prep_img_and_psf, _ensure_positive, _zero2eps
import scipy.signal.signaltools as sig
from scipy.signal import fftconvolve
from scipy.ndimage import convolve


def _get_fshape_slice(image, psf):
    """This is necessary for the fast Richardson-Lucy Algorithm"""
    s1 = np.array(image.shape)
    s2 = np.array(psf.shape)
    assert (s1 >= s2).all()
    shape = s1 + s2 - 1
    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [sig.fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    return fshape, fslice

def _rl_core_matlab(image, otf, psf, u_t, **kwargs):
    """The core update step of the RL algorithm

    This is a fast but inaccurate version modeled on matlab's version
    One improvement is to pad everything out when the shape isn't
    good for fft."""
    reblur = irfftn(otf * rfftn(u_t, u_t.shape, **kwargs), u_t.shape, **kwargs)
    reblur = _zero2eps(reblur)
    im_ratio = image / reblur  # _zero2eps(array(0.0))
    estimate = irfftn(np.conj(otf) * rfftn(im_ratio, im_ratio.shape, **kwargs),
                      im_ratio.shape, **kwargs)
    # The below is to compensate for the slight shift that using np.conj
    # can introduce verus actually reversing the PSF. See notebooks for
    # details.
    for i, (s, p) in enumerate(zip(image.shape, psf.shape)):
        if s % 2 and not p % 2:
            estimate = np.roll(estimate, 1, i)
    estimate = _ensure_positive(estimate)
    return u_t * estimate  # / (1 + np.sqrt(np.finfo(u_t.dtype).eps))

eps = np.finfo(float).eps

def _rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration

    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766."""
    # TODO: everything here can be wrapped in ne.evaluate
    alpha = (g_tm1 * g_tm2).sum() / ((g_tm2**2).sum() +
                                         eps)
    alpha = max(min(alpha, 1), 0)
    # if alpha is positive calculate predicted step
    if alpha:
        # first order correction
        h1_t = u_t - u_tm1
        u_tp1 = u_t + alpha * h1_t
        if prediction_order > 1:
            # second order correction
            h2_t = (u_t - 2 * u_tm1 + u_tm2)
            u_tp1 = u_tp1 + alpha**2 / 2 * h2_t
        return u_tp1
    else:
        return u_t


def richardson_lucy_cpu(image, psf, iterations=10, prediction_order=1, init="matlab", **kwargs):
    """
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function. Assumes that it has no background
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    prediction_order : int (0, 1 or 2)
        Use Biggs-Andrews to accelerate the algorithm [2] default is 1
    core_type : str
        Type of core to use (see Notes)
    init : str
        How to initialize the deconvolution (see Notes)

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------

    Notes
    -----
    This algorithm can use a variety of cores to calculate the update step [1].
    The update step basically consists of two convolutions which can be
    performed in a few different ways. The "direct" core uses direct
    convolution to calculate them and is painfully slow but is included for
    completeness. The "accurate" core uses `fftconvolve` to properly and
    quickly calculate the convolution. "fast" optimizes "accurate" by only
    performing the FFTs on the PSF _once_. The "matlab" core is based on the
    MatLab implementation of Richardson-Lucy which speeds up the convolution
    steps by avoiding padding out both data and psf to avoid the wrap around
    effect of the fftconvolve. This approach is generally accurate because the
    psf is generally very small compared to data but will not give exactly the
    same results as "accurate" or "fast" and should not be used for large or
    complicated PSFs. The "matlab" routine also avoids the tapering effect
    implicit in fftconvolve and should be used whenever any of the image
    extends to the edge of data.

    The algorithm can also be initilized in two ways: using the original image
    ("matlab") or using a constant array the same size of the image filled with
    the mean value of the image ("mean"). The advantage to starting with the
    image is that you need fewer iterations to get a good result. However,
    the disadvantage is that if there original image has low SNR the result
    will be severly degraded. A good rule of thumb is that if the SNR is low
    (SNR < 10) that `init="mean"` be used and double the number of iterations.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766.

    """
    # TODO: Make sure that data is properly padded for fast FFT numbers.
    # checked against matlab on 20160805 and agrees to within machine precision
    image, psf = _prep_img_and_psf(image, psf)
    # choose core
    core = _rl_core_matlab
    # set up the proper dict for the right core
    if core is _rl_core_matlab:
        if psf.shape != image.shape:
            # its been assumed that the background of the psf has already been
            # removed and that the psf has already been centered
            psf = fft_pad(psf, image.shape, mode='constant')
        otf = rfftn(ifftshift(psf))
        core_dict = dict(image=image, otf=otf, psf=psf)
    else:
        raise RuntimeError("{} is not a valid core".format(core))
    # initialize variable for iterations
    # previous estimate
    u_tm1 = u_tm2 = None
    if init == "matlab":
        core_dict["u_t"] = u_t = image
    else:
        # current estimate, for the initial estimate we use the mean of the
        # data this promotes a smooth solution and helps to reduce noise.
        core_dict["u_t"] = u_t = np.ones_like(image) * image.mean()
    # previous difference
    g_tm1 = g_tm2 = None
    for i in range(iterations):
        # if prediction is requested perform it
        if prediction_order and g_tm2 is not None and u_tm2 is not None:
            # need to save prediction as intermediate value
            y = _rl_accelerate(g_tm1, g_tm2, u_t, u_tm1, u_tm2,
                               prediction_order)
        else:
            y = u_t
        # update estimate and ensure positive
        core_dict["u_t"] = _ensure_positive(y)
        # call the update function
        u_tp1 = core(**core_dict, **kwargs)
        # update
        # update g's
        g_tm2 = g_tm1
        # this is where the magic is, we need to compute from previous step
        # which may have been augmented by acceleration
        g_tm1 = u_tp1 - y
        # now move u's along
        # Here we don't want to update with accelerated version.
        # why not? is this a mistake?
        u_t, u_tm1, u_tm2 = u_tp1, u_t, u_tm1
        
    # return everything
    return dict(u_t=u_t, u_tm1=u_tm1, u_tm2=u_tm2, g_tm1=g_tm1, g_tm2=g_tm2)

# this one has been changed too, so that it doesn't automatically save the generated data
# %load "../fixtures/2D Test Fake Data/gen_rl_example.py"
#!/usr/bin/env python
# gen_rl_example.py
"""
A short script to generate and output example data for the RL algorithm

Copyright (c) 2016, David Hoffman
"""


def gen_data():
    print("Generating the data ...")
    x = np.linspace(-2.5, 2.5, 64, True)
    kernel = np.exp(-x**2)
    kernel = kernel[np.newaxis] * kernel[:, np.newaxis]
    # normalize kernel
    k_norm = kernel / kernel.sum()
    # make signal
    x = np.linspace(-10, 10, 1024)
    signal = 5.0 * np.logical_and(x < 3, x > -3)
    signal = signal[np.newaxis] * signal[:, np.newaxis]
    blurred = convolve(signal, k_norm, mode="reflect")
    blurred = _ensure_positive(blurred)
    kernel = _ensure_positive(kernel)
    # save ground truth images
    # add some noise to both
    print("Add noise ...")
    np.random.seed(12345)
    blurred[blurred < 0] = 0
    blurred_noisy = np.random.poisson(blurred) + np.random.randn(*blurred.shape) ** 2
    kernel[kernel < 0] = 0
    psf = np.random.poisson(kernel * 100) + np.random.randn(*kernel.shape) ** 2
    psf /= psf.sum()
    return signal, blurred, kernel, blurred_noisy, psf

# make the shitty data
signal, blurred, kernel, blurred_noisy, psf = gen_data()

# test run
decon = richardson_lucy_cpu(blurred_noisy, psf, init="mean")

# display for comparison purposes
display_grid(dict(decon=decon["u_t"], blurred_noisy=blurred_noisy, psf=psf), figsize=5)

# profile run
get_ipython().run_line_magic('lprun', '-f richardson_lucy_cpu -f _rl_accelerate -f _rl_core_matlab richardson_lucy_cpu(blurred_noisy, psf, init="mean")')

# jitting this function doesn't seem to work well or speed up the calculation.
@njit
def _rl_accelerate_njit(g_tm1, g_tm2, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration

    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766."""
    # TODO: everything here can be wrapped in ne.evaluate
    alpha = (g_tm1 * g_tm2).sum() / ((g_tm2**2).sum() +
                                         eps)
    alpha2 = max(min(alpha, 1), 0)
    # if alpha is positive calculate predicted step
    if alpha2:
        # first order correction
        h1_t = u_t - u_tm1
        u_tp1 = u_t + alpha2 * h1_t
        if prediction_order > 1:
            # second order correction
            h2_t = (u_t - 2 * u_tm1 + u_tm2)
            u_tp1 = u_tp1 + alpha2**2 / 2 * h2_t
        return u_tp1
    else:
        return u_t

np.allclose(_rl_accelerate(**decon, prediction_order=1), _rl_accelerate_njit(**decon, prediction_order=1))

((_rl_accelerate(**decon, prediction_order=1)- _rl_accelerate_njit(**decon, prediction_order=1))**2).sum()

get_ipython().run_line_magic('timeit', '_rl_accelerate(**decon, prediction_order=1)')
get_ipython().run_line_magic('timeit', '_rl_accelerate_njit(**decon, prediction_order=1)')

_rl_accelerate_njit.inspect_types()

# jitting this function doesn't seem to work well or speed up the calculation.
@njit
def _alpha_calc(g_tm1, g_tm2):
    numerator = 0.0
    denominator = 0.0
    g1 = g_tm1.ravel()
    g2 = g_tm2.ravel()
    for i in range(g1.size):
        numerator += g1[i] * g2[i]
        denominator += g2[i] ** 2
    return numerator/(denominator + eps)

@njit
def _rl_accelerate2(g_tm1, g_tm2, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration

    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766."""
    # TODO: everything here can be wrapped in ne.evaluate
    alpha = (g_tm1 * g_tm2).sum() / ((g_tm2**2).sum() + eps)
    alpha = max(min(alpha, 1.0), 0.0)
    # if alpha is positive calculate predicted step
    if alpha:
        # first order correction
        u_tp1 = np.empty_like(u_t)
        for j in range(u_tp1.shape[0]):
            for i in range(u_tp1.shape[1]):
                h1_t = u_t[j, i] - u_tm1[j, i]
                u_tp1[j, i] = u_t[j, i] + alpha * h1_t
                if prediction_order > 1:
                    # second order correction
                    h2_t = (u_t[j, i] - 2 * u_tm1[j, i] + u_tm2[j, i])
                    u_tp1[j, i] += alpha**2 / 2 * h2_t
        return u_tp1
    else:
        return u_t

get_ipython().run_line_magic('timeit', '_rl_accelerate(**decon, prediction_order=1)')
get_ipython().run_line_magic('timeit', '_rl_accelerate_njit(**decon, prediction_order=1)')
get_ipython().run_line_magic('timeit', '_rl_accelerate2(**decon, prediction_order=1)')

np.allclose(_rl_accelerate(**decon, prediction_order=1), _rl_accelerate2(**decon, prediction_order=1))

_rl_accelerate2.inspect_types()

@numba.vectorize(["float32(float32, float32)", "float64(float64, float64)"], target="cuda")
def add_gpu(a, b):
    return a + b

@numba.vectorize(["float32(float32, float32)"], target="cuda")
def mult_gpu(a, b):
    return a * b

@numba.cuda.reduce
def sum_gpu(a, b):
    return a + b

@numba.cuda.jit
def mult_gpu_2d(a, b, c):
    x, y = numba.cuda.grid(2)
    if x < c.shape[0] and y < c.shape[1]:
        c[x, y] = a[x, y] * b[x, y]
        
@numba.cuda.jit
def add_gpu_2d(a, b, c):
    x, y = numba.cuda.grid(2)
    if x < c.shape[0] and y < c.shape[1]:
        c[x, y] = a[x, y] + b[x, y]

# excplicit cuda kernels are faster.
dg_tm1 = numba.cuda.to_device(decon["g_tm1"].astype(np.float32))
dg_tm2 = numba.cuda.to_device(decon["g_tm2"].astype(np.float32))
a = numba.cuda.device_array_like(dg_tm1)
print("numpy/cpu version", end="...")
get_ipython().run_line_magic('timeit', '(decon["g_tm1"] * decon["g_tm2"])')
print("mult_gpu version", end="...")
get_ipython().run_line_magic('timeit', 'b = mult_gpu(dg_tm1, dg_tm2)')
print("mult_gpu version raveled", end="...")
get_ipython().run_line_magic('timeit', 'b = mult_gpu(dg_tm1.ravel(), dg_tm2.ravel())')
print("mult_gpu_2d version", end="...")
get_ipython().run_line_magic('timeit', 'mult_gpu_2d[(16, 16), (32, 32)](dg_tm1, dg_tm2, a)')

print("numpy/cpu version", end="...")
get_ipython().run_line_magic('timeit', 'decon["g_tm1"].sum()')
print("mult_gpu.reduce version", end="...")
get_ipython().run_line_magic('timeit', 'mult_gpu.reduce(dg_tm1.ravel())')
print("sum_gpu version", end="...")
get_ipython().run_line_magic('timeit', 'sum_gpu(dg_tm1.ravel())')

get_ipython().run_line_magic('timeit', 'dg_tm1.copy_to_host().sum()')

def _rl_accelerate_gpu(g_tm1, g_tm2):#, u_t, u_tm1, u_tm2, prediction_order):
    """Biggs-Andrews Acceleration

    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766."""
    # TODO: everything here can be wrapped in ne.evaluate
    alpha = add_gpu.reduce(mult_gpu(g_tm1, g_tm2).ravel()) / (add_gpu.reduce(mult_gpu(g_tm2, g_tm2).ravel()))
#     alpha = max(min(alpha, 1), 0)
#     # if alpha is positive calculate predicted step
#     if alpha:
#         # first order correction
#         h1_t = u_t - u_tm1
#         u_tp1 = u_t + alpha * h1_t
#         if prediction_order > 1:
#             # second order correction
#             h2_t = (u_t - 2 * u_tm1 + u_tm2)
#             u_tp1 = u_tp1 + alpha**2 / 2 * h2_t
#         return u_tp1
#     else:
#         return u_t
    return alpha

# very rough beginnings of an actual core, need to check the convolve example for tips.
def _rl_core_matlab(image, otf, psf, u_t, **kwargs):
    """The core update step of the RL algorithm

    This is a fast but inaccurate version modeled on matlab's version
    One improvement is to pad everything out when the shape isn't
    good for fft."""
    # forward fft for image
    fft_u_t = cuda.fft.fft(u_t, u_t.shape)
    # kernel function to multiply two images
    fft_reblur = mult_gpu(otf, fft_u_t)
    # inverse fft
    reblur = cuda.fft.ifft(fft_reblur, u_t.shape, **kwargs)
    # add eps on gpu
    reblur = _zero2eps_gpu(reblur)
    im_ratio = image / reblur  # _zero2eps(array(0.0))
    estimate = irfftn(np.conj(otf) * rfftn(im_ratio, im_ratio.shape, **kwargs),
                      im_ratio.shape, **kwargs)
    # The below is to compensate for the slight shift that using np.conj
    # can introduce verus actually reversing the PSF. See notebooks for
    # details.
    for i, (s, p) in enumerate(zip(image.shape, psf.shape)):
        if s % 2 and not p % 2:
            estimate = np.roll(estimate, 1, i)
    estimate = _ensure_positive(estimate)
    return u_t * estimate  # / (1 + np.sqrt(np.finfo(u_t.dtype).eps))

