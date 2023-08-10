import numpy as np
from numpy.fft import *
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

plt.set_cmap("inferno")

sim_size = 128
# make kernel
kernel = np.zeros((sim_size, sim_size))
kernel[sim_size // 2 - 2:sim_size // 2 + 3, sim_size // 2 - 2:sim_size // 2 + 3] = 1
# make data
data = np.random.randn(sim_size, sim_size)
plt.matshow(kernel)
plt.matshow(data)

# do the fourier convolution, "matlab" style
k_kernel = rfftn(ifftshift(kernel), data.shape)
k_data = rfftn(data, data.shape)
convolve_data0 = irfftn(k_kernel * k_data, data.shape)
plt.matshow(convolve_data0)

# make sure that the kernel is placed in the right place (imaginary part should be zero)
plt.matshow(k_kernel.real)
plt.colorbar()
plt.matshow(k_kernel.imag)
plt.colorbar()

# check that real space convolution works as expected
np.allclose(convolve(data, np.ones((5, 5))), convolve(data, kernel))

# check reflection mode
convolve_data_reflect = convolve(data, np.ones((5, 5)))
plt.matshow(convolve_data_realspace - convolve_data0)
np.allclose(convolve_data_realspace, convolve_data0)

# check wrap mode
convolve_data_wrap = convolve(data, np.ones((5, 5)), mode="wrap")
plt.matshow(convolve_data_wrap - convolve_data0)
plt.colorbar()
np.allclose(convolve_data_wrap, convolve_data0)

# scipy's FFT convolution doesn't work quite the same way, it will pad
# out the data first with zeros so that the convolution doesn't wrap
# around, this leads to some shifting.
convolve_data_sp = fftconvolve(data, np.ones((5, 5)), "same")
plt.matshow(convolve_data_sp - convolve_data0)
plt.colorbar()
np.allclose(convolve_data_sp, convolve_data0)

# note that if we had used a kernel (which was already fft_padded) we'd
# have to reverse it because of the way fftconvolve pads the data internally
convolve_data_sp = fftconvolve(data, kernel[::-1, ::-1], "same")
plt.matshow(convolve_data_sp - convolve_data0)
plt.colorbar()
np.allclose(convolve_data_sp, convolve_data0)

# But if we pad with zeros then the convolutions agree
convolve_data_zeros = convolve(data, np.ones((5, 5)), mode="constant")
plt.matshow(convolve_data_zeros - convolve_data_sp)
plt.colorbar()
np.allclose(convolve_data_zeros, convolve_data_sp)

# need new data for this
from skimage.draw import circle_perimeter
data = np.zeros((sim_size, sim_size))
data[circle_perimeter(sim_size//2, sim_size//2, sim_size//4)] = 1

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
axs = axs.ravel()

k_kernel = rfftn(ifftshift(kernel), data.shape)
k_data = rfftn(data, data.shape)
convolve_data0 = irfftn(k_kernel * k_data, data.shape)
axs[0].matshow(convolve_data0)
axs[0].set_title("Matlab Method")

k_kernel = rfftn(kernel, data.shape)
k_data = rfftn(data, data.shape)
convolve_data1 = irfftn(k_kernel * k_data, data.shape)
axs[1].matshow(convolve_data1)
axs[1].set_title("No shifting")

k_kernel = rfftn(kernel, data.shape)
k_data = rfftn(ifftshift(data), data.shape)
convolve_data2 = irfftn(k_kernel * k_data, data.shape)
axs[2].matshow(convolve_data2)
axs[2].set_title("Shift Data")

k_kernel = rfftn(kernel, data.shape)
k_data = rfftn(data, data.shape)
convolve_data3 = irfftn(fftshift(k_kernel * k_data), data.shape)
axs[3].matshow(convolve_data3)
axs[3].set_title("Shift product")

k_kernel = rfftn(kernel, data.shape)
k_data = rfftn(data, data.shape)
convolve_data4 = irfftn(fftshift(k_kernel) * k_data, data.shape)
axs[4].matshow(convolve_data4)
axs[4].set_title("Shift k_kernel")

k_kernel = rfftn(ifftshift(kernel), data.shape)
k_data = rfftn(ifftshift(data), data.shape)
convolve_data5 = fftshift(irfftn(k_kernel * k_data, data.shape))
axs[5].matshow(convolve_data5)
axs[5].set_title("fftshift result")



