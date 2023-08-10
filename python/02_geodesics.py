import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Add the optrans package to the Python path
# (change the path below to the location of the optimaltransport directory on your computer)
# Note: this step is not necessary if you have installed optimaltransport through pip.
import sys
sys.path.append('../../optimaltransport')

from optrans.utils import signal_to_pdf

# Create a delta function reference
sig0 = np.zeros(128)
sig0[31] = 1.

# Create a delta function signal
sig1 = np.zeros(128)
sig1[95] = 1.

# Convert the reference and signal to PDFs, and smooth them such that they become translated Gaussians
sig0 = signal_to_pdf(sig0, sigma=5.)
sig1 = signal_to_pdf(sig1, sigma=5.)

# Plot the signals
_, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].plot(sig0, 'r-')
ax[0].set_title('Reference: sig0')
ax[1].plot(sig1, 'b-')
ax[1].set_title('Signal: sig1')
plt.show()

# alpha values to plot
alpha = np.linspace(0, 1, 5)

fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(16,4))
for i,a in enumerate(alpha):
    # Interpolation in signal space
    sig_interp = (1. - a) * sig0 + a * sig1
    ax[i].plot(sig_interp)
    ax[i].set_title('alpha =\n{:.2f}'.format(a))
plt.show()

from optrans.continuous import CDT

# Compute CDT of sig1 w.r.t. sig0
cdt = CDT()
sig1_hat = cdt.forward(sig0, sig1)

# Compute identity: x = f + u
x = cdt.transport_map_ + cdt.displacements_

# Plot interpolation in CDT space using same alpha values as before
fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(16,4))
for i,a in enumerate(alpha):
    # Interpolation in CDT space
    fa = x - a * cdt.displacements_
    siga = cdt.apply_inverse_map(fa, sig0)
    ax[i].plot(siga)
    ax[i].set_title('alpha =\n{:.2f}'.format(a))
plt.show()

from optrans.utils import signal_to_pdf

# Create a delta function reference
img0 = np.zeros((128,128))
img0[95,31] = 1.

# Create a delta function image
img1 = np.zeros((128,128))
img1[31,95] = 1.

# Convert the reference and image to PDFs, and smooth them such that they become translated Gaussians
img0 = signal_to_pdf(img0, sigma=5.)
img1 = signal_to_pdf(img1, sigma=5.)

# Plot the images
_, ax = plt.subplots(1, 2)
ax[0].imshow(img0, cmap='gray')
ax[0].set_title('Reference: img0')
ax[1].imshow(img1, cmap='gray')
ax[1].set_title('Image: img1')
plt.show()

# alpha values to plot
alpha = np.linspace(0, 1, 5)

# Minimum and maximum pixel values (for plotting purposes)
vmin = img0.min()
vmax = img0.max()

fig, ax = plt.subplots(1, 5, figsize=(16,4))
for i,a in enumerate(alpha):
    # Interpolation in image space
    img_interp = (1. - a) * img0 + a * img1
    ax[i].imshow(img_interp, cmap='gray', vmin=vmin, vmax=vmax)
    ax[i].set_title('alpha =\n{:.2f}'.format(a))

from optrans.continuous import RadonCDT

# Compute the Radon-CDT of img1 w.r.t. img0
radoncdt = RadonCDT()
img1_hat = radoncdt.forward(img0, img1)

# Compute identity: x = f + u
x = radoncdt.transport_map_ + radoncdt.displacements_

# Plot interpolation in Radon-CDT space using same alpha values as before
fig, ax = plt.subplots(1, 5, figsize=(16,4))
for i,a in enumerate(alpha):
    # Interpolation in Radon-CDT space
    fa = x - a * radoncdt.displacements_
    imga = radoncdt.apply_inverse_map(fa, img0)
    ax[i].imshow(imga, cmap='gray', vmin=vmin, vmax=vmax)
    ax[i].set_title('alpha =\n{:.2f}'.format(a))
plt.show()



