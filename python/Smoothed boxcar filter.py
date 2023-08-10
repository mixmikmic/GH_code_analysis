import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def window(n, b, delta):
    """Returns wt, wf which is window in time and freq domains. delta controls error."""
    assert (n % 2) == 1  # For now, assume an odd n.
    n2 = (n - 1) / 2
    kappa = 0.5
    w = (1 - 0.5 * kappa) / b
    c_delta = -np.log(delta)
    sigma_t = (2 * b * np.sqrt(2 * c_delta)) / (np.pi * kappa)
    sigma_f = 1.0 / (2.0 * np.pi * sigma_t)

    # We use [-n2, n2] instead of [0, n-1] to more easily compute the window in time domain.
    t = np.arange(-n2, n2 + 1).astype(float)
    q = t / sigma_t  # Temp variable.
    wt = np.exp(-0.5 * q * q) * (w * np.sinc(t * w))
    # The ifftshift is to move from [-n2, n2] to [0, n-1] in time domain before we can do the FFT.
    # The fftshift is to move from [0, n-1] to [-n2, n2] in frequency domain for better visualization.
    wf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(wt)))
    
    # In time domain, we truncate so that convolution is cheap and scale with b not n.
    p = int(np.ceil(2 * np.sqrt(2 * c_delta) * sigma_t + 1))
    if (p % 2) == 0:
        p += 1
    p2 = (p - 1) / 2
    mask = np.abs(t) <= p2
    wt_truncated = wt.copy()
    wt_truncated[~mask] = 0
    wf_truncated = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(wt_truncated)))
    
    # We also need a closed form for approximating wf. We only care about the region inside the bin.
    
    assert np.sum(np.abs(np.imag(wf))) < 1e-5
    assert np.sum(np.abs(np.imag(wf_truncated))) < 1e-5
    
    # Next, we compute our approximant function in frequency domain.
    xi = 0.5 * t / n2
    tmp = 1.0 / (np.sqrt(2) * sigma_f)
    wf_approx = 0.5 * (erf((xi + 0.5 * w) * tmp) - erf((xi - 0.5 * w) * tmp))
    
    return wt, np.real(wf), wt_truncated, np.real(wf_truncated), wf_approx, t, xi

def experiment():
    n = 2887  # Prime. Not needed here.
    b = 5
    wt, wf, wt_truncated, wf_truncated, wf_approx, t, xi = window(n, b, 1e-8)
    
    f, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(t, wt)
    axs[0].set_title('Original filter')
    axs[0].set_xlabel('time')
    axs[1].plot(t, wt_truncated)
    axs[1].set_title('Time-truncated filter')
    axs[1].set_xlabel('time')
    plt.show()
    
    f, axs = plt.subplots(1, 3, figsize=(17, 6))
    axs[0].plot(xi, wf)
    axs[0].set_title('Original filter')
    axs[0].set_xlabel('freq')
    axs[1].plot(xi, wf_truncated)
    axs[1].set_title('Time-truncated filter')
    axs[1].set_xlabel('freq')
    axs[2].plot(xi, wf_approx)
    axs[2].set_title('Time-truncated filter')
    axs[2].set_xlabel('freq')
    plt.show()
    
    err = np.sum(np.abs(wf - wf_truncated)) / n
    print 'Mean L1 error for time-truncated: %e' % err
    
    err = np.sum(np.abs(wf - wf_approx)) / n
    print 'Mean L1 error for approximant: %e' % err

    
experiment()

