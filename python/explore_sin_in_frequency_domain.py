get_ipython().magic('pylab inline')

import numpy as np
from librosa.core import stft, istft

from play import play

# freq = 1981
freq = 2000
gensin = np.sin(np.linspace(0, freq*np.pi, 44100))
# gensin *= (2**15-1)

# gensin = np.array(gensin, dtype='int16')

plot(gensin[:2000])

play(gensin)

dsin = stft(gensin, n_fft=2048, hop_length=2048)

def imshowsq(m):
    """ A helper for showing spectrograms forces to square aspect ratio with no interpolation """
    imshow(m, aspect=float(m.shape[1]) / m.shape[0], interpolation='none')
    colorbar()

binlo = 38
binhi = 54

imshowsq(dsin.real[binlo:binhi,:])
title('bins {} through {} of real part of frequency domain'.format(binlo, binhi))

imshowsq(dsin.imag[binlo:binhi,:])
title('bins {} through {} of imaginary part of frequency domain'.format(binlo, binhi))

plot(dsin.imag[binlo:binhi,10])
plot(dsin.imag[binlo:binhi,11])
plot(dsin.imag[binlo:binhi,12])
plot(dsin.imag[binlo:binhi,13])
title('a few slices of imaginary space at a few particular times')

imshowsq(abs(dsin)[binlo:binhi,:])
title('absolute value (maginude complex vector) of frequency domain')

imshowsq(np.angle(dsin)[binlo:binhi,:])
title('angle of frequency domain')

plot(np.abs(dsin)[38:54,11])
title('slice in time of magnitude')

plot(np.angle(dsin)[38:54,11])
title('same time slice of angle')

plot(np.sum(abs(dsin), axis=1))
title('sum of magnitudes over time')
xlim(binlo-20,binhi+20)

plot(np.sum(dsin.real, axis=1))
title('sum of real over time')
xlim(binlo-20,binhi+20)

# this appears to be some beat frequency combining sin frequency and fft window frequency
plot(dsin.imag[46,:])
title('imaginary value at target frequency bin over time')

plot(dsin.real[46,:])
title('real value at target frequency bin over time')

plot(np.abs(dsin[46,:]))
title('maginude at target frequency bin over time')

plot(np.angle(dsin)[46,:])
title('angle at target frequency bin over time')

np.fft.fftfreq(dsin.shape[0])*44100

# given number of bins in freq space in D and sample rate, lookup
# frequence of sound in bin 46 which matches ~2khz sin wave
# generated above
(np.fft.fftfreq(dsin.shape[0])*44100)[46]

# TODO: now show same plots, but with a different frequency band closer to 1979
# TODO: any way to make these plots change as a slider changes the frequency

