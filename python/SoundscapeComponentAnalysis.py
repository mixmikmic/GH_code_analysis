from pylab import * # numpy, matplotlib, plt
from bregman.suite import * # Bregman audio feature extraction library
from soundscapeecology import * # 2D time-frequency shift-invariant convolutive matrix factorization
get_ipython().magic('matplotlib inline')
rcParams['figure.figsize'] = (15.0, 9.0)

sound_path = 'sounds'
sounds = os.listdir(sound_path)
print "sounds:", sounds

N=4096; H=N/4
x,sr,fmt = wavread(os.path.join(sound_path,sounds[0]))
print "sample_rate:", sr, "(Hz), fft size:", (1000*N)/sr, "(ms), hop size:", (1000*H)/sr, "(ms)"

# 1. Instantiate a new SoundscapeEcololgy object using the spectral analysis parameters defined above
S = SoundscapeEcology(nfft=N, wfft=N/2, nhop=H)
# Inspect the contents of this object
print S.__dict__

# 2. load_audio() - sample segments of the soundfile without replacement, to speed up analysis
# The computational complexity of the analysis is high, and the information in a soundscape is largely redundant
# So, draw 25 random segments in time order, each consisting of 20 STFT frames (~500ms) of audio data

S.load_audio(os.path.join(sound_path,sounds[0]), num_samples=25, frames_per_sample=20) # num_samples=None means analyze the whole sound file

# 3. analyze() into shift-invariant kernels 
# The STFT spectrum will be converted to a constant-Q transform by averaging over logarithmically spaced bins
# The shift-invariant kernels will have shift and time-extent dimensions
# The default kernel shape yields 1-octave of shift (self.feature_params['nbpo']), 
# and its duration is frames_per_sample. Here, the num_components and win parameters are illustrated.

S.analyze(num_components=7, win=(S.feature_params['nbpo'], S.frames_per_sample))

# 4. visualize() - visualize the spectrum reconstruction and the individual components
# inputs:
#    plotXi - visualize individual reconstructed component spectra [True]
#    plotX - visualize original (pre-analysis) spectrum and reconstruction [False]
#    plotW - visualize component time-frequency kernels [False]
#    plotH - visualize component shift-time activation functions [False]
#    **pargs - plotting key word arguments [**self.plt_args]

S.visualize(plotX=True, plotXi=True, plotW=True, plotH=True)

# 5. resynthesize() - sonify the results
# First, listen to the original (inverse STFT) and the full component reconstruction (inverse CQFT with random phases)
x_orig = S.F.inverse(S.X)
x_recon = S.F.inverse(S.X_hat, Phi_hat=(np.random.rand(*S.F.STFT.shape)*2-1)*np.pi) # random phase reconstruction
play(balance_signal(x_orig))
play(balance_signal(x_recon))

# First, listen to the original (inverse CQFT with original phases in STFT reconstruction) 
# and the all-components reconstruction (inverse CQFT with random phases)
# Second, listen to the individual component reconstructions
# Use the notebook's "interrupt kernel" button (stop button) if this is too long (n_comps x audio sequence)
# See above plots for the individual component spectrograms

for k in range(S.n_components):
    x_hat = S.resynthesize(k) # resynthesize individual component
    play(balance_signal(x_hat)) # play it back





