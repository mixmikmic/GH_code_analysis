get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (16, 6)  # make the default figure size larger
matplotlib.rcParams['image.interpolation'] = 'nearest'  # don't blur/smooth image plots
from matplotlib import pyplot as plt
import webbpsf
import webbpsf.wfirst

webbpsf.setup_logging()

wfi = webbpsf.wfirst.WFI()

webbpsf.show_notebook_interface('wfi')

mono_psf = wfi.calc_psf(monochromatic=1.2e-6, display=True)

mono_psf.info()

webbpsf.display_psf(mono_psf, ext='DET_SAMP')

plt.figure(figsize=(8, 6))
webbpsf.display_profiles(mono_psf)

mono_psf.writeto('./mono_psf_1.2um.fits', clobber=True)



