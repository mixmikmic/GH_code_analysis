import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import pyspeckit
from multicube.subcube import SubCube
from multicube.astro_toolbox import make_test_cube, get_ncores
from IPython.utils import io
import warnings
warnings.filterwarnings('ignore')

make_test_cube((300,10,10), outfile='foo.fits', sigma=(10,5))
sc = SubCube('foo.fits')

# TODO: move this to astro_toolbox.py
#       as a general synthetic cube generator routine
def tinker_ppv(arr):
    scale_roll = 15
    rel_shift  = 30
    rel_str    = 5
    shifted_component = np.roll(arr, rel_shift) / rel_str
    for y,x in np.ndindex(arr.shape[1:]):
        roll  = np.sqrt((x-5)**2 + (y-5)**2) * scale_roll
        arr[:,y,x] = np.roll(arr[:,y,x], int(roll))
    return arr + shifted_component
sc.cube = tinker_ppv(sc.cube)

sc.plot_spectrum(3,7)

npeaks = 2
sc.update_model('gaussian')
sc.specfit.fitter.npeaks = npeaks
minpars = [0.1, sc.xarr.min().value, 0.1] + [0.1, -13, 0.5]
maxpars = [2.0, sc.xarr.max().value, 2.0] + [0.5,  -5, 1.5]
finesse = [5, 10, 5] + [5, 20, 5]
sc.make_guess_grid(minpars, maxpars, finesse)

sc.generate_model()
sc.get_snr_map()
sc.best_guess()

sc.fiteach(fittype   = sc.fittype,
           guesses   = sc.best_guesses, 
           multicore = get_ncores(),
           errmap    = sc._rms_map,
           **sc.fiteach_args);

sc.get_modelcube()
import matplotlib.gridspec as gridspec
plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(*sc.cube.shape[1:])
gs.update(wspace=0., hspace=0.)
for ii in range(np.prod(sc.cube.shape[1:])):
    xy = np.unravel_index(ii, dims=sc.cube.shape[1:])
    ax = plt.subplot(gs[ii])
    sc.plot_spectrum(*xy, axis=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_ylim(-0.15, 1.3)
    ax.plot(sc.xarr.value, sc.model_grid[sc._best_map[xy]], label='Initial guess\nat (x,y)={}'.format(xy))
    ax.plot(sc.xarr.value, sc._modelcube[:, xy[1], xy[0]], label='Final fit\nat (x,y)={}'.format(xy))

