#Download the data (1MB)
from urllib.request import urlretrieve, urlopen
from zipfile import ZipFile
files = urlretrieve("https://www.dropbox.com/s/ecdlgwxjq04m5mx/HyperSpy_demos_EDS_TEM_files.zip?raw=1", "./HyperSpy_demos_EDX_TEM_files.zip")
with ZipFile("HyperSpy_demos_EDX_TEM_files.zip") as z:
    z.extractall()

get_ipython().magic('matplotlib nbagg')
import hyperspy.api as hs

c = hs.load('bare_core.hdf5')
cs = hs.load('core_shell.hdf5')

c.metadata

axes = hs.plot.plot_images(hs.transpose(*(c.get_lines_intensity() + cs.get_lines_intensity())),
                           scalebar='all', axes_decor=None, per_row=2, cmap='RdBu')

cs.change_dtype('float')
cs.decomposition()

ax = cs.plot_explained_variance_ratio()

cs.blind_source_separation(3)

axes = cs.plot_bss_loadings()

axes = cs.plot_bss_factors()

s_bss = cs.get_bss_factors().inav[0]

pt_la = c.get_lines_intensity(['Pt_La'])[0]
mask = pt_la > 6

axes = hs.plot.plot_images(hs.transpose(*(mask, pt_la * mask)), axes_decor=None, colorbar=None,
                           label=['Mask', 'Pt L${\\alpha}$ intensity'], cmap='RdBu')

c_mask = c.sum(-1)
c_mask.data = mask.data

s_bare = (c * c_mask).sum()

s_bare.change_dtype('float')
s = hs.stack([s_bare, s_bss], new_axis_name='Bare or BSS')
s.metadata.General.title = 'Bare or BSS'

axes = hs.plot.plot_spectra(s, style='mosaic', legend=['Bare particles', 'BSS #0'])

w = s.estimate_background_windows()

s.plot(background_windows=w)

w

w[1, 0] = 8.44
w[1, 1] = 8.65

s.plot(background_windows=w, navigator='slider')

sI = s.get_lines_intensity(background_windows=w)

m = s.isig[5.:15.].create_model()

m.add_family_lines(['Cu_Ka', 'Co_Ka'])

m.components

m.plot()

m.multifit()

m.fit_background()

m.calibrate_energy_axis()

m.plot()

sI = m.get_lines_intensity()[-2:]

#From Brucker software (Esprit)
kfactors = [1.450226, 5.075602]

composition = s.quantification(method="CL", intensities=sI, factors=kfactors,
                 plot_result=True)

from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import watershed
from skimage.feature import peak_local_max

distance = distance_transform_edt(mask.data)
local_maxi = peak_local_max(distance, indices=False,
                            min_distance=2, labels=mask.data)
labels = watershed(-distance, markers=label(local_maxi)[0],
                   mask=mask.data)

axes = hs.plot.plot_images(
    [pt_la.T, mask.T, hs.signals.Signal2D(distance), hs.signals.Signal2D(labels)],
    axes_decor='off', per_row=2, colorbar=None, cmap='RdYlBu_r',
    label=['Pt L${\\alpha}$ intensity', 'Mask',
           'Distances', 'Separated particles'])



