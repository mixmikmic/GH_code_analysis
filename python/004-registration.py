

import numpy as np
from dipy.data import get_data
import dipy.align.imwarp as imwarp
from dipy.viz import regtools
get_ipython().magic('matplotlib inline')

fname_moving = get_data('reg_o')
fname_static = get_data('reg_c')

moving = np.load(fname_moving)
static = np.load(fname_static)

fig = regtools.overlay_images(static, moving, 'Static', 'Overlay', 'Moving')
fig.set_size_inches([10, 10])

from dipy.align.metrics import SSDMetric
metric = SSDMetric(static.ndim)

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
level_iters = [200, 100, 50, 25]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter = 50)

mapping = sdr.optimize(static, moving)

warped_forward, warped_back = regtools.plot_2d_diffeomorphic_map(mapping, 10)

warped_moving = mapping.transform(moving)
fig = regtools.overlay_images(static, warped_moving, 'Static','Overlay')
fig.set_size_inches([10, 10])

warped_static = mapping.transform_inverse(static, 'linear')
regtools.overlay_images(warped_static, moving,
                        'Warped static','Overlay','Moving',
                        'inverse_warp_result.png')

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

from dipy.data.fetcher import fetch_syn_data, read_syn_data
fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
syn_b0 = np.array(nib_syn_b0.get_data())

from dipy.segment.mask import median_otsu
stanford_b0_masked, stanford_b0_mask = median_otsu(stanford_b0, 4, 4)
syn_b0_masked, syn_b0_mask = median_otsu(syn_b0, 4, 4)

static = stanford_b0_masked
static_affine = nib_stanford.get_affine()
moving = syn_b0_masked
moving_affine = nib_syn_b0.get_affine()

pre_align = np.array([[1.02783543e+00, -4.83019053e-02, -
                       6.07735639e-02, -2.57654118e+00],
                      [4.34051706e-03, 9.41918267e-01,
                       -2.66525861e-01, 3.23579799e+01],
                      [5.34288908e-02, 2.90262026e-01,
                       9.80820307e-01, -1.46216651e+01],
                      [0.00000000e+00, 0.00000000e+00,
                       0.00000000e+00, 1.00000000e+00]])

from dipy.align.imaffine import AffineMap
affine_map = AffineMap(pre_align,
                       static.shape, static_affine,
                       moving.shape, moving_affine)

resampled = affine_map.transform(moving)

fig = regtools.overlay_slices(static, resampled, None, 1, 'Static', 'Moving')
fig.set_size_inches([20, 20])

from dipy.align.metrics import CCMetric
metric = CCMetric(dim=3)

level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)

warped_moving = mapping.transform(moving)

fig = regtools.overlay_slices(static, warped_moving, None, 1, 'Static')
fig.set_size_inches([20, 20])



