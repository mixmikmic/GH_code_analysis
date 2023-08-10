# Import the IO module
import menpo.io as mio
# Import Matplotlib so we can plot subplots
import matplotlib.pyplot as plt

# Import a couple of interesting images that are landmarked!
takeo = mio.import_builtin_asset('takeo.ppm')
takeo = takeo.as_masked()
lenna = mio.import_builtin_asset('lenna.png')
lenna = lenna.as_masked()

get_ipython().magic('matplotlib inline')
takeo = takeo.crop_to_landmarks()
takeo = takeo.constrain_mask_to_landmarks()

plt.subplot(121)
takeo.view_landmarks();
plt.subplot(122)
takeo.mask.view();

get_ipython().magic('matplotlib inline')
lenna = lenna.crop_to_landmarks()
lenna = lenna.constrain_mask_to_landmarks()

plt.subplot(121)
lenna.view_landmarks();
plt.subplot(122)
lenna.mask.view();

from menpo.transform import ThinPlateSplines, PiecewiseAffine

tps_lenna_to_takeo = ThinPlateSplines(lenna.landmarks['LJSON'].lms, takeo.landmarks['PTS'].lms)
pwa_lenna_to_takeo = PiecewiseAffine(lenna.landmarks['LJSON'].lms, takeo.landmarks['PTS'].lms)

tps_takeo_to_lenna = ThinPlateSplines(takeo.landmarks['PTS'].lms, lenna.landmarks['LJSON'].lms)
pwa_takeo_to_lenna = PiecewiseAffine(takeo.landmarks['PTS'].lms, lenna.landmarks['LJSON'].lms)

warped_takeo_to_lenna_pwa = takeo.as_unmasked(copy=False).warp_to_mask(lenna.mask, pwa_lenna_to_takeo)
warped_takeo_to_lenna_tps = takeo.as_unmasked(copy=False).warp_to_mask(lenna.mask, tps_lenna_to_takeo)

get_ipython().magic('matplotlib inline')
# Takeo to Lenna with PWA
warped_takeo_to_lenna_pwa.view();

import numpy as np
np.nanmax(warped_takeo_to_lenna_pwa.pixels) + 1

warped_takeo_to_lenna_pwa.pixels[0,1,1]

get_ipython().magic('matplotlib inline')
# Takeo to Lenna with TPS
warped_takeo_to_lenna_tps.view();

warped_lenna_to_takeo_pwa = lenna.as_unmasked(copy=False).warp_to_mask(takeo.mask, pwa_takeo_to_lenna)
warped_lenna_to_takeo_tps = lenna.as_unmasked(copy=False).warp_to_mask(takeo.mask, pwa_takeo_to_lenna)

get_ipython().magic('matplotlib inline')
# Lenna to Takeo with PWA
warped_lenna_to_takeo_pwa.view();

get_ipython().magic('matplotlib inline')
# Lenna to Takeo with TPS
warped_lenna_to_takeo_tps.view();

