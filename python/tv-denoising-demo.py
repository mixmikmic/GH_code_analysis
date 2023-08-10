get_ipython().magic('matplotlib inline')
import numpy as np
import scipy 
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

lena = data.astronaut()
plt.imshow(lena)

lena_cl = lena
#lena_cl = lena[100:300, 100:320, 1]

noisy = lena_cl.copy() + 20.0* lena.std() * np.random.random(lena_cl.shape)
noisy -= np.min(noisy)
noisy /= np.max(noisy)
#noisy = np.clip(noisy, 0, 1)
plt.gray()
plt.imshow(noisy)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('noisy')
ax[1].imshow(denoise_tv_chambolle(noisy, weight=100, multichannel=True))
ax[1].axis('off')
ax[1].set_title('TV')
fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

from skimage import filters
img_edges = filters.sobel(lena[:, :, 2])
plt.imshow(img_edges) 

from IPython.core.display import HTML
def css_styling():
    styles = open("./styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

