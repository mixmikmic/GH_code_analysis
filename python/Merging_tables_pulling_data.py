import astropy.io.ascii as ascii
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import astropy.visualization as viz
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as WCS # you need this astropy module for processing fits files
import matplotlib as mpl
import numpy as np
import tqdm
from astropy.table import Table, join
get_ipython().run_line_magic('matplotlib', 'inline')

filename = 'galaxyzoo2--assets--14-07-2017.csv'
path = '/Users/jegpeek/Dropbox/xorastro/'
data = ascii.read(path + filename, format='csv', fast_reader=False)

metadata = Table.read(path + "xorastro_metadata_dr7id.fits")

data.rename_column('name', 'dr7objid')

joined = join(data, metadata, keys=['dr7objid'])

plt.hist(joined['ra_1']-joined['ra_2'], range=[-0.0001, 0.0001])

plt.hist(joined['dec_1']-joined['dec_2'], range=[-0.0001, 0.0001])

joined[100]['location']

imgcube = np.zeros([424, 424, len(joined)])
for i, d in enumerate(joined):
    response = requests.get(d['location'])
    img = Image.open(BytesIO(response.content))
    red, green, blue = img.split()
    imgcube[:, :, i] = green # confusingly green is SDSS r band

plt.imshow(imgcube[:, :, 0], cmap='Greys')
plt.show()

np.save('imgcube.npy', imgcube)

np.save('top10k_meta.np', joined[0:10000])

joined10k = joined[0:10000]

joined10k.write('top10k_meta', format='ascii.html')

get_ipython().run_line_magic('pinfo', 'Table.read')

plt.hist(joined10k[:]['g']-joined10k[:]['i'], bins=100, range=[0, 2])
plt.yscale('log')
plt.show()

color = joined10k[:]['g']-joined10k[:]['i']
np.sum((color > 0.6) & (color < 1.7))



