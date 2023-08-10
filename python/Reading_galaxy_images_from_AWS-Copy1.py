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
get_ipython().run_line_magic('inline', 'matplotlib')

filename = 'galaxyzoo2--assets--14-07-2017.csv'
path = '/Users/jegpeek/Dropbox/xorastro/'

data = ascii.read(path + filename, format='csv', fast_reader=False)

response = requests.get(data[90]['location'])
img = Image.open(BytesIO(response.content))

red, green, blue = img.split()

imgcube = np.zeros([424, 424, len(data)])

for i, d in enumerate(data[0:10]):
    response = requests.get(d['location'])
    img = Image.open(BytesIO(response.content))
    red, green, blue = img.split()
    imgcube[:, :, i] = green # confusingly green is SDSS r band

imgcube[:, :, 0]

len(data)

metadata = Table.read(path + "xorastro_metadata_dr7id.fits")

len(metadata)

data.rename_column('name', 'dr7objid')

metadata

joined = join(data, metadata, keys=['dr7objid'])

len(joined)

joined



