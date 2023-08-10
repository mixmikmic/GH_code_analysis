get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread   # required for QA track
from scipy.ndimage import gaussian_filter

from PIL import Image as pImage
import dicom

from pylinac.flatsym import BeamImage
from pylinac.core import image as Image
from pylinac.core.image import DicomImage, FileImage, ArrayImage

my_img = BeamImage("flatsym_demo.dcm")
flatness_test = my_img.plot_flatness()
flatness_test[0].get_figure().savefig('test.png')    # save the figure



my_dcm = dicom.read_file("flatsym_demo.dcm")   # get a file object as in QAtrack

plt.imshow(my_dcm.pixel_array)

pylinac_img = Image.load(my_dcm.pixel_array)     # load into a pylinac image

pylinac_img.plot()



