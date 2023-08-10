import urllib
import SimpleITK as sitk
import tarfile
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#  Can take a while, the file is about 56MB
if not os.path.exists('PublicBrain001.anonymized.dcm.tgz'):
    publicbrain = urllib.URLopener()
    publicbrain.retrieve('http://www.insight-journal.org/midas/bitstream/view/375', 
                         'PublicBrain001.anonymized.dcm.tgz')

if not os.path.exists('./anonymized'):
    tar = tarfile.open('PublicBrain001.anonymized.dcm.tgz')
    tar.extractall()
    tar.close()

def read_dicom_series(folder):
    """Read a folder with DICOM files and outputs a SimpleITK image.
    Assumes that there is only one DICOM series in the folder.

    Parameters
    ----------
    folder : string
      Full path to folder with dicom files.

    Returns
    -------
    SimpleITK image.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder.encode('ascii'))
    #  There are multiple series_ids in the folder, but after experimenting a bit, 
    # we pick the second one, this is [1] below.
    filenames = reader.GetGDCMSeriesFileNames(folder, series_ids[1],
                                              False,  # useSeriesDetails
                                              False,  # recursive
                                              True)  # load sequences
    reader.SetFileNames(filenames)
    image = reader.Execute()

    return image

image = read_dicom_series('./anonymized/')

image.GetSize()

image.GetSpacing()

array = sitk.GetArrayFromImage(image)

array.shape

fig, axs = plt.subplots(4,5, figsize=(15, 12))
axs = axs.ravel()
for i in range(20):
    axs[i].imshow(array[i],cmap='gray')
    axs[i].set_title('slice {}'.format(i))



