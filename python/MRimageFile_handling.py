from medpy.io import load
from glob import glob

dcm_path = '/home/seonwhee/Deep_Learning/Datasets/TCGA-GBM/TCGA-02-0003/1.3.6.1.4.1.14519.5.2.1.1706.4001.145725991542758792340793681239/1.3.6.1.4.1.14519.5.2.1.1706.4001.100169298880243060237139829068/'
dcm_files = glob('%s*dcm' %(dcm_path))
print(dcm_files)

for a_file in dcm_files:
    image_data, image_header = load(a_file)
    print(image_data.shape, image_data.dtype)

from __future__ import print_function
import SimpleITK as sitk
import sys, os

print( "Reading Dicom directory:", dcm_path )

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
reader.SetFileNames(dicom_names)

image = reader.Execute()
size = image.GetSize()
print( "Image size:", size[0], size[1], size[2] )

mha_name = dcm_path + "TCGA-02-0003.mha"
print( "Writing image:", mha_name )
sitk.WriteImage( image, mha_name)
if ( not "SITK_NOSHOW" in os.environ ):
    sitk.Show( image, "Dicom Series" )

import skimage.io as io

def read_image_into_ndArray(imagefile, PlugIn):
    imageArray = io.imread(imagefile, plugin=PlugIn)
    print("The dimension of the image is ", imageArray.shape)
    return imageArray

img = read_image_into_ndArray(mha_name, PlugIn='simpleitk')

