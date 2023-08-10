from matplotlib import pyplot as plt, cm
import numpy as np
import glob, os

get_ipython().magic('matplotlib inline')

#!pip install SimpleITK
import SimpleITK as sitk

DATA_DIR = "/Volumes/data/tonyr/dicom/Lung CT/stage1"
    
patients = glob.glob(os.path.join(DATA_DIR, '*')) # Get the folder names for the patients

if len(patients) == 0:
    raise IOError('Directory ' + DATA_DIR + ' not found or no files found in directory')
    
print('Found the following subfolders with DICOMs: {}'.format(patients))

for patientDirectory in patients[:1]:  # TODO: Go through just one patient for testing. Later we'll use the whole loop
    
    # The next 4 lines set up a SimpleITK DICOM reader pipeline.
    reader = sitk.ImageSeriesReader()
    
    # Now get the names of the DICOM files within the directory
    filenamesDICOM = reader.GetGDCMSeriesFileNames(patientDirectory)
    reader.SetFileNames(filenamesDICOM)
    # Now execute the reader pipeline
    patientObject = reader.Execute()
    

print('This is a {} dimensional image'.format(patientObject.GetDimension()))
print('There are {} slices for this patient.'.format(patientObject.GetDepth()))
print('Each slice is {} H x {} W pixels.'.format(patientObject.GetHeight(), patientObject.GetWidth()))
print('Color depth of the image is {}.'.format(patientObject.GetNumberOfComponentsPerPixel()))
print('The real world size of these pixels (i.e. voxel size) is {} mm H x {} mm W x {} mm D [slice thickness]'.
     format(patientObject.GetSpacing()[0], patientObject.GetSpacing()[1], patientObject.GetSpacing()[2]))
print('The real world origin for these pixels is {} mm. This helps us align different studies.'.format(patientObject.GetOrigin()))

def sitk_show(img, title=None, margin=0.05, dpi=40):
    
    # Here we are just converting the image from sitk to a numpy ndarray
    ndArray = sitk.GetArrayFromImage(img)
    
    # Conversion note:
    # SimpleITK stores things as (x, y, z). numpy ndarray is (z, y, x). So be careful!
    
    spacing = img.GetSpacing() # This returns the realworld size of the pixels (in mm)
    figsize = (1 + margin) * ndArray.shape[0] / dpi, (1 + margin) * ndArray.shape[1] / dpi
    extent = (0, ndArray.shape[1]*spacing[1], ndArray.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap(cm.bone)  # TODO: This might be specific to CT scans(?)
    ax.imshow(ndArray,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()

sitk_show(patientObject[:,:,0], title='Slice#0', dpi=80)  # Display slice 0
sitk_show(patientObject[:,:,100], title='Slice #100', dpi=80) # Display slice 100
sitk_show(patientObject[:,:,250], title='Slice #250', dpi=80) # Display slice 250

import h5py
import ntpath

def getImageTensor(patientDirectory):
    """
    Helper function for injesting all of the DICOM files for one study into a single tensor
    
    input: 'patientDirectory', the directory where the DICOMs for a single patient are stored
    outputs:
            imgTensor = a flattened numpy array (1, C*H*W*D)
            C = number of channels per pixel (1 for MR, CT, and Xray)
            H = number of pixels in height
            W = number of pixels in width
            D = number of pixels in depth
    """
    reader = sitk.ImageSeriesReader()  # Set up the reader object
    
    # Now get the names of the DICOM files within the directory
    filenamesDICOM = reader.GetGDCMSeriesFileNames(patientDirectory)
    reader.SetFileNames(filenamesDICOM)
    # Now execute the reader pipeline
    patientObject = reader.Execute()

    C = patientObject.GetNumberOfComponentsPerPixel() # There is just one color channel in the DICOM for CT and MRI
    H = patientObject.GetHeight()  # Height in pixels
    W = patientObject.GetWidth()   # Width in pixels
    #D = patientObject.GetDepth()  # Depth in pixels
    D = 128   # Let's limit to 128 for now - 
        
    # We need to tranpose the SimpleITK ndarray to the right order for neon
    # Then we need to flatten the array to a single vector (1, C*H*W*D)
    imgTensor = sitk.GetArrayFromImage(patientObject[:,:,:D]).transpose([1, 2, 0]).ravel().reshape(1,-1)
            
    return imgTensor, C, H, W, D

outFilename = 'dicom_out.h5'  # The name of our HDF5 data file

with h5py.File(outFilename, 'w') as df:  # Open hdf5 file for writing our DICOM dataset

    numPatients = len(patients)  # Number of patients in the directory

    for patientDirectory in patients[:1]:  # Start with the first patient to set up the HDF5 dataset

        patientID = ntpath.basename(patientDirectory) # Unique ID for patient

        print('({} of {}): Processing patient: {}'.format(1, numPatients, patientID))

        imgTensor, original_C, original_H, original_W, original_D = getImageTensor(patientDirectory)
        
        dset = df.create_dataset('input', data=imgTensor, maxshape=[None, original_C*original_H*original_W*original_D])
        
        # Now iterate through the remaining patients and append their image tensors to the HDF5 dataset
        
        for i, patientDirectory in enumerate(patients[1:]): # Now append the remaining patients
            
            print('({} of {}): Processing patient: {}'.format(i+2, numPatients, ntpath.basename(patientDirectory)))

            imgTensor, C, H, W, D = getImageTensor(patientDirectory)
            
            # Sanity check
            # Let's make sure that all dimensions are the same. Otherwise, we need to pad (?)
            assert(C == original_C)
            assert(H == original_H)
            assert(W == original_W)
            assert(D == original_D)
            
            # HDF5 allows us to dynamically resize the dataset
            row = dset.shape[0] # How many rows in the dataset currently?
            dset.resize(row+1, axis=0)   # Add one more row (i.e. new patient)
            dset[row, :] = imgTensor  # Append the new row to the dataset
           
        
    # Output attribute 'lshape' to let neon know the shape of the tensor.
    df['input'].attrs['lshape'] = (C, H, W, D) # (Channel, Height, Width, Depth)

    print('FINISHED. Output to HDF5 file: {}'.format(outFilename))

from neon.data import HDF5Iterator  # Neon's HDF5 data loader
from neon.backends import gen_backend

be = gen_backend(backend='cpu', batch_size=1)  

train_set = HDF5Iterator(outFilename)

train_set.get_description()

plt.figure(figsize=[10,10])
plt.title('Slice #100')
plt.imshow(train_set.inp[0,:].reshape(512,512,128)[:,:,100]);  # Show the slice 100 of patient 0

train_set.cleanup()  # Need to close the HDF5 file when we are finished



