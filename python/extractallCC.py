import h5py
from scipy import ndimage
import matplotlib.pylab as plt
import numpy as np
import sys

sys.path.append('../')
get_ipython().magic('matplotlib inline')

def shingler(original_line, shingle_dim=(120,120)):

    # Pull shingle from the line
    # TODO: pull out shingle_dim[n] into two holder variables
    (height, width) = original_line.shape
    max_x = max(width - shingle_dim[1], 1)
    max_y = max(height - shingle_dim[0], 1)
    x_start = np.random.randint(0, max_x)
    y_start = np.random.randint(0, max_y)
    # check if the line is too small on at least one axis
    if width < shingle_dim[1]:
        x_slice = slice(0,width)
    else:
        x_slice = slice(x_start, x_start+shingle_dim[1])
    if  height < shingle_dim[0]: 
        y_slice = slice(0,height)
    else:
        y_slice = slice(y_start, y_start+shingle_dim[1])
    slice_width = x_slice.stop - x_slice.start
    slice_height = y_slice.stop - y_slice.start
    # create an output shingle, copy our thing onto it
    output_arr = np.zeros(shingle_dim)
    output_arr.fill(255)
    output_arr[:slice_height,:slice_width] = original_line[y_slice, x_slice]
    return output_arr

# Calculate the connected components
def connectedcomponents( im ):
    im = im.value
    if im.max()==1:
        im = 255*(1-im)
    im = im < 128
    return ndimage.label(im > 0.5)

# Threshold connected components based on number of pixels
def thresholdcc( ccis, minthresh=500 ):
    ccs = []
    for i in xrange(1,ccis[1]):
        if np.array(ccis[0]==i).sum() > minthresh:
            ccs+=[i]
    return ccs

def shinglesfromcc( ccis, minthresh=250, maxthresh=2000, shingle_dim=(56,56) ):
    ccs = []
    for i in xrange(1,ccis[1]):
        energy = np.array(ccis[0]==i).sum()
        if energy > minthresh and energy < maxthresh:
            ii = np.where( ccis[0] == i )
            xb = ii[0].min()
            yb = ii[1].min()
            xe = ii[0].max()
            ye = ii[1].max()
            ccs += [ shingler( ccis[0][xb:xe, yb:ye], shingle_dim=shingle_dim ) ]
    print "Finished finding "+str(len(ccs))+" features from image."
    return np.expand_dims( np.array( ccs ), 1 )
            

from globalclassify.fielutil import load_verbatimnet, load_minifielnet, denoise_conv4p120_model, load_fiel120
from denoiser.noisenet import conv4p56_model

shingle_dim = (120,120)
if shingle_dim[0]==56:
    featext  = load_verbatimnet('fc7', paramsfile='/fileserver/iam/iam-processed/models/fiel_657.hdf5')
    featext.compile(loss='mse', optimizer='sgd')
    denoiser = conv4p56_model()
    denoiser.load_weights('/work/models/conv4p_linet56-iambin-tifs.hdf5')
    
else: # inputshape is 120
    ### feature extractor input at 120x120 window size
    featext  = load_fiel120()
    ### denoiser input at 120x120 window size
    denoiser = denoise_conv4p120_model()
    denoiser.load_weights('/fileserver/iam/iam-processed/models/noisemodels/conv4p_linet120-iambin-tifs.hdf5')

hdf5file='/fileserver/nmec-handwriting/flat_nmec_cropped_bin_uint8.hdf5'
# hdf5file='/fileserver/nmec-handwriting/flat_nmec_cleaned56_uint8.hdf5'
flatnmec=h5py.File(hdf5file,'r')

outputdir = '/fileserver/nmec-handwriting/localfeatures/nmec_bw_crop_cc_deNNiam120_fiel657-120/'

# Extract connected components, and then shingles with minimum threshold 500
for imname in flatnmec.keys()[237:]:
    ccis = connectedcomponents( flatnmec[imname] )
    shards = shinglesfromcc( ccis, minthresh=500, shingle_dim=shingle_dim )
    if len(shards)==0:
        print "WARNING "+str(imname)+" has no features!"
        continue
    denoised = denoiser.predict( shards, verbose=1 )
    features = featext.predict( np.expand_dims( np.reshape(denoised, (denoised.shape[0],)+shingle_dim), 1), verbose = 1 )
    
    print imname
    np.save(outputdir+imname+'.npy', features)

print imname
flatnmec.keys().index('FR-034-007.bin.crop.png.cropcrop.png')



