get_ipython().magic('matplotlib inline')
import os
import numpy as np
import dicom as dicom
from dicompylercore import dicomparser, dvh, dvhcalc
import matplotlib.pyplot as plt
import urllib.request
import os.path
from ipywidgets import FloatSlider, interactive
from scipy.misc import imresize
from scipy import interpolate

path = os.getcwd()
print(path)
os.listdir(path)

files = ['Case_30_AAA_Structures.dcm',  'Case_30_AAA_Doses.dcm', 'Case_30_Dm_Doses.dcm']

Structures_set = dicomparser.DicomParser(files[0])
AAA = dicomparser.DicomParser(files[1])
Dm = dicomparser.DicomParser(files[2])

key = 5   #  Keys start from 1
structures = Structures_set.GetStructures()  # returns a dict of structures
structures[key]

for i, structure in enumerate(structures):
    print(str(i+1) +  '   ' + structures[structure]['name'])

key = 5
PTV = AAA.GetDVHs()[key]   # returns a DVH object
PTV.name = structures[key]['name']  # assign the structure name
PTV.describe()

plt.figure(figsize=(10, 6))
plt.axis([0, 20, 0, 100])
for s in structures.values():
    if not s['empty']:
        dvh.DVH.from_dicom_dvh(AAA.ds, s['id'], name=s['name']).relative_volume.plot()

def compare_dvh(key=1):
    structure = Structures_set.GetStructures()[key]
    AAA_ = dvh.DVH.from_dicom_dvh(AAA.ds, key, name=structure['name'] + 'AAA')
    Dm_ = dvh.DVH.from_dicom_dvh(Dm.ds, key, name=structure['name'] + 'Dm')
    AAA_.compare(Dm_)

compare_dvh(key=5)

# stolen from https://github.com/dicompyler/dicompyler-plugins/blob/master/plugins/plansum/plansum.py
# add ability to calc difference

def SumPlanRC(old, new, difference=False, q=None, progressFunc=None):
    """ Given two Dicom RTDose objects, returns a summed RTDose object"""
    """The summed RTDose object will consist of pixels inside the region of 
    overlap between the two pixel_arrays.  The pixel spacing will be the 
    coarser of the two objects in each direction.  The new DoseGridScaling
    tag will be the sum of the tags of the two objects.."""
    
    #Recycle the new Dicom object to store the summed dose values
    sum_dcm = new
    
    #Test if dose grids are coincident.  If so, we can directly sum the 
    #pixel arrays.
    if (old.ImagePositionPatient == new.ImagePositionPatient and
        old.pixel_array.shape == new.pixel_array.shape and
        old.PixelSpacing == new.PixelSpacing and
        old.GridFrameOffsetVector == new.GridFrameOffsetVector):
        print("PlanSum: Using direct summation")
        if progressFunc:
            wx.CallAfter(progressFunc, 0, 1, 'Using direct summation')
        sum = old.pixel_array*old.DoseGridScaling +                 new.pixel_array*new.DoseGridScaling
        
    else:    
        #Compute mapping from xyz (physical) space to ijk (index) space
        scale_old = np.array([old.PixelSpacing[0],old.PixelSpacing[1],
                    old.GridFrameOffsetVector[1]-old.GridFrameOffsetVector[0]])
        
        scale_new = np.array([new.PixelSpacing[0],new.PixelSpacing[1],
                    new.GridFrameOffsetVector[1]-new.GridFrameOffsetVector[0]])
        
        scale_sum = np.maximum(scale_old,scale_new)
        
        #Find region of overlap
        xmin = np.array([old.ImagePositionPatient[0],
                         new.ImagePositionPatient[0]])
        ymin = np.array([old.ImagePositionPatient[1],
                         new.ImagePositionPatient[1]])
        zmin = np.array([old.ImagePositionPatient[2],
                         new.ImagePositionPatient[2]])
        xmax = np.array([old.ImagePositionPatient[0] + 
                         old.PixelSpacing[0]*old.Columns,
                         new.ImagePositionPatient[0] + 
                         new.PixelSpacing[0]*new.Columns])
        ymax = np.array([old.ImagePositionPatient[1] + 
                         old.PixelSpacing[1]*old.Rows,
                         new.ImagePositionPatient[1] +
                          new.PixelSpacing[1]*new.Rows])
        zmax = np.array([old.ImagePositionPatient[2] + 
                         scale_old[2]*len(old.GridFrameOffsetVector),
                         new.ImagePositionPatient[2] + 
                         scale_new[2]*len(new.GridFrameOffsetVector)])
        x0 = xmin[np.argmin(abs(xmin))]
        x1 = xmax[np.argmin(abs(xmax))]
        y0 = ymin[np.argmin(abs(ymin))]
        y1 = ymax[np.argmin(abs(ymax))]
        z0 = zmin[np.argmin(abs(zmin))]
        z1 = zmax[np.argmin(abs(zmax))]
        
        
        sum_ip = np.array([x0,y0,z0])
        
        #Create index grid for the sum array
        i,j,k = np.mgrid[0:int((x1-x0)/scale_sum[0]),
                         0:int((y1-y0)/scale_sum[1]),
                         0:int((z1-z0)/scale_sum[2])] 
        
        x_vals = np.arange(x0,x1,scale_sum[0])
        y_vals = np.arange(y0,y1,scale_sum[1])
        z_vals = np.arange(z0,z1,scale_sum[2])
        
        #Create a 3 x i x j x k array of xyz coordinates for the interpolation.
        sum_xyz_coords = np.array([i*scale_sum[0] + sum_ip[0],
                                      j*scale_sum[1] + sum_ip[1],
                                      k*scale_sum[2] + sum_ip[2]])
        
        #Dicom pixel_array objects seem to have the z axis in the first index
        #(zyx).  The x and z axes are swapped before interpolation to coincide
        #with the xyz ordering of ImagePositionPatient
        
        image1 = interpolate_image(np.swapaxes(old.pixel_array,0,2), scale_old, 
            old.ImagePositionPatient, sum_xyz_coords,
            progressFunc)*old.DoseGridScaling
        
        image2 = interpolate_image(np.swapaxes(new.pixel_array,0,2), scale_new, 
            new.ImagePositionPatient, sum_xyz_coords,
            progressFunc)*new.DoseGridScaling
        
        if difference:
            sum = image1 - image2
        else:
            sum = image1 + image2
        
        
        '''sum = interpolate_image(np.swapaxes(old.pixel_array,0,2), scale_old, 
            old.ImagePositionPatient, sum_xyz_coords,
            progressFunc)*old.DoseGridScaling + \
            interpolate_image(np.swapaxes(new.pixel_array,0,2), scale_new, 
            new.ImagePositionPatient, sum_xyz_coords,
            progressFunc)*new.DoseGridScaling'''
        
        #Swap the x and z axes back
        sum = np.swapaxes(sum, 0, 2)
        sum_dcm.ImagePositionPatient = list(sum_ip)
        sum_dcm.Rows = len(y_vals)
        sum_dcm.Columns = len(x_vals)
        sum_dcm.NumberofFrames = len(z_vals)
        sum_dcm.PixelSpacing = [scale_sum[0],scale_sum[1]]
        sum_dcm.GridFrameOffsetVector = list(z_vals - sum_ip[2])
                

    sum_scaling = old.DoseGridScaling + new.DoseGridScaling
    
    sum = sum/sum_scaling
    sum = np.uint32(sum)
    
    sum_dcm.pixel_array = sum
    sum_dcm.BitsAllocated = 32
    sum_dcm.BitsStored = 32
    sum_dcm.HighBit = 31
    sum_dcm.PixelData = sum.tostring()
    sum_dcm.DoseGridScaling = sum_scaling
    if progressFunc:
        wx.CallAfter(progressFunc, 1, 1, 'Done')
    if q:
        q.put(sum_dcm)
    else:
        return sum_dcm

    
def interpolate_image(input_array, scale, offset, xyz_coords, progressFunc):     
    indices = np.empty(xyz_coords.shape)
    indices[0] = (xyz_coords[0] - offset[0])/scale[0]
    indices[1] = (xyz_coords[1] - offset[1])/scale[1]
    indices[2] = (xyz_coords[2] - offset[2])/scale[2]    
    print("PlanSum: Using trilinear_interp")
    return trilinear_interp(input_array, indices, progressFunc)

def trilinear_interp(input_array, indices, progressFunc=None):
    """Evaluate the input_array data at the indices given"""
    
    output = np.empty(indices[0].shape)
    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]
    
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
    #Check if xyz1 is beyond array boundary:
    x1[np.where(x1==input_array.shape[0])] = x0.max()
    y1[np.where(y1==input_array.shape[1])] = y0.max()
    z1[np.where(z1==input_array.shape[2])] = z0.max()

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    output = (input_array[x0,y0,z0]*(1-x)*(1-y)*(1-z) +
                 input_array[x1,y0,z0]*x*(1-y)*(1-z) +
                 input_array[x0,y1,z0]*(1-x)*y*(1-z) +
                 input_array[x0,y0,z1]*(1-x)*(1-y)*z +
                 input_array[x1,y0,z1]*x*(1-y)*z +
                 input_array[x0,y1,z1]*(1-x)*y*z +
                 input_array[x1,y1,z0]*x*y*(1-z) +
                 input_array[x1,y1,z1]*x*y*z)

    return output

# SUM: first run uses Using trilinear_interp, second uses direct summation so must be overwriting
#old = AAA.ds
#new = Dm.ds
#sum_dcm = SumPlanRC(old, new)  # uses sum

# DIFFERENCE first run uses Using trilinear_interp, second uses direct summation so must be overwriting, ONLY RUN ONCE
old = AAA.ds
new = Dm.ds
#diff_dcm = SumPlanRC(old, new, difference=True)

diff_slice = diff_dcm.pixel_array[50,:,:]
#plt.imshow(diff_slice);

#plt.savefig('diff_slice.png')
np.savetxt("diff_slice.csv", np.asarray(diff_slice), delimiter=",")  # save to csv

print(diff_slice.max())

# need to normalise these images somehow

#plt.hist(diff_dcm.pixel_array[50,:,:].flatten(),  bins=256, range=(4.29496E+9, 4.29497E+9));  # all interesting pixels in small range

z_slices = diff_dcm.pixel_array.shape[0]

z_slices = diff_dcm.pixel_array.shape[0]   # get number of slices in converted difference image

planes =     (np.array(AAA.ds.GridFrameOffsetVector)     * AAA.ds.ImageOrientationPatient[0])     + AAA.ds.ImagePositionPatient[2]  # get all the planes in the image data, need to convert these for numpy array of difference data

y = FloatSlider(
    value=-25,
    min=planes[0],
    max=planes[-1],
    step=np.diff(planes)[0],
    description='Slice Position (mm):',
)

image_scale = np.linspace(planes[0], planes[-1], num=z_slices)   # from image data
array_scale = np.linspace(0, z_slices, num = z_slices)                # from numpy

interp_func = interpolate.interp1d(image_scale, array_scale)   # returns an interpolate function, use to get correct array slice given image slice number
#ynew = int(interp_func(115))   # use interpolation function returned by `interp1d`, test using this

def showdoseboth(z):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 12))
    
    AAA_dose_scaling = AAA.GetDoseData()['dosegridscaling']
    Dm_dose_scaling = Dm.GetDoseData()['dosegridscaling']
    
    AAA_dose = AAA.GetDoseGrid(z)
    Dm_dose = Dm.GetDoseGrid(z)
    max_dose = max(AAA.GetDoseGrid(z).max(), Dm.GetDoseGrid(z).max())  # get the max dose to use for color map scaling
    
    AAA_scaled_max = AAA_dose.max()*AAA_dose_scaling
    Dm_scaled_max = Dm_dose.max()*Dm_dose_scaling
    
    
    ax1.imshow(AAA_dose, vmin = 0, vmax = max_dose)
    ax1.set_title('AAA, max dose ' + str(round(AAA_scaled_max,2)))
    
    ax2.imshow(Dm_dose, vmin = 0, vmax = max_dose)    ## AAA_dose and Dm_dose are on the same scale
    ax2.set_title('Dm, max dose ' + str(round(Dm_scaled_max, 2)))
    
    
    diff_image = diff_dcm.pixel_array[int(interp_func(z)),:,:]
    
    diff_range = diff_image[diff_image > 4.0E+8]   #ignore background
    im = ax3.imshow(diff_image, vmin = diff_range.min(), vmax = diff_range.max(), cmap='inferno')  # 
    ax3.set_title('AAA - Dm, max (%) dose difference = ' + str(round((100.0*(AAA_scaled_max - Dm_scaled_max)/AAA_scaled_max),2)))
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.05, 0.25])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    

interactive(showdoseboth, z=y)

get_ipython().magic('pinfo add_axes')

planes =     (np.array(AAA.ds.GridFrameOffsetVector)     * AAA.ds.ImageOrientationPatient[0])     + AAA.ds.ImagePositionPatient[2]

AAA_dd = AAA.GetDoseData()
Dm_dd = Dm.GetDoseData()

y = FloatSlider(
    value=0.56,
    min=planes[0],
    max=planes[-1],
    step=np.diff(planes)[0],
    description='Slice Position (mm):',
)

def showdoseboth(z):
    
    # first test of resizing using imresize - produces artefacts
    AAA_display = imresize(AAA.GetDoseGrid(z), Dm.GetDoseGrid(z).shape)  # make image grids identical and images    
    Dm_display =  imresize(Dm.GetDoseGrid(z), Dm.GetDoseGrid(z).shape)  # make image grids identical    
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    ax1.imshow(AAA_display)
    ax1.set_title(str(z) + ' pos. AAA ' )
    
    ax2.imshow(Dm_display)    
    ax2.set_title('Dm ' )
    
    
    difference_display = np.subtract(AAA_display, Dm_display)
    ax3.imshow(difference_display, cmap='hot') #  vmin=0,vmax=50, 
    
    ax3.set_title('AAA - Dm ' )
    plt.show()
    

interactive(showdoseboth, z=y)

