import subprocess
import os

#Change system to working directory
workdir = "C:\\Users\\abhubba1\\Documents\\Python Scripts\\DEVELOP_class"
os.chdir(workdir)

MODIS_file = "MOD11A2.A2016201.h11v05.006.2016242234243.hdf"

#Send gdal_translate command to system shell, capture result, and print it
dayLST_fname = MODIS_file.rstrip('.hdf')+'_GDAL_dayLST.tif'
trans_day_cmd = ['gdal_translate', 'HDF4_EOS:EOS_GRID:"'+MODIS_file+                 '":MODIS_Grid_8Day_1km_LST:LST_Day_1km', dayLST_fname]
p_trans_day = subprocess.Popen(trans_day_cmd, stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
print(p_trans_day.communicate())

import os

from osgeo import gdal
import numpy as np

gdal.UseExceptions()

#Change system to working directory
workdir = "C:\\Users\\abhubba1\\Documents\\Python Scripts\\DEVELOP_class"
os.chdir(workdir)

MODIS_file = "MOD11A2.A2016201.h11v05.006.2016242234243.hdf"
dayLST_fname = MODIS_file.rstrip('.hdf')+'_GDAL_dayLST.tif'

#Register driver for this file type
driver = gdal.GetDriverByName("GTiff")
driver.Register()
#Open raster as GDAL dataset
dayLST_dataset = gdal.Open(dayLST_fname)
#Get geotransform and projection from GDAL dataset. These contain 
#the geospatial information of the data, and we will need them 
#later to write the array back to a raster file.
geotrans = dayLST_dataset.GetGeoTransform()
proj = dayLST_dataset.GetProjection()
#Open the only band in the dataset. Note that band numbering 
#starts from 1 as far as GDAL is concerned.
dayLST_band = dayLST_dataset.GetRasterBand(1)
#Pull data from band into a NumPy array
dayLST_array = dayLST_band.ReadAsArray()
#Get the NoData value for this band
fillval = dayLST_band.GetNoDataValue()
#Create a new masked array, where all areas of NoData are masked 
#out
dayLST_ma_array = np.ma.masked_equal(dayLST_array, fillval)
#Empty band and dataset objects to avoid lock issues later. Be 
#sure to empty the band object first, as there can be problems 
#otherwise.
dayLST_band = None
dayLST_dataset = None

scale = 0.02
dayLST_array_sc = dayLST_ma_array * scale

gdal.UseExceptions()

#Create new dataset to contain output
scale_fname = MODIS_file.rstrip('.hdf')+'_GDAL_scale.tif'
out_dataset = driver.Create(scale_fname, dayLST_array_sc.shape[1], 
                            dayLST_array_sc.shape[0], eType = gdal.GDT_UInt16)
#Set geotransform and projection of output dataset
out_dataset.SetGeoTransform(geotrans)
out_dataset.SetProjection(proj)
#Create a band for our data
out_band = out_dataset.GetRasterBand(1)
#Write our data to the band
out_band.WriteArray(dayLST_array_sc)
#Tell the raster which value signifies NoData
out_band.SetNoDataValue(fillval)
#Write the data from memory to disk. Not strictly necessary, as 
#this should occur anyway at some point, but it is good practice.
out_band.FlushCache()
#Clear band and dataset to avoid lock problems
out_band = None
out_dataset = None

import subprocess
import os

#Change system to working directory
workdir = "C:\\Users\\abhubba1\\Documents\\Python Scripts\\DEVELOP_class"
os.chdir(workdir)

MODIS_file = "MOD11A2.A2016201.h11v05.006.2016242234243.hdf"
scale_fname = MODIS_file.rstrip('.hdf')+'_GDAL_scale.tif'

#Send gdalwarp command to system shell, capture result, and print it
reproj_fname = MODIS_file.rstrip('.hdf')+'_GDAL_reproj.tif'
reproj_cmd = ['gdalwarp', '-t_srs', 'EPSG:4326', scale_fname, reproj_fname]
p_reproj = subprocess.Popen(reproj_cmd, stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
print(p_reproj.communicate())

