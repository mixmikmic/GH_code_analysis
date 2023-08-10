import numpy as np
import datetime as dt
import os
import gdal
import netCDF4
import re
import glob 

netCDF4.Dataset()

#!/usr/bin/env python
'''
Convert a bunch of GDAL readable grids to a NetCDF Time Series.
Here we read a bunch of files that have names like:
/usgs/data0/prism/1890-1899/us_tmin_1895.01
/usgs/data0/prism/1890-1899/us_tmin_1895.02
...
/usgs/data0/prism/1890-1899/us_tmin_1895.12
'''


#ds = gdal.Open('/usgs/data0/prism/1890-1899/us_tmin_1895.01')
infiles = glob.glob(r'F:\MOD11A2_Gapfilled_Output\LST_Night\Output_Monthly_Means\5km\*.Mean.tif')
outfile = r'C:\temp\LST_Night_5km_Monthly.nc'
ds = gdal.Open(infiles[0])
a = ds.ReadAsArray()
nlat,nlon = np.shape(a)

gt = ds.GetGeoTransform() #bbox, interval
lon = np.arange(nlon)*gt[1]+gt[0]
lat = np.arange(nlat)*gt[5]+gt[3]
b = ds.GetRasterBand(1)
ndv = b.GetNoDataValue()
b = None
ds = None
basedate = dt.datetime(1970,1,1,0,0,0)

# create NetCDF file
#nco = netCDF4.Dataset(r'I:\EVI_5k_time_series.nc','w',clobber=True)
nco = netCDF4.Dataset(outfile,'w',clobber=True)
# chunking is optional, but can improve access a lot: 
# (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
chunk_lon=16
chunk_lat=16
chunk_time=12

# create dimensions, variables and attributes:
nco.createDimension('lon',nlon)
nco.createDimension('lat',nlat)
nco.createDimension('time',None)
timeo = nco.createVariable('time','f4',('time'))
#timeo.units = 'days since 1858-11-17 00:00:00'
timeo.units = 'days since 1970-01-01 00:00:00'
timeo.standard_name = 'time'

lono = nco.createVariable('lon','f4',('lon'))
lono.units = 'degrees_east'
lono.standard_name = 'longitude'

lato = nco.createVariable('lat','f4',('lat'))
lato.units = 'degrees_north'
lato.standard_name = 'latitude'

# create container variable for CRS: lon/lat WGS84 datum
crso = nco.createVariable('crs','i4')
crso.long_name = 'Lon/Lat Coords in WGS84'
crso.grid_mapping_name='latitude_longitude'
crso.longitude_of_prime_meridian = 0.0
crso.semi_major_axis = 6378137.0
crso.inverse_flattening = 298.257223563

# create short float variable for temperature data, with chunking
tmno = nco.createVariable('lst_night', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
tmno.units = 'Celsius'
tmno.scale_factor = 1# 0.01
tmno.add_offset = 0.00
tmno.long_name = 'LST Nighttime monthly 5k mean'
tmno.standard_name = 'lst_night'
tmno.grid_mapping = 'crs'
tmno.set_auto_maskandscale(False)

nco.Conventions='CF-1.6'

#write lon,lat
lono[:]=lon
lato[:]=lat

#pat = re.compile('us_tmin_[0-9]{4}\.[0-9]{2}')
itime=0

#step through data, writing time and data to NetCDF
#for root, dirs, files in os.walk('/usgs/data0/prism/1890-1899/'):
for fn in infiles:
    #dirs.sort()
    #files.sort()
    #for f in files:
    #    if re.match(pat,f):
            # read the time values by parsing the filename
    fname = os.path.basename(fn)
    parts = fname.split('.')
    year = int(parts[1])
    mon = int(parts[2])
    #year=int(f[8:12])
    #mon=int(f[13:15])
    date=dt.datetime(year,mon,1,0,0,0)
    print(date)
    dtime=(date-basedate).total_seconds()/86400.
    timeo[itime]=dtime
   # min temp
    tmn_path = fn
    #tmn_path = os.path.join(root,f)
    print(tmn_path)
    tmn=gdal.Open(tmn_path)
    a=tmn.ReadAsArray()  #data
    tmno[itime,:,:]=a
    itime=itime+1


nco.close()

pfprpath = r'\\129.67.26.176\map_data\ROAD-MAP\data\Website_GIS\Pf_Mapping\Served_Rasters\PfPR_actual_annual_means\*.stable.tif'
pfincpath = r'\\129.67.26.176\map_data\ROAD-MAP\data\Website_GIS\Pf_Mapping\Served_Rasters\Incidence_actual_annual_means\*.stable.tif'
actpath=r'\\129.67.26.176\map_data\ROAD-MAP\data\Website_GIS\Pf_Mapping\Served_Rasters\Interventions\ACT\*.stable.tif'
irspath=r'\\129.67.26.176\map_data\ROAD-MAP\data\Website_GIS\Pf_Mapping\Served_Rasters\Interventions\IRS\*.stable.tif'
itnpath=r'\\129.67.26.176\map_data\ROAD-MAP\data\Website_GIS\Pf_Mapping\Served_Rasters\Interventions\ITN\*.stable.tif'

pfprfiles = glob.glob(pfprpath)
incfiles = glob.glob(pfincpath)
actfiles = glob.glob(actpath)
irsfiles = glob.glob(irspath)
itnfiles = glob.glob(itnpath)

pfprfiles.sort()
incfiles.sort()
actfiles.sort()
irsfiles.sort()
itnfiles.sort()


#ds = gdal.Open('/usgs/data0/prism/1890-1899/us_tmin_1895.01')
outfile = r'C:\temp\PF_2000-2015_Africa_Data_NC3.nc'

ds = gdal.Open(pfprfiles[0])
a = ds.ReadAsArray()
nlat,nlon = np.shape(a)

gt = ds.GetGeoTransform() #bbox, interval
lon = np.arange(nlon)*gt[1]+gt[0]
lat = np.arange(nlat)*gt[5]+gt[3]
b = ds.GetRasterBand(1)
ndv = b.GetNoDataValue()
b = None
ds = None
basedate = dt.datetime(1970,1,1,0,0,0)

# create NetCDF file
#nco = netCDF4.Dataset(r'I:\EVI_5k_time_series.nc','w',clobber=True)
nco = netCDF4.Dataset(outfile,'w',clobber=True, format='NETCDF3_64BIT')
# chunking is optional, but can improve access a lot: 
# (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
chunk_lon=16
chunk_lat=16
chunk_time=12

# create dimensions, variables and attributes:
nco.createDimension('lon',nlon)
nco.createDimension('lat',nlat)
nco.createDimension('time',None)
timeo = nco.createVariable('time','f4',('time'))
#timeo.units = 'days since 1858-11-17 00:00:00'
timeo.units = 'days since 1970-01-01 00:00:00'
timeo.standard_name = 'time'

lono = nco.createVariable('lon','f4',('lon'))
lono.units = 'degrees_east'
lono.standard_name = 'longitude'

lato = nco.createVariable('lat','f4',('lat'))
lato.units = 'degrees_north'
lato.standard_name = 'latitude'

# create container variable for CRS: lon/lat WGS84 datum
crso = nco.createVariable('crs','i4')
crso.long_name = 'Lon/Lat Coords in WGS84'
crso.grid_mapping_name='latitude_longitude'
crso.longitude_of_prime_meridian = 0.0
crso.semi_major_axis = 6378137.0
crso.inverse_flattening = 298.257223563


# create short float variable for each data variable, with chunking
pfpr = nco.createVariable('pfpr', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
pfpr.units = '% parasite prevalance rate in under 5s'
pfpr.scale_factor = 1# 0.01
pfpr.add_offset = 0.00
pfpr.long_name = 'Pf PR annual rate'
pfpr.standard_name = 'pfpr_annual'
pfpr.grid_mapping = 'crs'
pfpr.set_auto_maskandscale(True)

inc = nco.createVariable('pf_inc', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
inc.units = 'pf malaria incidence in cases per 1000 pop per yr'
inc.scale_factor = 1# 0.01
inc.add_offset = 0.00
inc.long_name = 'Pf malaria annual incidence'
inc.standard_name = 'pfinc_annual'
inc.grid_mapping = 'crs'
inc.set_auto_maskandscale(True)

itn = nco.createVariable('itn', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
itn.units = '% population protected by ITNs'
itn.scale_factor = 1# 0.01
itn.add_offset = 0.00
itn.long_name = 'ITN usage rate'
itn.standard_name = 'itn_usage'
itn.grid_mapping = 'crs'
itn.set_auto_maskandscale(True)

irs = nco.createVariable('irs', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
irs.units = '% population protected by IRS treatment'
irs.scale_factor = 1# 0.01
irs.add_offset = 0.00
irs.long_name = 'IRS usage rate'
irs.standard_name = 'irs_usage'
irs.grid_mapping = 'crs'
irs.set_auto_maskandscale(True)

act = nco.createVariable('act', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=ndv)
act.units = '% parasite prevalance rate in under 5s'
act.scale_factor = 1# 0.01
act.add_offset = 0.00
act.long_name = 'ACT usage rate'
act.standard_name = 'act_usage'
act.grid_mapping = 'crs'
act.set_auto_maskandscale(True)


nco.Conventions='CF-1.6'

#write lon,lat
lono[:]=lon
lato[:]=lat

#pat = re.compile('us_tmin_[0-9]{4}\.[0-9]{2}')
itime=0

#step through data, writing time and data to NetCDF
#for root, dirs, files in os.walk('/usgs/data0/prism/1890-1899/'):
for i in range(len(pfprfiles)):
#for fn in pfprfiles:
    fn = pfprfiles[i]
    fname = os.path.basename(fn)
    parts = fname.split('.')
    year = int(parts[1])
   # mon = int(parts[2])
    date=dt.datetime(year,1,1,0,0,0)
    print(date)
    dtime=(date-basedate).total_seconds()/86400.
    timeo[itime]=dtime
   
    print(fn)
    ds=gdal.Open(fn)
    a=ds.ReadAsArray()  #data
    pfpr[itime,:,:]=a
    
    fn=incfiles[i]
    ds = gdal.Open(fn)
    a=ds.ReadAsArray()  #data
    inc[itime,:,:]=a
    
    fn=actfiles[i]
    ds = gdal.Open(fn)
    a=ds.ReadAsArray()  #data
    act[itime,:,:]=a
   
    fn=itnfiles[i]
    ds = gdal.Open(fn)
    a=ds.ReadAsArray()  #data
    itn[itime,:,:]=a
   
    fn=irsfiles[i]
    ds = gdal.Open(fn)
    a=ds.ReadAsArray()  #data
    irs[itime,:,:]=a
   
    itime=itime+1
   
ds = None
nco.close()



