import datetime
import os.path

def himawari8_path(UTC_datetime, band):
    base = "/g/data2/rr5/satellite/obs/himawari8/FLDK/{year:d}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/"
    file_name = "{year:d}{month:02d}{day:02d}{hour:02d}{minute:02d}00-P1S-ABOM_BRF_B{band:02d}-PRJ_GEOS141_2000-HIMAWARI8-AHI.nc"

    st = datetime.datetime.strptime(UTC_datetime, "%Y%m%d%H%M")

    path = (base.format(year=st.year, month=st.month, day=st.day, hour=st.hour, minute=st.minute) +
            file_name.format(year=st.year, month=st.month, day=st.day, hour=st.hour, minute=st.minute, band=band))

    if not os.path.isfile(path):
        return None
    
    return path

# For example, to get the path to band 1 taken of the 25th Dec 2015 at 00:00 UTC
himawari8_path("201512250000", 1)

get_ipython().magic('matplotlib inline')

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

ds = gdal.Open(himawari8_path("201512250000", 1))
band1 = ds.GetRasterBand(1)
raster = band1.ReadAsArray()
no_data = band1.GetNoDataValue()
    
masked = np.ma.array(raster, mask=(raster==no_data))
plt.imshow(masked)

def him_rgb(UTC_datetime, rgb_bands, clip):
    red_path = himawari8_path(UTC_datetime, rgb_bands[0])
    ds_r = gdal.Open(red_path)
    red = ds_r.GetRasterBand(1).ReadAsArray()
    red = (red.clip(0, clip) / clip * 255).astype(np.uint8)

    green_path = himawari8_path(UTC_datetime, rgb_bands[1])
    ds_g = gdal.Open(green_path)
    green = ds_g.GetRasterBand(1).ReadAsArray()
    green = (green.clip(0, clip) / clip * 255).astype(np.uint8)

    blue_path = himawari8_path(UTC_datetime, rgb_bands[2])
    ds_b = gdal.Open(blue_path)
    blue = ds_b.GetRasterBand(1).ReadAsArray()
    blue = (blue.clip(0, clip) / clip * 255).astype(np.uint8)

    return np.stack((red, green, blue), axis=2)

rgb = him_rgb("201512250000", [3, 2, 1], .65)
print(rgb.shape)
plt.imshow(rgb)

plt.imshow(rgb[3200:5000, 1000:3500, :])

import json

wgs84_wkt ='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

def get_raster(geojson_poly, pix_size):
    coords = json.loads(geojson_poly)["coordinates"]
    
    min_lat = min(coords[0][1][1], coords[0][0][1])
    max_lat = max(coords[0][1][1], coords[0][0][1])
    
    min_lon = min(coords[0][2][0], coords[0][0][0])
    max_lon = max(coords[0][2][0], coords[0][0][0])
    
    geotrans = [min_lon, pix_size, 0.0, max_lat, 0.0, -1.0*pix_size]
    size_x = int((max_lon - min_lon) / pix_size)
    size_y = int((max_lat - min_lat) / pix_size)

    mem_drv = gdal.GetDriverByName('MEM')
    ds = mem_drv.Create('', size_x, size_y, 1, gdal.GDT_Float32)
    ds.SetProjection(wgs84_wkt)
    ds.SetGeoTransform(geotrans)

    return ds

def reproject(src_file, dst_ds):
    src_ds = gdal.Open(src_file)
    gdal.ReprojectImage(src_ds, dst_ds, src_ds.GetProjection(), dst_ds.GetProjection(), gdal.GRA_NearestNeighbour)

    return dst_ds.GetRasterBand(1).ReadAsArray()

canberra_gjson = '{"type":"Polygon","coordinates":[[[148.5,-36],[148.5,-35],[150.2,-35],[150.2,-36],[148.5,-36]]]}'
him_dest = get_raster(canberra_gjson, .01)
him_proj = reproject(himawari8_path("201512250000", 1), him_dest)

plt.imshow(him_proj, cmap="gray")

clouds_proj = him_proj > .30
plt.imshow(clouds_proj, cmap="gray")

clouds_proj.mean()

dem_file = "/g/data1/rr1/Elevation/NetCDF/1secSRTM_DEMs_v1.0/DEM-S/Elevation_1secSRTM_DEMs_v1.0_DEM-S_Mosaic_dems1sv1_0.nc"

elv_dest = get_raster(canberra_gjson, .01)
elv_proj = reproject(dem_file, elv_dest)

plt.imshow(elv_proj)

plt.hist(elv_proj.ravel(), bins=64, range=(0.0, 2000))
plt.xlabel('Height')
plt.ylabel('Number of Pixels')
plt.show()

elv_canberra = elv_proj[10:50, 70:120]
elv_mountains = elv_proj[50:, :60]
elv_coast = elv_proj[50:, 130:]

print("The average height in Canberra is {0:.2f} m.".format(elv_canberra.mean()))
print("The average height in the mountains is {0:.2f} m.".format(elv_mountains.mean()))
print("The average height in the coast is {0:.2f} m.".format(elv_coast.mean()))

def give_me_cloud_coverage(utc_datetime, raster_dest):

    him_proj = reproject(himawari8_path(utc_datetime, 1), raster_dest)
    clouds_proj = him_proj > .30
    clouds_canberra = clouds_proj[10:50, 70:120]
    clouds_mountains = clouds_proj[50:, :60]
    clouds_coast = clouds_proj[50:, 130:]

    return int(clouds_canberra.mean()*100), int(clouds_mountains.mean()*100), int(clouds_coast.mean()*100)

cbr, mtn, cst = give_me_cloud_coverage("201609010000", him_dest)

print("Cloud cover 1st Sept 2016. Canberra: {0:d}% Mountains: {1:d}% Coast: {2:d}%".format(cbr, mtn, cst))

def clouds_in_a_period(UTC_start, UTC_end, raster_dest):
    start = datetime.datetime.strptime(UTC_start, "%Y%m%d%H%M")
    end = datetime.datetime.strptime(UTC_end, "%Y%m%d%H%M")
    
    cbr_list = []
    mtn_list = []
    cst_list = []
    while start <= end:
        cbr, mtn, cst = give_me_cloud_coverage(start.strftime("%Y%m%d%H%M"), raster_dest)
        cbr_list.append(cbr)
        mtn_list.append(mtn)
        cst_list.append(cst)
        start = start + datetime.timedelta(hours=24)

    return cbr_list, mtn_list, cst_list

cbr_month, mtn_month, cst_month = clouds_in_a_period("201609010000", "201609300000", him_dest)
days = np.arange(1, 31, 1)
plt.plot(days, cbr_month, 'ro-', days, mtn_month, 'g^-', days, cst_month, 'bs-', label=["a", "b", "c"])
plt.xlabel('Day of the month')
plt.ylabel('Cloud Cover')
lgnd = ['Canberra', 'Mountains', 'Coast']
plt.legend(lgnd)
plt.show()

print("Canberra had an average {0:d}% of clouds during September 2016.".format(int(np.array(cbr_month).mean())))
print("The mountains had and average {0:d}% of clouds during September 2016.".format(int(np.array(mtn_month).mean())))
print("The coast had and average {0:d}% of clouds during September 2016.".format(int(np.array(cst_month).mean())))

