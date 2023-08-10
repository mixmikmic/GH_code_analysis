import requests
import math
import maya
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

tile_url='http://wri-tiles.s3.amazonaws.com/glad_prod/tiles/12/1376/2156.png'
im_arrays = misc.imread(requests.get(tile_url, stream=True).raw, mode='RGBA')

for band in range(0,4):
    print(f"BAND {band}: max={im_arrays[:,:,band].max()}, min={im_arrays[:,:,band].min()}")
    plt.imshow(im_arrays[:,:,band])
    plt.show()

def xyz_from_url(url):
    """Parse the url to  X, Y, Z"""
    z,x,y = tile_url.split('/')[-3:]
    y = y.split('.png')[0]
    z = int(z)
    y = int(y)
    x = int(x)
    return  x, y, z


def num2deg(xtile, ytile, zoom):
    """From a given Z,X,Y we can identify the upper left corner lat long"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

x,y,z = xyz_from_url(tile_url)
print(f'x={x}, y={y}, z={z}')

ul_corner_lat, ul_corner_lon = num2deg(x,y,z)
print(f"Upper Left-hand corner Latitude = {ul_corner_lat}, \nUpper Left-hand corner Longitude = {ul_corner_lon}")

def gen_date_array(im_arrays, confident_values=True, verbose=False):
    """Recieve array (256,256,4) of R,G,B,A. Extract the date which has been encoded in the R +G
    bands. Date is days since 01-01-2015, where integer day is in the G-band, in base 256, once
    255 is reached, the corresponding array element in the R-band is incremented.
    Should return a numpy array of strings with either None, or a string date (YYYY-MM-DD).
    The dates are only returned if the blue-band has a value of 255, indicating a valid pixel.
    E.g. R=1, G=1 should equal 256 days since 01-01-2015
         R=0, G=10 should equal 10 days since 01-01-2015
    """
    base_date = maya.parse('01-01-2015')
    tmp = []
    red = im_arrays[:,:,0]
    green = im_arrays[:,:,1]
    
    if confident_values:
        blue = im_arrays[:,:,2]
        mask = blue != 255
        red = np.ma.masked_array(red, mask=mask)
        green = np.ma.masked_array(green, mask=mask)
    rg_flat = [pair for pair in zip(red.flatten(), green.flatten())]
    for r_band, g_band in rg_flat:
        if(np.sum([r_band, g_band]) > 0):
            days_to_add = (r_band * 255) + g_band
            final_date = base_date.add(days=int(days_to_add)).datetime().date()
            if verbose:
                print(f'r={r_band}, g={g_band}: {r_band * 255} + {g_band} = {days_to_add} , date = {final_date}')
            tmp.append(str(final_date))
        else:
            tmp.append(None)
    tmp = np.array(tmp).reshape((256,256))
    return tmp


def date_array_to_xydate_list(date_array):
    """Get a list of the x, y index for every element in the date array that is not None
    iterate over it and create a list of all the x,y, date-string values"""
    xydate = []
    idx = date_array.nonzero()
    xydate = [ [x,y,date_array[x,y]] for x,y in zip(idx[0], idx[1])]
    return xydate


def return_latlongdate(alerts_array, tilelon, tilelat,  tilezoom):
    """Convert a list of x,y, dates from date_array_to_xydate_list to a list of
    lat, lon, date (where lon, lat are the upper-left corner of the pixel).
    Pixel size in decimal degress is given by distance variable in this function.
    This works via trigonmetry!
    """
    distance = math.fabs((360/256)*(math.cos(tilelat)/(2**tilezoom)))
    return [[(alert[0]*distance - tilelat) , (alert[1]*distance + tilelon ), alert[2]] for alert in alerts_array]
        

# Get the tile metadata
x,y,z = xyz_from_url(tile_url)
ul_corner_lat, ul_corner_lon = num2deg(x,y,z)

# Scrape a list of x,y, dates from the tile (where alert confidence is high)
date_array = gen_date_array(im_arrays, confident_values=True)
good_list = date_array_to_xydate_list(date_array)

# convert the xydate to lon,lat,time
output_list = return_latlongdate(good_list, ul_corner_lon, ul_corner_lat, z)

output_list[90:100]

# If you want to write these data out to a text file as a list

with open("./output.txt",'w') as f:
    for line in output_list:
        a,b,c = line
        f.write(f"{a}, {b}, {c}\n")



