# Author: Cindy Y.Liu, Tengfei Zheng

import numpy as np
from numpy import random
import pandas as pd
import scipy
from scipy import stats
import pylab as pl
get_ipython().magic('pylab inline')
import math
import csv
import shapefile

my_folder = '/Users/cindyliu/Documents/Line of Sight_Materials/All Deliverables' # file path substituted here

# Output folder
file_path_1 = '{0}/Pixel Files/All'.format(my_folder)

# Select the plume types    
plume_type = ['Ammonia','Carbon Dioxide','Difluoromethane','Cholorodifluoromethane','Tetrafluoroethane']

# Change the x pixel of westside2 and westside3 images
def adjust_x_pixel(df):
    # create a column to store the adjusted x pixel for the detections in westside2 and 3
    df['Detection pixel x_adjusted']=""
    for index,row in df.iterrows():
        if "Westside2" in row.Cube:
            df.loc[index,'Detection pixel x_adjusted'] = df.loc[index,'Detection pixel x'] + 650
        elif "Westside3" in row.Cube:
            df.loc[index,'Detection pixel x_adjusted'] = df.loc[index,'Detection pixel x'] + 1400
        else:
            df.loc[index,'Detection pixel x_adjusted'] = df.loc[index,'Detection pixel x']
    return df

def get_all_detections_pixel(plume_name):
    df = pd.read_csv('AllDays_Top8agents.csv')
    # determine the y pixel range in which detections are kept
    y_pixel_lower_limit = 15
    y_pixel_upper_limit = 100
    # filter out those detections which have vertical positions higher and lower than buildings
    df = df[(df['Detection pixel y']>y_pixel_lower_limit) & (df['Detection pixel y']<y_pixel_upper_limit)]
    # filter out those detections which are not in the range of Westside1,2,3
    df = df[df['Cube'].str.contains('Westside1|Westside2|Westside3', na=False)]
    # subset the detections of the input plume
    df = df[df['ID'].str.contains(plume_name, na=False)]
    # adjust_x_pixel
    df = adjust_x_pixel(df)
    # export to csv file 
    df.to_csv('{0}/{1}_All_Pixels.csv'.format(file_path_1,plume_name))

for i in plume_type:
    get_all_detections_pixel(i)

# Select a specific day
which_date = '12'

def get_one_day_detections_pixel(plume_name,which_date):
    df = pd.read_csv(('{0}/{1}_All_Pixels.csv'.format(file_path_1,plume_name)))
    # subset the detections of a specific date
    date_format = '4/'+ which_date + '/15'
    df = df[df['NY Time'].str.contains(date_format, na=False)]
    # export to csv file 
    df.to_csv('{0}/201504{1}/{2}_All_Pixels.csv'.format(file_path_1,which_date,plume_name))

for i in plume_type:
    get_one_day_detections_pixel(i,which_date)

# output folder
file_path_2 = '{0}/Point Files/All'.format(my_folder) 

# get 
x0,y0 =  -74.025852,40.744314
xn,yn =  -73.983227,40.769178
xs,ys =  -74.017737,40.705468
k1 = (yn-y0)/(xn-x0)
k2 = (ys-y0)/(xs-x0)
theta1 = math.atan(k1)
theta2 = math.atan(k2)
theta_all = theta1-theta2

# pixel to lat,lon transfer
def transfer_to_lon_lat(detections_pixel_range):
    detections_pixel_range['theta'] = theta_all*(1941 - detections_pixel_range['Detection Pixel x_adjusted'])/(1941-2)
    detections_pixel_range['k'] = (tan(detections_pixel_range['theta'])+k2)/(1- tan(detections_pixel_range['theta'])*k2)
    detections_pixel_range['b'] = y0 - x0*detections_pixel_range['k']
    detections_pixel_range['Lon_End'] = -73.93  # choose an arbitrary longtitude end for the line of sight
    detections_pixel_range['Lat_End'] = detections_pixel_range['k']*detections_pixel_range['Lon_End']  + detections_pixel_range['b']
    return detections_pixel_range

# if only need transfer pixels on a specific date, change the file path to the specific date folder
file_path_1 = '{0}/Pixel Files/All/201504{1}'.format(my_folder,which_date)
file_path_2 = '{0}/Point Files/All/201504{1}'.format(my_folder,which_date)

# get longitude and latitude information of detections
def get_lon_lat(plume_name):
    dm = pd.read_csv('{0}/{1}_All_Pixels.csv'.format(file_path_1,plume_name))
    # get the two boundaries' pixel information for detections
    dm['range_side_1'] = dm['Detection pixel x_adjusted'] - dm['Pixels']/2
    dm['range_side_2'] = dm['Detection pixel x_adjusted'] + dm['Pixels']/2
    range_side_1 = dm['range_side_1'].tolist()
    range_side_2 = dm['range_side_2'].tolist()
    # extract detection index information
    detection_index = dm['Unnamed: 0']
    detection_index_double_list = []
    for j in detection_index:
        detection_index_double_list.append(j)
        detection_index_double_list.append(j)
    # extract time information
    NY_Time_list = dm['NY Time']
    NY_Time_double_list = []
    for i in NY_Time_list:
        NY_Time_double_list.append(i)
        NY_Time_double_list.append(i)
    # extract x pixel information
    pixel_x_list=[]
    for i in range(len(range_side_1)):
        pixel_x_list.append(range_side_1[i])
        pixel_x_list.append(range_side_2[i])
    # create a dataframe to store all the information together
    detections_pixel_range = pd.DataFrame({'Detection Index':detection_index_double_list,'NY Time':NY_Time_double_list,                                          'Detection Pixel x_adjusted':pixel_x_list})
    # transfer to longitude and latitude from pixel
    detections_pixel_range = transfer_to_lon_lat(detections_pixel_range)
    # export to csv file
    detections_pixel_range.to_csv('{0}/{1}_All_Lon_Lat.csv'.format(file_path_2,plume_name))

for i in plume_type:
    get_lon_lat(i)

# Select a specific day
which_date = '12'

# 
file_path_4 = '{0}/Pixel Files/Source'.format(my_folder)
file_path_5 = '{0}/Point Files/Source'.format(my_folder) 

# pixel to lat,lon transfer
def source_transfer_to_lon_lat(detections_pixel_range):
    detections_pixel_range['theta'] = theta_all*(1941 - detections_pixel_range['Detection Pixel x'])/(1941-2)
    detections_pixel_range['k'] = (tan(detections_pixel_range['theta'])+k2)/(1- tan(detections_pixel_range['theta'])*k2)
    detections_pixel_range['b'] = y0 - x0*detections_pixel_range['k']
    detections_pixel_range['Lon_End'] = -73.93  # choose an arbitrary longtitude end for the line of sight
    detections_pixel_range['Lat_End'] = detections_pixel_range['k']*detections_pixel_range['Lon_End']  + detections_pixel_range['b']
    return detections_pixel_range

def source_get_lon_lat(plume_name):
    if plume_name == 'Carbon Dioxide':
        plume_name = "Carbon_Dioxide"
    dm = pd.read_csv('{0}/{1}/{2}_Source_Pixels_1504{3}.csv'.format(file_path_4,plume_name,plume_name,which_date))
    # extract x pixel information
    pixel_x_list = dm['Detection pixel x']
    # create a dataframe to store all the information together
    detections_pixel_range = pd.DataFrame({'Detection Pixel x':pixel_x_list})
    # transfer to longitude and latitude from pixel
    detections_pixel_range = source_transfer_to_lon_lat(detections_pixel_range)
    # export to csv file
    detections_pixel_range.to_csv('{0}/201504{1}/{2}_All_Lon_Lat.csv'.format(file_path_5,which_date,plume_name))

for i in plume_type:
    source_get_lon_lat(i)

# Output file path
file_path_3 = '{0}/Polygon Files/All'.format(my_folder) 

# Steven Institute of Technology Observation Point
X0 = -74.0239
Y0 = 40.7449 

# if only need polygon files a specific date, change the file path to the specific date folder
file_path_3 = '{0}/Polygon Files/All/201504{1}'.format(my_folder,which_date)

# get a list of polygon points for generating the polygon shapefile
def get_polygon_points(plume_name):
    dm = pd.read_csv('{0}/{1}_All_Lon_Lat.csv'.format(file_path_2,plume_name))  #input files
    # change NY time to timestamp format
    dm['NY Time'] = pd.to_datetime(dm['NY Time'])
    # new lists for the ending points' lontitude and latitude
    X = dm['Lon_End'].tolist()
    Y = dm['Lat_End'].tolist()
    # new list for NY timestamp
    d_time = dm['NY Time'].tolist()
    list_of_time = []
    for i in range(len(d_time)/2):
        list_of_time.append(d_time[2*i])
    # connect the observation points to the two ending points to create three-sided polygon
    list_of_points = []
    for i in np.arange(0,dm.shape[0],2):
        points = [[X0,Y0]]
        point_one = []
        point_one.append(X[i])
        point_one.append(Y[i])
        point_two = []
        point_two.append(X[i+1])
        point_two.append(Y[i+1]) 
        points.append(point_one)
        points.append(point_two)
        list_of_points.append(points)
    return plume_name,list_of_points,list_of_time

def get_polygon_shapefile(plume_name,list_of_points,list_of_time):  
    w = shapefile.Writer(shapefile.POLYGON)
    for i in range(0,len(list_of_points)):
        w.poly(parts = [list_of_points[i]])
    w.field('FIRST_FLD','C','40')
    w.field('NY_Time','C','40')
    for j in range(0,len(list_of_time)):
        w.record('First',list_of_time[j])
    w.save('{0}/{1}/{2}'.format(file_path_3,plume_name,plume_name))

for i in plume_type:
    p,o,t = get_polygon_points(i)
    get_polygon_shapefile(p,o,t)

# output folder
file_path_6 = '{0}/Polygon Files/Source'.format(my_folder)

# get a list of polygon points for generating the polygon shapefile
def source_get_polygon_points(plume_name):
    if plume_name == 'Carbon Dioxide':
        plume_name = "Carbon_Dioxide"
    dm = pd.read_csv('{0}/201504{1}/{2}_All_Lon_Lat.csv'.format(file_path_5,which_date,plume_name))
    # new lists for the ending points' lontitude and latitude
    X = dm['Lon_End'].tolist()
    Y = dm['Lat_End'].tolist()
    # connect the observation points to the two ending points to create three-sided polygon
    list_of_points = []
    for i in np.arange(0,dm.shape[0],2):
        points = [[X0,Y0]]
        point_one = []
        point_one.append(X[i])
        point_one.append(Y[i])
        point_two = []
        point_two.append(X[i+1])
        point_two.append(Y[i+1]) 
        points.append(point_one)
        points.append(point_two)
        list_of_points.append(points)
    return plume_name,list_of_points

def source_get_polygon_shapefile(plume_name,list_of_points):  
    w = shapefile.Writer(shapefile.POLYGON)
    for i in range(0,len(list_of_points)):
        w.poly(parts = [list_of_points[i]])
    w.field('FIRST_FLD','C','40')
    for j in range(0,len(list_of_points)):
        w.record('First','Polygon')
    w.save('{0}/201504{1}/{2}/{3}'.format(file_path_6,which_date,plume_name,plume_name))

for i in plume_type:
    p,o = source_get_polygon_points(i)
    source_get_polygon_shapefile(p,o)



