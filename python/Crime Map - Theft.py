import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import geopandas as gpd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper
from bokeh.plotting import figure, save, output_file, show
from shapely.geometry import Point
from geopandas import GeoDataFrame
from bokeh.models import HoverTool

df = pd.read_csv("./CrimeData/TheftData.csv", sep =';')

#shape_filepath = r"./CrimeData/Ald2012/alderman.shp"
#shape_filepath = r"./CrimeData/MPD_police_district/poldist.shp"
#shape_filepath = r"./CrimeData/mygeodata/citylimit.shp"
shape_filepath = r"./CrimeData/neighborhood/neighborhood.shp"

shapes = gpd.read_file(shape_filepath)

shapes['geometry'] = shapes['geometry'].to_crs(epsg=3070)

def get_xy_coords(geometry, coord_type):
    if coord_type == 'x':
        return geometry.coords.xy[0]
    elif coord_type == 'y':
        return geometry.coords.xy[1]

def get_poly_coords(geometry, coord_type):
    ext = geometry.exterior
    return get_xy_coords(ext, coord_type)
    
def get_multi_polygon_coords(multi_polygon, coord_type):
    for i, part in enumerate(multi_polygon):
        if i == 0:
            coord_arrays = get_poly_coords(part, coord_type=coord_type)
        else:
            coord_arrays += get_poly_coords(part, coord_type=coord_type)

    return coord_arrays

def get_coords(row, geom, coord_type):
    geometry = row[geom]
    
    gtype = geometry.geom_type
    if gtype == 'Polygon':
        return list(get_poly_coords(geometry, coord_type))
    elif gtype == 'MultiPolygon':
        return list(get_multi_polygon_coords(geometry, coord_type))
    elif gtype == 'Point':
        return get_xy_coords(geometry, coord_type)


shapes['x'] = shapes.apply(get_coords, geom = 'geometry', coord_type = 'x', 
                           axis =1)
shapes['y'] = shapes.apply(get_coords, geom = 'geometry', coord_type = 'y', 
                           axis =1)

s_df = shapes.drop('geometry', axis= 1).copy()

x = []
y = []

def gen_coords(loc):
    data = loc[1:-1].split(',')
    data = list((float(data[0]), float(data[1])))
    x.append(data[1])
    y.append(data[0])
    
    
df['Location'].apply(gen_coords)
points = [Point(xy) for xy in zip(x,y)]
crs = {'init': 'epsg:4326'}
geo_df = GeoDataFrame(df,crs=crs, geometry=points)
geo_df['geometry'] = geo_df['geometry'].to_crs(epsg=3070)

geo_df['x'] = geo_df.apply(get_coords, geom = 'geometry', coord_type = 'x', 
                           axis =1)
geo_df['y'] = geo_df.apply(get_coords, geom = 'geometry', coord_type = 'y', 
                           axis =1)
geo_df['x']= geo_df['x'].apply(lambda x: x[0])
geo_df['y']= geo_df['y'].apply(lambda x: x[0])

def count_points(polygon, points):
    i = 0
    for index, point in points:
        if polygon.contains(point):
            i += 1
    return i
        
s_df['Crime Count'] = shapes['geometry'].copy()

crime_list =[]    
for index, polygon in s_df['Crime Count'].items():
    crime_list.append(count_points(polygon,geo_df['geometry'].items()))

geo_df = geo_df.drop('geometry', axis= 1).copy()
s_df['Crime Count'] = crime_list

psource = ColumnDataSource(geo_df)
s_source = ColumnDataSource(s_df)

from bokeh.palettes import PuBu as Palette

#output_file("theftmap.html")
palette = Palette[9]
palette = palette[::-1]
color_mapper = LogColorMapper(palette=palette)
hover = HoverTool(names=['Points'])
hover.tooltips = [('Address', '@Address'),
                  ('Crime Commited', '@{Offense 1}'),
                  ('Date of Crime', '@Date'),
                  ('Time of Crime','@Time'),]

p = figure(title="Theft in Milwaukee from 1/1/2015 to 1/31/2015", plot_width = 500, plot_height=750, toolbar_location=None,
           active_drag ='pan', active_scroll='wheel_zoom',)
p.axis.visible = False

p.patches('x', 'y', source = s_source, fill_alpha=0.8, line_color="black", line_width=.3,
         fill_color ={'field':'Crime Count','transform': color_mapper},)
p.circle(x='x', y='y',source = psource, size = 5, color='black', name ='Points')
p.add_tools(hover)

show(p)
outfp = r"./maps_view/theftmap.html"
save(p, outfp)

