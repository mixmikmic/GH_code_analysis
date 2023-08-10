from xml.dom import minidom
import numpy as np
import openslide
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator
from glob import glob

path = 'wsi_data/A02.xml'

xml = minidom.parse(path)
# The first region marked is always the tumour delineation
regions_ = xml.getElementsByTagName("Region")
regions, region_labels = [], []
for region in regions_:
    vertices = region.getElementsByTagName("Vertex")
    attribute = region.getElementsByTagName("Attribute")
    if len(attribute) > 0:
        r_label = attribute[0].attributes['Value'].value
    else:
        r_label = region.getAttribute('Text')
    region_labels.append(r_label)
    
    # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
    coords = np.zeros((len(vertices), 2))
    
    for i, vertex in enumerate(vertices):
        coords[i][0] = vertex.attributes['X'].value
        coords[i][1] = vertex.attributes['Y'].value
        
    regions.append(coords)

print(np.shape(regions))
print(np.shape(region_labels))

region_labels

from shapely.geometry import Polygon, Point

label_map = {'Normal': 0,
             'Benign': 1,
             'Carcinoma in situ': 2,
             'Invasive carcinoma': 3,
            }


def generate_label(regions, region_labels, point):
    # regions = array of vertices (all_coords)
    # point [x, y]
    for i in range(len(region_labels)):
        poly = Polygon(regions[i])
        if poly.contains(Point(point[0], point[1])):
            return label_map[region_labels[i]]
    return label_map['Normal']

generate_label(regions, region_labels, [7500, 21600])

patch_size = 256
percent_overlap = 0
file_dir = "wsi_data/"
file_name = "A01.svs"
xml_file = "A01.xml"
xml_dir = "wsi_data/"
level = 12

overlap = int(patch_size*percent_overlap / 2.0)
tile_size = patch_size - overlap*2

slide = open_slide(file_dir + file_name) 
tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)

if level >= tiles.level_count:
    print("Error: requested level does not exist. Slide level count: " + str(tiles.level_count))

x_tiles, y_tiles = tiles.level_tiles[level]

print(x_tiles)
print(y_tiles)

tiles.get_tile_coordinates(level, (5, 2))[0]

patches, coords, labels = [], [], []
x, y = 0, 0
count = 0
while y < y_tiles:
    while x < x_tiles:
        new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.int)
        # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge 
        # patches are smaller than the others. We will ignore such patches.
        if np.shape(new_tile) == (patch_size, patch_size, 3):
            patches.append(new_tile)
            coords.append(np.array([x, y]))
            count += 1

            # Calculate the patch label based on centre point.
            if xml_file:
                converted_coords = tiles.get_tile_coordinates(level, (x, y))[0]
                labels.append(generate_label(regions, region_labels, converted_coords))
        x += 1
    y += 1
    x = 0

# image_ids = [im_id]*count

print(np.shape(patches))
print(np.shape(coords))
print(np.shape(labels))

np.count_nonzero(labels)

name = 'tester.svs'

name[:-4]



