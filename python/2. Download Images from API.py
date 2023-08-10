import sys
sys.path.append("../Src")
from img_lib import RasterGrid

GRID = RasterGrid()

GRID.config

GRID.config["dataset"]["filename"]

list_i,list_j=GRID.get_gridcoordinates()

GRID.output_image_dir

GRID.config["satellite"]["step"]

GRID.download_images(list_i, list_j)

