from telluric import GeoRaster2

rs1 = GeoRaster2.open("../tests/data/raster/overlap1.tif")
rs2 = GeoRaster2.open("../tests/data/raster/overlap2.tif")

rs1

rs2

from shapely.geometry import Polygon

from telluric.constants import WEB_MERCATOR_CRS
from telluric import GeoVector

roi = GeoVector(
    Polygon.from_bounds(1375786, 5166840, 1382728, 5173963),
    WEB_MERCATOR_CRS
)

roi

from telluric.georaster import merge, merge_all, MergeStrategy

merged_all = merge_all([rs1, rs2], roi, merge_strategy=MergeStrategy.INTERSECTION)
merged_all

merged_all.resolution()

merged_all_30 = merge_all([rs1, rs2], roi, dest_resolution=30.0, merge_strategy=MergeStrategy.INTERSECTION)
merged_all_30

merged_all_30.resolution()

rs1_r = rs1.limit_to_bands(["red"])
rs1_gb = rs1.limit_to_bands(["green", "blue"])

rs1_r

rs1_rgb = merge_all([rs1_r, rs1_gb], rs1_r.footprint(), merge_strategy=MergeStrategy.UNION)
rs1_rgb

rs2_g = rs2.limit_to_bands(["green"])
rs2_g

rs_1r_2g = merge_all([rs1_r, rs2_g], roi, merge_strategy=MergeStrategy.UNION)
rs_1r_2g.save("rs_1r_2g.tif")

