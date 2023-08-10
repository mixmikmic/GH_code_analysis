# connect to Enterprise GIS
from arcgis.gis import GIS
import arcgis.geoanalytics

gis = GIS("http://dev003246.esri.com/portal", "arcgis_python_api", "sharing.1")

# check if GeoAnalytics is supported
arcgis.geoanalytics.is_supported()

ago_gis = GIS()
arcgis.geoanalytics.is_supported(ago_gis)

search_result = gis.content.search("", item_type = "big data file share")
search_result

data_item = search_result[5]
data_item

data_item.layers

year_2015 = data_item.layers[0]
year_2015

from arcgis.geoanalytics.summarize_data import aggregate_points

arcgis.env.process_spatial_reference=3857

agg_result = aggregate_points(year_2015, bin_size=1, bin_size_unit='Kilometers')

