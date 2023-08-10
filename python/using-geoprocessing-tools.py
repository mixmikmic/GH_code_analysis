# connect to ArcGIS Online
from arcgis.gis import GIS
from arcgis.geoprocessing import import_toolbox
gis = GIS()

# import the Zion toolbox
zion_toolbox_url = 'http://gis.ices.dk/gis/rest/services/Tools/ExtractZionData/GPServer'
zion = import_toolbox(zion_toolbox_url)

result = zion.extract_zion_data()

type(result)

result

result.download()

viewshed = import_toolbox('http://sampleserver1.arcgisonline.com/ArcGIS/rest/services/Elevation/ESRI_Elevation_World/GPServer')

help(viewshed.viewshed)

import arcgis
arcgis.env.out_spatial_reference = 4326

map = gis.map('South San Francisco', zoomlevel=12)
map

from arcgis.features import Feature, FeatureSet

def get_viewshed(m, g):
    res = viewshed.viewshed(FeatureSet([Feature(g)]),"5 Miles") # "5 Miles" or LinearUnit(5, 'Miles') can be passed as input
    m.draw(res)
    
map.on_click(get_viewshed)

sandiego_toolbox_url = 'https://gis-public.co.san-diego.ca.us/arcgis/rest/services/InitialResearchPacketCSV_Phase2/GPServer'
multioutput_tbx = import_toolbox(sandiego_toolbox_url)

help(multioutput_tbx.initial_research_packet_csv)

report_output_csv_file, output_map_flags_file, soil_output_file, _ = multioutput_tbx.initial_research_packet_csv() 

report_output_csv_file

output_map_flags_file

soil_output_file

results = multioutput_tbx.initial_research_packet_csv()

results.report_output_csv_file

results.job_status

hotspots = import_toolbox('https://sampleserver6.arcgisonline.com/arcgis/rest/services/911CallsHotspot/GPServer')

help(hotspots.execute_911_calls_hotspot)

result_layer, output_features, hotspot_raster = hotspots.execute_911_calls_hotspot()

result_layer

hotspot_raster

from IPython.display import Image
Image(hotspot_raster['mapImage']['href'])

