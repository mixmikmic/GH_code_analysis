import arcgis
from arcgis.gis import GIS
from IPython.display import display

gis = GIS('https://www.arcgis.com', 'arcgis_python', 'P@ssword123')

toolboxes = gis.content.search('travel', 'Geoprocessing Toolbox', 
                               outside_org=True, max_items=3)

for toolbox in toolboxes:
    display(toolbox)

from arcgis.geoprocessing import import_toolbox

ocean_currents_toolbox = toolboxes[1]
ocean_currents_toolbox

ocean_currents = import_toolbox(ocean_currents_toolbox)

import inspect

# list the public functions in the imported module
[ f[0] for f in inspect.getmembers(ocean_currents, inspect.isfunction) 
             if not f[0].startswith('_')]

zion_toolbox_url = 'http://gis.ices.dk/gis/rest/services/Tools/ExtractZionData/GPServer'
zion = import_toolbox(zion_toolbox_url)

help(zion.extract_zion_data)

