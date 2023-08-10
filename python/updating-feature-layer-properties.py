# connect to your GIS with publisher or higher privileges
from arcgis.gis import GIS
gis = GIS('portal url', 'user name', 'password')

search_result= gis.content.search("Ports along west coast", "Feature Layer")
ports_item = search_result[0]
ports_item

from arcgis.features import FeatureLayerCollection
ports_flc = FeatureLayerCollection.fromitem(ports_item)

ports_flc.properties

update_dict = {'description':'Updated using ArcGIS Python API',
              'copyrightText':'Rohit Singh'}
ports_flc.manager.update_definition(update_dict)

ports_flc.properties.description

ports_flc.properties.copyrightText

update_dict2 = {"capabilities": "Query",
               "syncEnabled": False}
ports_flc.manager.update_definition(update_dict2)

ports_flc.properties.capabilities

ports_flc.properties.syncEnabled

"syncCapabilities" in ports_flc.properties

