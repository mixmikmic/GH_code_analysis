arcgis.env.out_spatial_reference = 4326

import logging
import sys

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)

# Set STDERR handler as the only handler 
logger.handlers = [handler]

arcgis.env.verbose = False 

geosurtools = import_toolbox('http://tps.geosur.info/arcgis/rest/services/Models/GeoSUR_ElevationDerivatives/GPServer')

try:
    geosurtools.slope_classificaiton()
except Exception as e:
    print('The tool encountered an error')

