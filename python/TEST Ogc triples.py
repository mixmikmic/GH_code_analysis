get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import json
import glob
from semproc.parser import Parser
from semproc.preprocessors.ogc_preprocessors import OgcReader
from semproc.serializers.rdfgraphs import RdfGrapher

with open('../response_examples/wms_v1.3.0.xml', 'r') as f:
    response = f.read()

# this shouldn't be necessary but cargo-culting here is fine by me.
response = response.replace('\\\n', '').replace('\r\n', '').replace('\\r', '').replace('\\n', '').replace('\n', '')
response = response.decode('utf-8', errors='replace').encode('unicode_escape') 
    
url = 'http://ferret.pmel.noaa.gov/thredds/wms/las/woa05_monthly/data_ferret.pmel.noaa.gov_thredds_dodsC_data_PMEL_WOA05nc_monthly_o0112mn1.nc.jnl'
identity = {
    "service": {
        "name": "WMS",
        "request": "GetCapabilities",
        "version": [
            "1.3.0"
        ]
    },
    "protocol": "OGC"
}

reader = OgcReader(identity, response, url, {'harvest_date': '2015-09-15T12:45:00Z'})

reader.parse()

reader.description['layers']

grapher = RdfGrapher(reader.description)
grapher.serialize()
print grapher.emit_format()

with open('../response_examples/wfs_v1_1_0.xml', 'r') as f:
    response = f.read()

# this shouldn't be necessary but cargo-culting here is fine by me.
response = response.replace('\\\n', '').replace('\r\n', '').replace('\\r', '').replace('\\n', '').replace('\n', '')
response = response.decode('utf-8', errors='replace').encode('unicode_escape') 
    
url = 'http://ferret.pmel.noaa.gov/thredds/wms/las/woa05_monthly/data_ferret.pmel.noaa.gov_thredds_dodsC_data_PMEL_WOA05nc_monthly_o0112mn1.nc.jnl'
identity = {
    "service": {
        "name": "WFS",
        "request": "GetCapabilities",
        "version": [
            "1.1.0"
        ]
    },
    "protocol": "OGC"
}

reader = OgcReader(identity, response, url, {'harvest_date': '2015-09-15T12:45:00Z'})
reader.parse()
grapher = RdfGrapher(reader.description)
grapher.serialize()
print grapher.emit_format()

with open('../response_examples/wcs_v1_1_2.xml', 'r') as f:
    response = f.read()

# this shouldn't be necessary but cargo-culting here is fine by me.
response = response.replace('\\\n', '').replace('\r\n', '').replace('\\r', '').replace('\\n', '').replace('\n', '')
response = response.decode('utf-8', errors='replace').encode('unicode_escape') 
    
url = 'http://ferret.pmel.noaa.gov/thredds/wms/las/woa05_monthly/data_ferret.pmel.noaa.gov_thredds_dodsC_data_PMEL_WOA05nc_monthly_o0112mn1.nc.jnl'
identity = {
    "service": {
        "name": "WCS",
        "request": "GetCapabilities",
        "version": [
            "1.1.2"
        ]
    },
    "protocol": "OGC"
}

reader = OgcReader(identity, response, url, {'harvest_date': '2015-09-15T12:45:00Z'})
reader.parse()
grapher = RdfGrapher(reader.description)
grapher.serialize()
print grapher.emit_format()



