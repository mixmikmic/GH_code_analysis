get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import json
from semproc.process_router import Router
from semproc.rawresponse import RawResponse

def _prep_response(filename):
    # open it and get rid of all the newline junk.
    with open(filename, 'r') as f:
        response = f.read()
    
    response = response.replace('\\\n', '').replace('\r\n', '').replace('\\r', '').replace('\\n', '').replace('\n', '')
    return response.decode('utf-8', errors='replace').encode('unicode_escape')

### opensearch osdd
identity = {
    "protocol": "OpenSearch",
    "service": {
        "name": "OpenSearchDescription",
        "version": "1.1"
    }
}
url = 'http://www.example.com/opensearch.xml'
response = _prep_response('../response_examples/opensearch_blended_parameters.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### opensearch atom feed
identity = {
    "protocol": "OpenSearch",
    "resultset": {
        "dialect": "ATOM",
        "version": "1.1"
    }
}
url = 'http://www.example.com/opensearch.atom'
response = _prep_response('../response_examples/opensearch_usgs_search_atom.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### oai-pmh identify
identity = {
    "protocol": "OAI-PMH",
    "service": {
        "name": "OAI-PMH",
        "request": "Identify"
    }
}
url = 'http://www.example.com/oai-pmh?verb=Identify'
response = _prep_response('../response_examples/oaipmh_identify.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### oai-pmh listrecords
identity = {
    "protocol": "OAI-PMH",
    "resultset": {
        "name": "OAI-PMH",
        "request": "ListRecords",
        "dialect": "oai_dc"
    }
}
url = 'http://www.example.com/oai-pmh?verb=ListRecords'
response = _prep_response('../response_examples/oaipmh_listrecords.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc wms
identity = {
    "protocol": "OGC",
    "service": {
        "name": "WMS",
        "request": "GetCapabilities", 
        "version": "1.3.0"
    }
}
url = 'http://www.example.com/wms?service=wms&request=getcapabilities&version=1.3.0'
response = _prep_response('../response_examples/wms_v1_3_0.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc wfs
identity = {
    "protocol": "OGC",
    "service": {
        "name": "WFS",
        "request": "GetCapabilities",
        "version": "1.1.0"
    }
}
url = ''
response = _prep_response('../response_examples/wfs_v1_1_0.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc wfs features
identity = {
    "protocol": "OGC",
    "dataset": {
        "name": "WFS",
        "request": "GetCapabilities",
        "version": "1.1.0"
    }
}
url = ''
response = _prep_response('../response_examples/wfs_v1_1_0.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc wcs
identity = {
    "protocol": "OGC",
    "service": {
        "name": "WCS",
        "request": "GetCapabilities",
        "version": "1.1.2"
    }
}
url = ''
response = _prep_response('../response_examples/wcs_v1_1_2.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc wcs features
identity = {
    "protocol": "OGC",
    "dataset": {
        "name": "WCS",
        "request": "DescribeCoverage",
        "version": "1.0.0"
    }
}
url = ''
response = _prep_response('../response_examples/wcs_v1_0_0_describe_coverage.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

from owscapable.coverage.wcsBase import DescribeCoverageReader
reader = DescribeCoverageReader('1.0.0', '', None, xml=response)

from semproc.geo_utils import *
from osgeo import osr

ll = reader.coverages[0].min_pos
ur = reader.coverages[0].max_pos
srs = reader.coverages[0].srs_urn

srs_epsg = identify_epsg(srs)
print srs_epsg
epsg = define_spref(srs_epsg)
print epsg

osr_srs = osr.SpatialReference()
osr_srs.ImportFromEPSG(int(srs_epsg.split(':')[-1]))


print int(srs_epsg.split(':')[-1]), ' == ', osr_srs.ExportToPrettyWkt()


# ll = map(float, ll.split())
# ur = map(float, ur.split())

# bbox = ll + ur
# geom = bbox_to_geom(bbox)
# reproject(geom, srs, 'EPSG:4326')

import os
from osgeo import gdal

os.environ['GDAL_DATA'] = r'/Library/Frameworks/GDAL.framework/Versions/1.11/Resources/gdal'
gdal.SetConfigOption( "GDAL_DATA", '/Library/Frameworks/GDAL.framework/Versions/1.11/Resources/gdal' )
print 'GDAL_DATA' in os.environ, os.environ['GDAL_DATA'], gdal.GetConfigOption('GDAL_DATA')


o = osr.SpatialReference()
res = o.ImportFromEPSG(4326)
print repr(res)
o.ExportToPrettyWkt()




### ogc csw
identity = {
    "protocol": "OGC",
    "service": {
        "name": "CSW",
        "request": "GetCapabilities",
        "version": "2.0.2"
    }
}
url = ''
response = _prep_response('../response_examples/datagov_csw_202_getcapabilities.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### ogc csw results
identity = {
    "protocol": "OGC",
    "resultset": {
        "name": "CSW",
        "request": "GetRecords",
        "dialect": "http://www.isotc211.org/2005/gmd",
        "version": "2.0.2"
    }
}
url = ''
response = _prep_response('../response_examples/datagov_csw_202_getrecords_iso.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

### thredds
identity = {
    "protocol": "UNIDATA",
    "service": {
        "name": "THREDDS-Catalog",
        "version": "1.1"
    },
    "dataset": {} # haha, this can be empty for thredds
}
url = 'http://www.example.com/opendap/hyrax/TRMM_3Hourly_3B42/1997/365/catalog.xml'
response = _prep_response('../response_examples/thredds_catalog.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description

# AHAHAHAHA! this is correct for the service description - there's no name/version attribute in this example

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
from semproc.preprocessors.thredds_preprocessors import ThreddsReader
from semproc.xml_utils import *

reader = ThreddsReader(identity, response, url)
reader.parse()
reader.description

#extract_elems(reader.parser.xml, ['dataset'])

#reader.parser.xml


### iso mi/md
identity = {
    "protocol": "ISO",
    "metadata": {
        "name": "19115"
    }
}
url = ''
response = _prep_response('../response_examples/iso-19115_mi.xml')

router = Router(identity, response, url)

print type(router.reader.reader)

router.reader.parse()
router.reader.description

### iso data series
identity = {
    "protocol": "ISO",
    "metadata": {
        "name": "Data Series"
    }
}
url = ''
response = _prep_response('../response_examples/iso-19115_ds.xml')

router = Router(identity, response, url)

print type(router.reader)

router.reader.parse()
router.reader.description



