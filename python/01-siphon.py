from siphon.catalog import TDSCatalog

cat = TDSCatalog('http://dapds00.nci.org.au/thredds/catalog.html')

cat = TDSCatalog('http://dapds00.nci.org.au/thredds/catalog.xml')

get_ipython().magic('pinfo cat')

cat.catalog_refs

ref = cat.catalog_refs['Bureau of Meteorology Observations Data']

get_ipython().magic('pinfo2 ref')

obs = ref.follow()

obs.catalog_url

url = 'http://dapds00.nci.org.au/thredds/catalog/ua6/authoritative/CMIP5/CSIRO-BOM/ACCESS1-0/amip/day/atmos/day/r1i1p1/latest/ua/catalog.xml'
cat = TDSCatalog(url)

cat.datasets

file = cat.datasets['ua_day_ACCESS1-0_amip_r1i1p1_19790101-19831231.nc']

get_ipython().magic('pinfo2 file')

file.access_urls

opendaps = [x.access_urls['OPENDAP'] for x in cat.datasets.values()]
opendaps



