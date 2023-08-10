from siphon.catalog import TDSCatalog

catalog = TDSCatalog('http://thredds.cencoos.org/thredds/catalog.xml')


info = """
Catalog information
-------------------

Base THREDDS URL: {}
Catalog name: {}
Catalog URL: {}
Metadata: {}
""".format(catalog.base_tds_url,
           catalog.catalog_name,
           catalog.catalog_url,
           catalog.metadata)

print(info)

for service in catalog.services:
    print(service.name)

print('\n'.join(catalog.datasets.keys()))

print('\n'.join(catalog.catalog_refs.keys()))

ref = catalog.catalog_refs['Global']

[value for value in dir(ref) if not value.startswith('__')]

info = """
Href: {}
Name: {}
Title: {}
""".format(
    ref.href,
    ref.name,
    ref.title)

print(info)

cat = ref.follow()

print(type(cat))

print('\n'.join(cat.datasets.keys()))

dataset = 'Global 1-km Sea Surface Temperature (G1SST)'

ds = cat.datasets[dataset]

ds.name, ds.url_path

for name, ds in catalog.datasets.items():
    if ds.access_urls:
        print(name)

from IPython.display import IFrame

url = 'http://thredds.cencoos.org/thredds/catalog.html?dataset=G1_SST_US_WEST_COAST'

IFrame(url, width=800, height=550)

services = [service for service in catalog.services if service.name == 'wms']

services

service = services[0]

url = service.base

url

from owslib.wms import WebMapService


if False:
    web_map_services = WebMapService(url)
    layer = [key for key in web_map_services.contents.keys() if 'G1_SST_US_WEST_COAST' in key][0]
    wms = web_map_services.contents[layer]

    title = wms.title
    lon = (wms.boundingBox[0] + wms.boundingBox[2]) / 2.
    lat = (wms.boundingBox[1] + wms.boundingBox[3]) / 2.
    time = wms.defaulttimeposition
else:
    layer = 'G1_SST_US_WEST_COAST/analysed_sst'
    title = 'Sea Surface Temperature'
    lon, lat = -122.50, 39.50
    time = 'undefined'

import folium

m = folium.Map(location=[lat, lon], zoom_start=4)

folium.WmsTileLayer(
    name='{} at {}'.format(title, time),
    url=url,
    layers=layer,
    fmt='image/png'
).add_to(m)

folium.LayerControl().add_to(m)

m

from IPython.display import Image

Image(m._to_png())

