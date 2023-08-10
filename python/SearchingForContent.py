from arcgis.gis import GIS

gis = GIS()
content = gis.content

from IPython.display import display
search_result = content.search(query="owner: esri")[:3]
for item in search_result:
    display(item)

search_result = content.search(query='earthquake', item_type="csv")[:3]
for item in search_result:
    display(item)

