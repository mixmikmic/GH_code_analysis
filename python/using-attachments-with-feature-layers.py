# connect to web GIS
from arcgis.gis import GIS
gis = GIS("portal url", "username","password")
search_result = gis.content.search("Chennai Rainfall", "Feature Layer")
chennai_rainfall = search_result[0]

#get feature layers from the item
cr_lyr = chennai_rainfall.layers[0]

cr_lyr.attachments.get_list(oid=1)

cr_lyr.attachments.download(oid=1, attachment_id=1)

cr_lyr.attachments.add(1, 'C:\\Users\\rohit\\AppData\\Local\\Temp\\AppTemplate.png')

cr_lyr.attachments.get_list(1)

cr_lyr.attachments.delete(1,4)

cr_lyr.attachments.get_list(1)

