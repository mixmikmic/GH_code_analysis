# Connect to enterprise GIS
from arcgis.gis import GIS
import arcgis.geoanalytics
portal_gis = GIS("portal url", "username", "password")

bigdata_datastore_manager = arcgis.geoanalytics.get_datastores()
bigdata_datastore_manager

bigdata_fileshares = bigdata_datastore_manager.search()
bigdata_fileshares

Chicago_accidents = bigdata_fileshares[0]
len(Chicago_accidents.datasets)

# let us view the first dataset for a sample
Chicago_accidents.datasets[0]

NYC_data_item = bigdata_datastore_manager.add_bigdata("NYCdata2", 
                                                      r"\\teton\atma_shared\datasets\NYC_taxi")

NYC_data_item

NYC_data_item.manifest

