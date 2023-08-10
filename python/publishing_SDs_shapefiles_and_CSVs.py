from IPython.display import display
from arcgis.gis import GIS
import os
gis = GIS("https://www.arcgis.com", "arcgis_python", "P@ssword123")

# path relative to this notebook
data_dir = "data/"

#Get list of all files
file_list = os.listdir(data_dir)

#Filter and get only .sd files
sd_file_list = [x for x in file_list if x.endswith(".sd")]
print("Number of .sd files found: " + str(len(sd_file_list)))

# Loop through each file and publish it as a service
for current_sd_file in sd_file_list:
    item = gis.content.add({}, data_dir + current_sd_file)   # .sd file is uploaded and a .sd file item is created
    published_item = item.publish()                          # .sd file item is published and a web layer item is created
    display(published_item)

data = "data/power_pedestals_2012.zip"
shpfile = gis.content.add({}, data)

shpfile

published_service = shpfile.publish()

display(published_service)

thumbnail_path = "data/power_pedestals_thumbnail.PNG"
item_properties = {"snippet":"""This dataset was collected from Utah DOT open data portal.
                            Source URL: <a href="http://udot.uplan.opendata.arcgis.com/
                            datasets/a627bb128ac44767832402f7f9bde909_10">http://udot.uplan.opendata.arcgis.com/
                            datasets/a627bb128ac44767832402f7f9bde909_10</a>""",
                   "title":"Locations of power pedestals collected in 2012",
                   "tags":"opendata"}

published_service.update(item_properties, thumbnail=thumbnail_path)
display(published_service)

csv_file = 'data/Chennai_precipitation.csv'
csv_item = gis.content.add({}, csv_file)

display(csv_item)

csv_lyr = csv_item.publish(None, {"Address":"LOCATION"})

display(csv_lyr)

# create a new folder called 'Rainfall Data'
new_folder_details = gis.content.create_folder("Rainfall Data")
print(new_folder_details)

# move both the csv_item and csv_lyr items into this new folder
csv_item.move(new_folder_details)  # Here you could either pass name of the folder or the dictionary
                                   # returned from create_folder() or folders property on a User object

csv_lyr.move(new_folder_details)

print(csv_lyr.ownerFolder)

