get_ipython().magic('matplotlib inline')

from IPython.display import Image

import ee
ee.Initialize()

# use ALOS DEM
dem = ee.Image("JAXA/ALOS/AW3D30_V1_1").select('MED')

# define export region
region = ee.Geometry.LineString([[93.116, 28.709], [93.148, 28.673]]).bounds()

# preview image to be exported
image = dem.visualize(**{ 'min': 3800, 'max': 5260 })

# generate url to image thumbnail
url = image.getThumbUrl({'region': region.getInfo(), 'dimensions': 256})

# display image
Image(url=url, embed=True, format='png')

# schedule a new export image task, here we will export to Google Drive
file_name = 'dem'
export_region = region.getInfo()['coordinates'] # BUG: it should just accept region as-is
task = ee.batch.Export.image.toDrive(dem, description=file_name, fileNamePrefix=file_name, region=export_region, scale=30)
task.start()

# remember task id and use it to check tasks status later
task_id = task.id
print(task_id)

# create a new task using id
task = ee.batch.Task(task_id)

# query task status, it should go from READY -> RUNNING -> COMPLETED
task.status()

# query current user tasks
tasks = ee.batch.Task.list()

print(len(tasks))
print(tasks[0])

