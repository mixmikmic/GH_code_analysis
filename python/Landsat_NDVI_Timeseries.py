import requests, json, numpy, datetime
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pywren

# Function to return a Landsat 8 scene list given a Longitude,Latitude string
# This uses the amazing developmentseed Satellite API
# https://github.com/sat-utils/sat-api
def getSceneList(lonLat):
    scenes=[]
    url = "https://api.developmentseed.org/satellites/landsat"
    params = dict(
        contains=lonLat,
        satellite_name="landsat-8",
        limit="1000") 
    # Call the API to grab the scene metadata
    sceneMetaData = json.loads(requests.get(url=url, params=params).content)
    # Parse the metadata
    for record in sceneMetaData["results"]:
        scene = str(record['aws_index'].split('/')[-2]) 
        # This horrible hack is to get around some versioning problem on the API :(
        # I'd love to get this sorted out, probaby by hosting my own sat-api instance
        # Related to this issue https://github.com/sat-utils/sat-api/issues/18 
        if scene[-2:] == '01':
            scene = scene[:-2] + '00'
        if scene[-2:] == '02':
            scene = scene[:-2] + '00'
        if scene[-2:] == '03':
            scene = scene[:-2] + '02'
        scenes.append(scene)   
    return scenes


# Function to call a AWS Lambda function to drill a single pixel and compute the NDVI
# Replace the url here with the endpoint of your local function :)
def getNDVI(scene):
    url = " https://w5xm4e5886.execute-api.us-west-2.amazonaws.com/production/l8_ndvi_point"
    params = dict(
        coords=lonLat,
        scene=scene)
    # Call the API and return the JSON results
    resp = requests.get(url=url, params=params)
    return json.loads(resp.text)

get_ipython().run_cell_magic('time', '', "\n# The location of the point of interest\nlonLat = '147.870599,-28.744617'\n\n\n# Call the api to retrieve the scenes available under the point of interest\nscenes = getSceneList(lonLat)\n\n# Set up a pywren executor and map the NDVI retrieval across all the available scenes\npwex = pywren.default_executor()\ntimeSeries = pywren.get_all_results(pwex.map(getNDVI, scenes))\n\n# Extract the data trom the list of results\ntimeStamps = [datetime.datetime.strptime(obs['date'],'%Y-%m-%d') for obs in timeSeries if 'date' in obs]\nndviSeries = [obs['ndvi'] for obs in timeSeries if 'ndvi' in obs]\ncloudSeries = [obs['cloud']/100 for obs in timeSeries if 'cloud' in obs]\n\n# Create a time variable as the x axis to fit the observations\n# First we convert to seconds\ntimeSecs = numpy.array([(obsTime-datetime.datetime(1970,1,1)).total_seconds() for obsTime in timeStamps])\n# And then normalise from 0 to 1 to avoid any numerical issues in the fitting\nfitTime = ((timeSecs-numpy.min(timeSecs))/(numpy.max(timeSecs)-numpy.min(timeSecs)))\n\n# Smooth the data by fitting a spline weighted by cloud amount\nsmoothedNDVI=UnivariateSpline(\n    fitTime[numpy.argsort(fitTime)],\n    numpy.array(ndviSeries)[numpy.argsort(fitTime)],\n    w=(1.0-numpy.array(cloudSeries)[numpy.argsort(fitTime)])**2.0,\n    k=2,\n    s=0.1)(fitTime)\n\n\n# Setup the figure and plot the data, fit and cloud amount\nfig = plt.figure(figsize=(16,10))\nplt.plot(timeStamps,ndviSeries, 'gx',label='Raw NDVI Data')\nplt.plot(timeStamps,ndviSeries, 'g:', linewidth=1)\nplt.plot(timeStamps,cloudSeries, 'b.', linewidth=1,label='Scene Cloud Percent')\nplt.plot(timeStamps,smoothedNDVI, 'r--', linewidth=3,label='Cloudfree Weighted Spline')\nplt.xlabel('Date', fontsize=16)\nplt.ylabel('NDVI', fontsize=16)\nplt.title('AWS Lambda Landsat 8 NDVI Drill', fontsize=20)\nplt.grid(True)\nplt.ylim([-.1,1.0])\nplt.legend(fontsize=14)\n#plt.savefig('lambdaNDVI.png', bbox_inches='tight')")

get_ipython().run_cell_magic('time', '', "\n# The location of the point of interest\nlonLat = '87.996185,26.680658'\n\n# Call the api to retrieve the scenes available under the point of interest\nscenes = getSceneList(lonLat)\n\n# Set up a pywren executor and map the NDVI retrieval across all the available scenes\npwex = pywren.default_executor()\ntimeSeries = pywren.get_all_results(pwex.map(getNDVI, scenes))\n\n# Extract the data trom the list of results\ntimeStamps = [datetime.datetime.strptime(obs['date'],'%Y-%m-%d') for obs in timeSeries if 'date' in obs]\nndviSeries = [obs['ndvi'] for obs in timeSeries if 'ndvi' in obs]\ncloudSeries = [obs['cloud']/100 for obs in timeSeries if 'cloud' in obs]\n\n# Create a time variable as the x axis to fit the observations\n# First we convert to seconds\ntimeSecs = numpy.array([(obsTime-datetime.datetime(1970,1,1)).total_seconds() for obsTime in timeStamps])\n# And then normalise from 0 to 1 to avoid any numerical issues in the fitting\nfitTime = ((timeSecs-numpy.min(timeSecs))/(numpy.max(timeSecs)-numpy.min(timeSecs)))\n\n# Smooth the data by fitting a spline weighted by cloud amount\n# Note we change the parameters a little in this example to account for the very cloudey environment\nsmoothedNDVI=UnivariateSpline(\n    fitTime[numpy.argsort(fitTime)],\n    numpy.array(ndviSeries)[numpy.argsort(fitTime)],\n    w=(1.0-numpy.array(cloudSeries)[numpy.argsort(fitTime)])**4.0,\n    k=3,\n    s=0.2)(fitTime)\n\n\n# Setup the figure and plot the data, fit and cloud amount\nfig = plt.figure(figsize=(16,10))\nplt.plot(timeStamps,ndviSeries, 'gx',label='Raw NDVI Data')\nplt.plot(timeStamps,ndviSeries, 'g:', linewidth=1)\nplt.plot(timeStamps,cloudSeries, 'b.', linewidth=1,label='Scene Cloud Percent')\nplt.plot(timeStamps,smoothedNDVI, 'r--', linewidth=3,label='Cloudfree Weighted Spline')\nplt.xlabel('Date', fontsize=16)\nplt.ylabel('NDVI', fontsize=16)\nplt.title('AWS Lambda Landsat 8 NDVI Drill', fontsize=20)\nplt.grid(True)\nplt.ylim([-.1,1.0])\nplt.legend(fontsize=14)\n#plt.savefig('lambdaNDVI.png', bbox_inches='tight')")



