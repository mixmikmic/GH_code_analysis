username = ''
token = ''

import netrc
netrc = netrc.netrc()
remoteHostName = "ooinet.oceanobservatories.org"
info = netrc.authenticators(remoteHostName)
username = info[0]
token = info[2]

import requests
import datetime
import pandas as pd

subsite = 'RS03AXPS'
node = 'PC03A'
sensor = '4A-CTDPFA303'

base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587/events/deployment/inv/'

asset_info_url ='/'.join((base_url, subsite, node, sensor, '-1'))

r = requests.get(asset_info_url,auth=(username, token))
asset_info = r.json()

asset_info[0]['location']

ref_des_list = []
start_time_list = []
end_time_list = []
deployment_list = []
depth_list = []
lat_list = [] 
lon_list = [] 

for i in range(len(asset_info)):
    refdes = asset_info[i]['referenceDesignator']
    ref_des_list.append(refdes)

    deployment = asset_info[i]['deploymentNumber']
    deployment_list.append(deployment)

    start = asset_info[i]['eventStartTime']
    end = asset_info[i]['eventStopTime']

    try:
        start_time = datetime.datetime.utcfromtimestamp(start/1000.0)
        start_time_list.append(start_time)

        end_time = datetime.datetime.utcfromtimestamp(end/1000.0)
        end_time_list.append(end_time)

    except:
        end_time = datetime.datetime.utcnow()
        end_time_list.append(end_time)
        
    depth = asset_info[i]['location']['depth']
    depth_list.append(depth)
    lat = asset_info[i]['location']['latitude']
    lat_list.append(lat)
    lon = asset_info[i]['location']['longitude']
    lon_list.append(lon)
    

data_dict = {
    'refdes':ref_des_list,
    'deployment':deployment_list,
    'start_time':start_time_list,
    'end_time':end_time_list,
    'depth': depth_list,
    'latitutde':lat_list,
    'longitude':lon_list}

deployment_data = pd.DataFrame(data_dict, columns = ['refdes', 'deployment','start_time', 'end_time'])

deployment_data





