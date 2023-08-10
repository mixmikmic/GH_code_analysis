from astropy.vo.samp import SAMPHubServer
hub = SAMPHubServer()
hub.start()

hub.is_running

import urllib
from astropy.table import Table
from astropy.vo.samp import SAMPIntegratedClient

client = SAMPIntegratedClient()
client.connect()

class Receiver(object):
    def __init__(self, client):
        self.client = client
        self.received = False
    def receive_call(self, private_key, sender_id, msg_id, mtype, params, extra):
        self.params = params
        self.received = True
        self.client.reply(msg_id, {"samp.status": "samp.ok", "samp.result": {}})
    def receive_notification(self, private_key, sender_id, mtype, params, extra):
        self.params = params
        self.received = True

r = Receiver(client)

client.bind_receive_call("table.load.votable", r.receive_call)
client.bind_receive_notification("table.load.votable", r.receive_notification)

# client.bind_receive_call("table.load.cdf", r.receive_call)
# client.bind_receive_notification("table.load.cdf", r.receive_notification)

# client.bind_receive_call("table.load.fits", r.receive_call)
# client.bind_receive_notification("table.load.fits", r.receive_notification)

r.received

r.params

t = Table.read(r.params['url'])

t

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.wkt as swkt
from shapely.geometry import Polygon
import re

def getParts(sRegion):
    lon=sRegion.split(' ')[2:][0::2]
    lon=np.asarray([float(i) for i in lon])
    if (lon.max()-lon.min()) > 180:
        lon = [[x, x-360][x>180] for x in lon]
    lat=sRegion.split(' ')[2:][1::2]
    parts = [[360-float(lon[i]),float(lat[i])] for i in range(len(lat))]
    if not (parts[0]==parts[-1]): parts.append(parts[0])
    return [parts]

def s_region_to_wkt(coded_s_region):
    q = getParts(re.sub(r'Polygon UNKNOWNFrame ', '', coded_s_region.decode("utf-8") ))[0]
    return 'POLYGON (('+','.join([' '.join([str(x) for x in y]) for y in q])+'))'

v_s_region_to_wkt = np.vectorize(s_region_to_wkt)

crs = {'init': 'epsg:4326'}
gdf = gpd.GeoDataFrame(t.to_pandas(), crs=crs, geometry=[swkt.loads(s_region_to_wkt(ob)) for ob in t['s_region']])

gdf

gdf.plot(figsize=[20,10])

lonmin , lonmax , latmin , latmax = 250, 300, -25, 25

tmp = gdf[(gdf.c1min > lonmin) & (gdf.c1max < lonmax) & (gdf.c2min > latmin) & (gdf.c2max < latmax)]

tmp.plot(figsize=[(lonmax-lonmin),(latmax-latmin)], column= 'solar_longitude_min')

plt.xlim([lonmin,lonmax])
plt.ylim([latmin,latmax])
plt.xlabel('Longitude')
plt.ylabel('Latitude')



