get_ipython().magic('load_ext autoreload')

import matplotlib.pyplot as plt

def plotit(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y, 'o')
    plt.show()



print wash_sw.ip, wash_sw.vfc, wash_sw.ofport, wash_sw.rtt

print config.switches
print config.dtns
print config.sites



from coord import get_config as gc
new_config = gc(config_file="calibers-denv.config")

new_config.dtns[0].switch

from coord import SingleFileGen

capacity = 500000 # 500 Mbps
epoch = 5 * 60 # 5 mins

buckets=[30*1024,15*1024,10*1024,5*1024,1*1025,512,128]
gen = SingleFileGen(dtns,capacity,epoch,buckets)
reqs = gen.generate_requests(iterations = 10, scale = 0.1, dst_dtn = scinet_dtn, min_bias=10)

print wash_dtn.requests[0]

x=[]
y_delay=[]
y_size=[]
req_nb = 0
for req_epoch in reqs:
    x.append(req_nb)
    ys_delay = []
    ys_size = []
    for req in req_epoch:
        ys_delay.append(req.delay_ratio)
        ys_size.append(req.size)
    y_delay.append(ys_delay)
    y_size.append(ys_size)
    req_nb += 1
plotit(x,y_delay)
plotit(x,y_size)



gen.save("scenario.data",reqs)

reqs = SingleFileGen.load("scenario.data", dtns)

dtns[0].requests[0]

get_ipython().magic('autoreload 2')
from coord import Coordinator

coord = Coordinator(app_ip="192.168.120.119",name="caliber-slice",epoch_time=10,config_file="calibers-denv.config", scenario_file="scenario.data",max_rate_mbps=500)
coord.scheduler.debug = False
coord.debug = False
coord.start()

get_ipython().run_cell_magic('bash', '', 'curl -H "Content-Type: application/json" http://localhost:5000/api/stop -X POST')

get_ipython().run_cell_magic('bash', '', 'curl -H "Content-Type: application/json" http://192.168.120.119:5000/api/configs/ -X GET')

coord.config.dtns[0].current_request.completed = True

import requests
import pickle

def get_config():
    get_url = "http://192.168.120.119:5000/api/config/"
    try:
        results = requests.get(get_url)
    except requests.exceptions.RequestException:
        return None
    if results.status_code==200:
        return pickle.loads(results.json())
    else: 
        return None

c = get_config()

c.dtns

get_ipython().run_cell_magic('bash', '', 'curl -H "Content-Type: application/json" http://192.168.120.119:5000/api/config/ -X GET')

nc = pickle.dumps(new_config)





import time
import datetime
import subprocess
import re
import json
from flask import Flask, request
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)


import socket
import fcntl
import struct

class FileTransfer(Resource):
    def put(self, size):
        print "got",size
        print request.json
        dest = request.json['dest']
        print "got request: " + dest
        subprocess.Popen(('globus-url-copy vb -fast -p 4 file:///storage/'+size+'.img ftp://'+dest).split())
        time.sleep(.4) # Wait for the connection to establish
        output = subprocess.check_output('ss -int'.split())
        return output

api.add_resource(FileTransfer, '/start/<string:size>')
app.run(host='localhost')
                                         

get_ipython().run_cell_magic('bash', '', 'curl -H "Content-Type: application/json" -d "{\'dest\':\'192.168.112.2:9002/data/test\'}" http://localhost:5000/start/1024000000 -X PUT ')







