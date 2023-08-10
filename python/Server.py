import zmq
import msgpack
import time
import numpy as np
import sys
import json
from pprint import pprint
import ceo

port = "5556"
config = json.loads(open('simulation.json').read())
pprint(config)
pprint(config["optical path"][0]["ubuntu_cuda70"]["path"])

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

message = socket.recv()
print "Received request: ", message

optical_path = config["optical path"][0][message]["path"]
pprint(optical_path)

socket.send_json(optical_path)

for k in range(len(optical_path)):
    message = socket.recv()
    print "Received request: ", message
    time.sleep (1)  
    socket.send_json(json.loads(open(message+'.json').read())) 

data_port = "5557"
data_context = zmq.Context()
print "Connecting to server..."
data_socket = data_context.socket(zmq.REQ)
data_socket.connect ("tcp://localhost:%s" % data_port)

ea = np.zeros((7,3))
ea[2,1] = ceo.constants.ARCSEC2RAD

print "Sending request ", "wavefront","..."
data_socket.send ("wavefront")

msg = data_socket.recv()
if msg=="euler angles":
    print "Received request: ", msg
    time.sleep (1) 
    msg = msgpack.packb(ea.tolist())
    data_socket.send(msg)
else:
    data = np.array( msgpack.unpackb(msg,use_list=False) )

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(data)

msg=='euler angles'

messa

