from astropy.vo.samp import SAMPHubServer

hub = SAMPHubServer()
hub.start()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import urllib
from astropy.table import Table
from astropy.vo.samp import SAMPIntegratedClient

client = SAMPIntegratedClient()
client.connect()

# Setting up Receiver class
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

r.received

r.received

r.params

t = Table.read(r.params['url'])
t

url = t[900]['thumbnail_url']

#print f.read()

f = urllib.urlopen(url)
fig = plt.figure(figsize=(8,8))
image = plt.imread(f)
plt.imshow(image)





