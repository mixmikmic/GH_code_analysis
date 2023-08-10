from pyrocko.fdsn import ws
from pyrocko import util, io, trace

# select time range
tmin = util.stt('2015-09-16 22:54:33')
tmax = util.stt('2015-09-17 00:54:33')

# select networkID, StationID, Location, Channel
# use asterisk '*' as wildcard
selection = [('X9', '*', '*', 'HHZ', tmin, tmax)] 

# Read access-token for the restricted network from file as 'rb' for binary
token=open('token.asc', 'rb').read()

# setup a waveform data request
request_waveform = ws.dataselect(site='geofon', selection=selection, token=token)

# Alternative method using username and password instead of token:
# request_waveform = ws.dataselect(site='geofon', selection=selection, user='user', passwd='passwd')

# write the incoming data stream to 'traces.mseed' as 'wb' for binary
with open('traces.mseed', 'wb') as file:
    file.write(request_waveform.read())

