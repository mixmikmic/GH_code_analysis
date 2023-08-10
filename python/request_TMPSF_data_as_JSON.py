USERNAME ='YOUR API KEY'
TOKEN= 'YOUR API TOKEN'

import requests
import json
import datetime

response = requests.get('https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/RS03ASHS/MJ03B/07-TMPSFA301/streamed/tmpsf_sample?beginDT=2017-07-04T17:54:58.050Z&endDT=2017-07-04T23:54:58.050Z&limit=1000', auth=(USERNAME, TOKEN))
data = response.json()

ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)

print ntp_seconds_to_datetime(data[0]['time'])

for key, value in data[0].items():
    if len(key) == 13 and key.startswith("t"):
        print key, value

