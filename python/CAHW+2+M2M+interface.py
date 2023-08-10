import kilroy
kilroy.ShowImageFromGitHub('oceanhackweek', 'cahw2018_tutorials','source_material','',                           'Diagram_Three_Installation_Types_in_CA.png',1000,700)

# Read connection credentials from a file (should be saved in this same directory)
authfile=open('../creds/ooiauth.txt','r')     # format of this file is username,token
line=authfile.readline().rstrip()    # please note rstrip() removes any trailing \n whitespace
authfile.close()
username, token = line.split(',')

# specify your inputs
# Reference designator RS01SBPS-SF01A-4A-NUTNRA101
sub_site = 'RS01SBPS'
platform = 'SF01A'
instrument = '4A-NUTNRA101'
delivery_method = 'streamed'                # this taken from the Method column in the table below
stream = 'nutnr_a_sample'                   # this taken from the Data Stream column in the same table
parameter = 'salinity_corrected_nitrate'    # this from clicking on the Data Stream link to the products page

# Friedrich's example
sub_site = 'RS03AXPS'
platform = 'SF03A'
instrument = '2A-CTDPFA302'
delivery_method = 'streamed'
stream = 'ctdpf_sbe43_sample'
parameter = 'seawater_pressure'

import datetime
import time
import requests
import pprint
from concurrent.futures import ThreadPoolExecutor

# setup the base url for the request that will be built using the inputs above.
BASE_URL = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

# we use ThreadPoolExecutor because it has a method .done, which can be polled for 
# completed of the task executed on that thread.
pool = ThreadPoolExecutor(1)

# time stamps are returned in time since 1900, so we subtract 70 years from 
# the time output using the ntp_delta variable
ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

# convert timestamps
def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)

# send request on a thread
def get_future_data(url, params, username, token):
    auth = (username, token)
    return pool.submit(requests.get, url, params=params, auth=auth)

# parse response for timestamp and inform user if no new data returned
def extract_keys(data, keys, min_time):
    rdict = {key: [] for key in keys}
    for record in data:
        if record['time'] <= min_time:
            time_r = record['time']
            time_r = ntp_seconds_to_datetime(time_r)
            time_r = time_r.strftime("%Y-%m-%d %H:%M:%S.000Z")
            print('No new data found since ' + str(time_r) + '. Sending new request.')
            continue
        for key in keys:
            rdict[key].append(record[key])
    print('Found %d new data points after filtering' % len(rdict['time']))
    return rdict

def requestNow(username, token, sub_site, platform, instrument, delivery_method, stream, parameter):
    
    # create the base url
    request_url = '/'.join((BASE_URL, sub_site, platform, instrument, delivery_method, stream))
    
    # specify parameters which will be used in the get_future_data function. 
    # with each new request being sent, only the beginDT will change. 
    # it will be set to the time stamp of the last data point received. 
    # notice that there is no endDT specified, as a request with a beginDT 
    # and no endDT will return everything  from beginDT until present, 
    # up to 1000 data points.
    params = {
        'beginDT': None,
        'limit': 1000,
        'user': 'realtime',
    }
    
    # start with the last 10 seconds of data from present time
    begin_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
    
    # last_time will be assigned as the time stamp of the last data point 
    # received once the first request is sent
    last_time = 0
    
    for i in range(10): # replace with `while True:` to enter an endless data request loop
        
        # update beginDT for this request
        begin_time_str = begin_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        params['beginDT'] = begin_time_str

        # send request in thread
        data_future = get_future_data(request_url, params, username, token)
        
        # poll until complete
        while not data_future.done:
            # while request not complete, yield control to event loop
            time.sleep(0.1)

        # request complete, if not 200, log error and try again
        response = data_future.result()
        if response.status_code != 200:
            print('Error fetching data', response.text)
            time.sleep(0.1)
            continue
        
        # store json response
        data = response.json()
        
        # use extract_keys function to inform users about whether 
        # or not data is being returned. parse data in json response 
        # for input parameter and corresponding timestamp
        data = extract_keys(data, ['time', parameter], min_time=last_time)

        # if no data is returned, try again
        if not data['time']:
            time.sleep(0.1)
            continue

        # set beginDT to time stamp of last data point returned
        last_time = data['time'][-1]
        begin_time = ntp_seconds_to_datetime(last_time)

        # print data points returned
        print("\n")
        pprint.pprint(data)
        print("\n")
        

requestNow(username, token, sub_site, platform, instrument, delivery_method, stream, parameter)



