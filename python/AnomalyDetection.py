import json

# Import compatibility libraries (python 2/3 support)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Python 3
try:
    from urllib.request import urlopen, Request
    from urllib.parse import urlparse, urlencode
    from http.client import HTTPSConnection
# Python 2.7
except ImportError:
    from urlparse import urlparse
    from urllib import urlencode
    from urllib2 import Request, urlopen
    from httplib import HTTPSConnection 

body = json.loads('''
{
  "data": [
    [ "9/21/2014 11:05:00 AM", "1.3" ],
    [ "9/21/2014 11:10:00 AM", "9.09" ],
    [ "9/21/2014 11:15:00 AM", "2.4" ],
    [ "9/21/2014 11:20:00 AM", "2.5" ],
    [ "9/21/2014 11:25:00 AM", "2.6" ],
    [ "9/21/2014 11:30:00 AM", "2.1" ],
    [ "9/21/2014 11:35:00 AM", "3.5" ],
    [ "9/21/2014 11:40:00 AM", "0" ],
    [ "9/21/2014 11:45:00 AM", "2.8" ],
    [ "9/21/2014 11:50:00 AM", "2.3" ]
  ],
  "params": {
    "tspikedetector.sensitivity": "4",
    "zspikedetector.sensitivity": "4",
    "trenddetector.sensitivity": "3.25",
    "bileveldetector.sensitivity": "3.25",
    "postprocess.tailRows": "0"
  }
}
''')

print(body)

f = urlopen('https://gist.githubusercontent.com/antriv/a6962d2c7580a0f7db4b7aabd6d768c5/raw/38a66f77c7fd0641324c8cbbff77828207041edc/config.json')
url = f.read()
CONFIG = json.loads(url)

subscription_key = CONFIG['subscription_key_ADM']

import base64
creds = base64.b64encode('userid:' + subscription_key)

headers = {'Content-Type':'application/json', 'Authorization':('Basic '+ creds)} 

# params will be added to POST in url request
# right now it's empty because for this request we don't need any params
# although we could have included 'selection' and 'offset' - see docs
params = urlencode({})

try:
    
    # Post method request - note:  body of request is converted from json to string

    conn = HTTPSConnection('api.datamarket.azure.com')
    
    conn.request("POST", "/data.ashx/aml_labs/anomalydetection/v2/Score/", 
                 body = json.dumps(body), headers = headers)
    
    response = conn.getresponse()
    data = response.read()
    conn.close()
except Exception as e:
    print("[Error: {0}] ".format(e))
    
try:
    # Print the results - json response format
    print(json.dumps(json.loads(json.loads(data)['ADOutput']), 
               sort_keys=True,
               indent=4, 
               separators=(',', ': ')))
except Exception as e:
    print(data)



