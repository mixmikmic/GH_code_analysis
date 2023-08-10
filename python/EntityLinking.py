import json

# Import compatibility libraries (python 2/3 support)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Python 3
try:
    import json
    from urllib.request import urlopen, Request
    from urllib.parse import urlparse, urlencode
    from http.client import HTTPSConnection
# Python 2.7
except ImportError:
    import json
    from urlparse import urlparse
    from urllib import urlencode
    from urllib2 import Request, urlopen
    from httplib import HTTPSConnection

response = urlopen('https://gist.githubusercontent.com/antriv/a6962d2c7580a0f7db4b7aabd6d768c5/raw/66d2f4219a566e2af995f6ce160e48851bf7811e/config.json')
data = response.read().decode("utf-8")
CONFIG = json.loads(data)
subscription_key = CONFIG['subscription_key_ELIS']

f = urlopen('https://raw.githubusercontent.com/michhar/bot-education/master/Student-Resources/CognitiveServices/Notebooks/sample_text.txt')

# Read in a process to decode the strange quotes
text = f.read().decode('utf-8')

# Substitute decoded quotes with regular single quotes
import re
text = re.sub('\u2019|\u201c|\u201d', "'", text).replace('\n', ' ')
text = text.encode('utf-8')

# http headers needed for POST request
# we keep these as dict
headers = {
    # Request headers - note content type is text/plain!
    'Content-Type': 'text/plain',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

# params will be added to POST in url request
# right now it's empty because for this request we don't need any params
# although we could have included 'selection' and 'offset' - see docs
params = urlencode({})

try:
    conn = HTTPSConnection('api.projectoxford.ai')
    
    # Post method request - note:  body of request is converted from json to string
    conn.request("POST", "/entitylinking/v1.0/link?%s" % params, body = text, headers = headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
except Exception as e:
    print("[Error: {0}] ".format(e))
    
# Print the results - json response format
print(json.dumps(json.loads(data), 
           sort_keys=True,
           indent=4, 
           separators=(',', ': ')))



