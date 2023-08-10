import urllib
import requests
import operator
from __future__ import print_function

_url = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/ocr'
_key = 'Sua chave aqui!'
_maxNumRetries = 10

from IPython.display import Image as ipImage, display

urlImage = 'https://i-technet.sec.s-msft.com/pt-br/mt629510.banne2(pt-br,MSDN.10).png'

img = ipImage(url=urlImage, width=800)
display(img)

def processRequest( json, data, headers, params ):

    retries = 0
    result = None

    while True:

        response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json()['error']['message'] ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break
        
    return result

headers = dict()
headers['Ocp-Apim-Subscription-Key'] = _key
headers['Content-Type'] = 'application/json' 

json = { 'url': urlImage } 
data = None

params = urllib.parse.urlencode({
    'visualFeatures': 'Categories,Tags,Description,Faces,ImageType,Color,Adult',
})

result = processRequest( json, data, headers, params )

import json

print(json.dumps(result, indent=2, sort_keys=True))



