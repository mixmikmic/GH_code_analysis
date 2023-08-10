import requests
import userconfig

def get_token(username, password, url):
    '''
    Production server: https://home-sales-data-api.herokuapp.com
    Dev server: http://home-sales-data-api-dev.herokuapp.com
    ''' 
    baseurl = url
    data = {"username":username, 
            "password":password}
    r = requests.post(baseurl + '/token/auth/', data = data)
    return r.json()['token']
    
token = get_token(config.LOCALUSER, config.LOCALPASS, config.LOCALURL)
print(token)

import requests
from datetime import datetime
baseurl = userconfig.LOCALURL
headers = {"Authorization": "Bearer " + token}

data = {"listing_timestamp": str(datetime.now()),
        "listing_type": 'F', # for sale
        "price": 123456,
        "size_units": 'I',
        "raw_address": "1701 Wynkoop St, Denver, CO 80202"
       }

r = requests.post(baseurl + '/api/property/', data = data, headers=headers)
r.json()

import requests
from datetime import datetime
import userconfig

baseurl = userconfig.LOCALURL
data = {'min_price': '150000',
        'max_price': '200000',
        'min_bedrooms': '1',
        'max_bedrooms': '3',
        'min_bathrooms': '1',
        'max_bathrooms': '2',
        'min_car_spaces': '1',
        'max_car_spaces': '2',
        'address_contains': 'main',
        'limit': '1',
        'offset': '1'
          }

r = requests.get(baseurl + '/api/property/', data = data)
r.json()



