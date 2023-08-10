get_ipython().system('pip install geocoder')

import json
import requests
import geocoder
import pandas as pd
import declarativewidgets
from requests.auth import HTTPBasicAuth
from declarativewidgets.widget_channels import channel

declarativewidgets.init()

creds = {
    "credentials": {
        "username": "3f46c86e-1151-4a18-a804-2de2678558f3",
        "password": "",
        "host": "twcservice.mybluemix.net",
        "port": 443,
        "url": "https://3f46c86e-1151-4a18-a804-2de2678558f3:BaCv3Tg4HF@twcservice.mybluemix.net"
    }
}

auth = HTTPBasicAuth(creds['credentials']['username'], 
                     creds['credentials']['password'])

url = 'https://{}/api/weather/v2'.format(creds['credentials']['host'])

def fetch_forecasts(location='Durham, NC'):
    '''Geocode a location and fetch its 10-day forecast.'''
    # geocode the user string
    g = geocoder.google(location)
    if g.address is None: 
        channel().set('geocoded_location', 'unknown location')
        return
    
    # make the request
    params = {
        'geocode' : '{:.2f},{:.2f}'.format(g.lat, g.lng),
        'units': 'e',
        'language': 'en-US'
    }
    resp = requests.get(url+'/forecast/daily/10day', params=params, auth=auth)
    resp.raise_for_status()
    
    # flatten nested json into a dataframe
    body = resp.json()
    df = pd.io.json.json_normalize(body['forecasts'])
    
    # publish the normalized geocoded location
    channel().set('geocoded_location', g.address)
    
    # return the dataframe sans rows without critical information
    return df.dropna(subset=['day.alt_daypart_name'])

# fetch_forecasts('qweirjoqwer')

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-input/paper-input.html"\n    is=\'urth-core-import\' package=\'PolymerElements/paper-input\'>')

get_ipython().run_cell_magic('html', '', '<template is="urth-core-bind">\n    <paper-input value="{{location}}" label="Enter a location to get a forecast" />\n</template>')

get_ipython().run_cell_magic('html', '', '<template is="urth-core-bind">\n    <urth-core-function ref="fetch_forecasts" arg-location="{{location}}" result="{{df}}" delay="500" auto></urth-core-dataframe>\n    <table class="forecast">\n        <caption>10-day forecast for [[ geocoded_location ]]</caption>\n        <thead>\n            <tr>\n                <th>Day</th>\n                <th>Forecast</th>\n            </tr>\n        </thead>\n        </thead>\n        <tbody>\n            <template is="dom-repeat" items="{{df.data}}">\n                <tr>\n                    <td>{{ item.4 }}</td>\n                    <td><img src="https://raw.githubusercontent.com/IBM-Bluemix/insights-for-weather-demo/master/public/images/weathericons/icon{{ item.13 }}.png" /> {{ item.62 }}</td>\n                </tr>\n            </template>\n        </tbody>\n    </table>\n    <style>\n    .forecast {\n        min-width: 30%;\n        width: 100%;\n    }\n    .forecast img {\n        width: 32px;\n        display: inline;\n    }\n    </style>\n</template>')

