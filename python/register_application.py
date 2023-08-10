import requests
import json
from arcgis.gis import GIS

gis = GIS(username='mpayson_startups')

app_props = {
    'title': 'My Awesome App',
    'tags': 'these, are, awesome, tags',
    'description': 'An awesome app with awesome tags',
    'type': 'Application'
}
redirect_uris = [
  "https://app.foo.com",
  "urn:ietf:wg:oauth:2.0:oob"
]

app_item = gis.content.add(app_props)
app_item

# note, for now assumes the portal supports https
register_url = '{0}/oauth2/registerApp'.format(gis._portal.resturl)
uri_str = json.dumps(redirect_uris)
register_props = {
    'itemId': app_item.id,
    'appType': 'multiple',
    'redirect_uris': uri_str,
    'token': gis._portal.con._token,
    'f': 'json'
}

# register it!
r = requests.post(register_url, data=register_props)
resp_dict = json.loads(r.text)
if "error" in resp_dict:
    print(resp_dict)
else:
    print("client id: {0}".format(resp_dict["client_id"]))

