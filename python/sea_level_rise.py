import requests 
import json
headers = {'ContentType': 'json'}

api = 'http://api.cal-adapt.org/api'
resource = 'rstores'

search_str = 'Sea+Level+Rise'
params = {'name': search_str, 'pagesize': 20}
params_str = "&".join("%s=%s" % (k,v) for k,v in params.items())
url = api + '/' + resource + '/' + '?' + params_str
url

response = requests.get(url, headers=headers)
data = response.json()
results = data['results']
for item in results:
    print(item['url'], item['name'])

lat = 38.106914
lng = -121.456679
point = '{"type":"Point","coordinates":[' + str(lng) + ',' + str(lat) + ']}' #geojson format
point = 'POINT(' + str(lng) + ' ' + str(lat) + ')' #WKT format

params = {'g': point}
params_str = "&".join("%s=%s" % (k,v) for k,v in params.items())
print('Depth of flooding depth (meters) for point location:', lat, lng, sep=' ')
print()
for item in results:
    url = item['url'] + '?' + params_str
    response = requests.get(url, headers=headers)
    # Request will return a server error if point is outside spatial extent of SLR dataset
    if response.status_code == requests.codes.ok:
        data = response.json()
        if float(data['image']) == float(data['nodata']):
            print(item['name'], 'No Data', sep=' - ')
        else:
            print(item['name'], data['image'], sep=' - ')
    else:
        print(item['name'], 'Outside extent', sep=' - ')



