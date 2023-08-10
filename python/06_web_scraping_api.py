import requests 
import json 

url = 'https://content.guardianapis.com/sections?api-key=test'
req = requests.get(url)
src = req.text 

sections = json.loads(src)['response']

print sections.keys()

print json.dumps(sections['results'][0], indent=2, sort_keys=True)

for result in sections['results']: 
    if 'tech' in result['id'].lower(): 
        print result['webTitle'], result['apiUrl']

# Specify the arguments
args = {
    'section': 'technology', 
    'order-by': 'newest', 
    'api-key': 'test', 
    'page-size': '100'
}

# Construct the URL
base_url = 'http://content.guardianapis.com/search'
url = '{}?{}'.format(
    base_url, 
    '&'.join(["{}={}".format(kk, vv) for kk, vv in args.iteritems()])
)

# Make the request and extract the source
req = requests.get(url) 
src = req.text

print 'Number of byes received:', len(src)

response = json.loads(src)['response']
print 'The following are available:\n ', sorted(response.keys())

assert response['status'] == 'ok'

print json.dumps(response['results'][0], indent=2, sort_keys=True)

for result in response['results']: 
    print result['webUrl'][:70], result['webTitle'][:20]





