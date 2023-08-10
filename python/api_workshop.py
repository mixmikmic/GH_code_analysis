import requests

USERNAME = '' # Enter your own username here

url = "http://api.geonames.org/searchJSON"
Q = {
    'q':'Cambridge',
    'username':USERNAME
}

Q['q']

Q['username'] # This also works when we've set the value to another variable

Q['q'] = 'Somerville' # You can also set the value of a key like you would a variable

R = requests.get(url,params=Q)

R.url

R.json()

Q['country'] = 'US'
Q['adminCode1'] = 'MA' 
Q['featureCode'] = 'PPL' # These keys didn't exist until we set them to something! We can create keys this way.
Q

R = requests.get(url,params=Q)

R.url

R.json()

Q

result = R.json()

type(result)

result['geonames'][0] # Here is the first response, since list slicing starts counting from 0

print(result['geonames'][0]['lat'], result['geonames'][0]['lng']) # Prints the lat and the long from the first result

def search_us_place(placename,state_abbrev):
    url = "http://api.geonames.org/searchJSON"
    Q = {
        'q':placename,
        'country':'US',
        'username':USERNAME,
        'adminCode1':state_abbrev
    }
    results = requests.get(url,params=Q).json()
    top_result = results['geonames'][0]
    return top_result['lat'], top_result['lng']

test = search_us_place('Worcester','MA')
print(type(test))
print(test)

def myfunc():
    return 0

import pandas as pd

df = pd.read_csv('botanical_gardens.csv')
df.head()

df = pd.read_csv('botanical_gardens.csv',sep='\t') # Working file opener with separator as tabs
df.head()

import time

def search_us_place(placename,state_abbrev):
    url = "http://api.geonames.org/searchJSON"
    Q = {
        'q':placename,
        'country':'US',
        'username':USERNAME,
        'adminCode1':state_abbrev
    }
    print('making request for {}, {}'.format(placename,state_abbrev))
    R = requests.get(url,params=Q)
    print(R.url)
    results = R.json()
    top_result = results['geonames'][0]
    print('got result for {}, {}'.format(placename,state_abbrev))
    time.sleep(1)
    return top_result['lat'], top_result['lng']

search_us_place('Boston','MA') # Now we'll run this new function

latLngs = df.City.apply(lambda x: search_us_place(x,'MA'))

latLngs

df.City

df['lat'],df['lng'] = zip(*latLngs)

df.head()

df.to_csv('botanical_gardens_with_location.csv',index=None)

# Put your code here, adding more cells as needed

omeka_api_key = '' # We'll give you a key to use for the site

# We're not using the key yet, since we're just viewing public information
R = requests.get('http://demo.omeka.fas.harvard.edu/api/items/36')
demo_item = R.json()

demo_item

def make_item(element_texts):
    """
    Takes a dictionary with format {element_id:element_text, ...}
    """
    base_item = {
        'element_texts':[],
        'featured': False,
        'public': True,
    }
    for _id, text in element_texts.items():
        element = {
            'element': { 'id': int(_id) },
            'text': text,
            'html': True
        }
        base_item['element_texts'].append(element)
    return base_item

test = {
    50: 'A Test Item',
    41: "The description of the test item. It might be a bit longer, which is fine since it won't be used as a page title or anything."
}
test_item = make_item(test)
print(test_item)

R = requests.post('http://demo.omeka.fas.harvard.edu/api/items',json=test_item, params={'key':omeka_api_key})

R.json()

def add_item_to_omeka(element_texts):
    item = make_item(element_texts)
    R = requests.post('http://demo.omeka.fas.harvard.edu/api/items',json=item, params={'key':omeka_api_key})
    return R.json()

def row_to_omeka(row):
    title = row['Name']
    description = "Located in {}, MA ({},{})".format(row['City'],row['lat'],row['lng'])
    element_texts = {
        '50': title,
        '41': description,
        '39': 'Your name here!'
    }
    response = add_item_to_omeka(**element_texts)
    return response

df.apply(row_to_omeka,axis=1)

print(zip([1,2],[3,4],[5,6],[7,8]))

for z in zip([1,2],[3,4],[5,6],[7,8]):
    print(z)

test = [[1,2],[3,4],[5,6],[7,8]]
for z in zip(test):
    print(z)

for z in zip(test[0],test[1],test[2],test[3]):
    print(z)

for z in zip(*test):
    print(z)

for ll in latLngs:
    print(ll)

for z in zip(*latLngs):
    print(z)

