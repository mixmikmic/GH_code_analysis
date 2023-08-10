import requests

r = requests.get('http://api.cal-adapt.org/api/')

r.json()

series = requests.get('http://api.cal-adapt.org/api/series/')

# The series json object has many properties. Let's look at count, next and results
series.json()

# Query parameters dict
params = {'name': 'yearly average precipitation', 'pagesize': 100}

# Use params with the url.
response = requests.get('http://api.cal-adapt.org/api/series/', params=params)

# It is a good idea to check there were no problems with the request.
if response.ok:
    data = response.json()
    # Get a list of raster series from results property of data object
    results = data['results']
    # Iterate through the list and print the url property of each object
    for item in results:
        print(item['slug'])



