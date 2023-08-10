# import required libraries

import requests
import json

# set key
API_key = ""

# set base url
base_url = ""

# set response format
response_format = ".json"

# set search parameters, the keys are dictated by the API documentation
search_params = {"query":"",
                 "api-key":API_key}       

# make request
r = requests.get(base_url + response_format, params=search_params)

print(r.url)

# inspect the content of the response, parsing the result as text

response_text = r.text
print(response_text[:1000])

# convert JSON response to a Python dictionary

data = json.loads(response_text)
print(data.keys())



