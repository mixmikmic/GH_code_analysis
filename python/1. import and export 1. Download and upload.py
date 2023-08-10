get_ipython().system('pip install tqdm')

import urllib
from tqdm import tqdm
import requests
from requests.auth import HTTPDigestAuth
import json
import os

#simplest version
urllib.urlretrieve ("http://download.thinkbroadband.com/10MB.zip", "10MB.zip")

# Python 3 variant:

from requests import get  
def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

url = "http://download.thinkbroadband.com/10MB.zip"
response = requests.get(url, stream=True)

with open("10MB", "wb") as handle:
    for data in tqdm(response.iter_content()):
        handle.write(data)

# get data from an api call

# visit http://www.icndb.com/api/ to see other options to quesry the Chuck norris jokes database
url = "http://api.icndb.com/jokes/random"

# It is a good practice not to hardcode the credentials. So ask the user to enter credentials at runtime
myResponse = requests.get(url)
#myResponse = requests.get(url,auth=HTTPDigestAuth(raw_input("username: "), raw_input("Password: ")), verify=True)
print (myResponse.status_code)
# For successful API call, response code will be 200 (OK)

jData = json.loads(myResponse.content)
print json.dumps(jData)

#uploading data

with open('output_file', 'wb') as fout:
    fout.write(os.urandom(1024)) 

r = requests.post('http://httpbin.org/post', files={'output_file': open('output_file', 'rb')})
print r

