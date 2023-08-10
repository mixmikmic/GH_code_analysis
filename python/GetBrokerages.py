import requests
import json

#enter token
Token = ""

#make the request
print ("Get Quovo brokerages")
BASEURL = "https://api.quovo.com/v2/brokerages"
headers = {
    'Authorization': "Bearer " + Token
    }
get_data = requests.get(BASEURL, headers=headers)

#print json data
data = get_data.json()
print (json.dumps(data, indent=4, sort_keys=True))

#write to brokerages.json
json_data = json.dumps(data)
f = open("brokerages.json", "w")
f.write(json_data)
f.close()

