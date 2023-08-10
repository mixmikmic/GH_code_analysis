import requests
response = requests.get("http://api.open-notify.org/iss-now.json")

print response.status_code

req = requests.get("http://api.open-notify.org/iss-pass.json")
status_code = req.status_code
print status_code
# 400 because didn't include parameters for query

# Dictionary params way
parameters = {"lat": 40.71, "lon": -74}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

# Print the content of the response (the data the server returned)
print response.content

# Write into url
response = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74", params=parameters)

print response.content

# Make the same request we did two screens ago.
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
print response.content

# Get the response data as a Python object.  Verify that it's a dictionary.
json_data = response.json()
print type(json_data) 
print json_data

json_data['response'][0]['duration']

print response.headers

peep_in_space = requests.get("http://api.open-notify.org/astros.json")
print peep_in_space.status_code
print peep_in_space.headers
print "\n"
print peep_in_space.content
print "\n"
print "# people in space: %i" % peep_in_space.json()["number"]

