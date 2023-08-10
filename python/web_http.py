import requests

url = "https://httpbin.org/html"
response = requests.get(url)
response

request = response.request
for key in request.headers: # The headers in the response are stored as a dictionary.
    print(f'{key}: {request.headers[key]}')

request.method

for key in response.headers:
    print(f'{key}: {response.headers[key]}')

response.status_code

response.text[:100]

post_response = requests.post("https://httpbin.org/post",
                              data={'name': 'sam'})
post_response

post_response.status_code

post_response.text

# This page doesn't exist, so we get a 404 page not found error
url = "https://www.youtube.com/404errorwow"
errorResponse = requests.get(url)
print(errorResponse)

# This specific page results in a 500 server error
url = "https://httpstat.us/500"
serverResponse = requests.get(url)
print(serverResponse)

