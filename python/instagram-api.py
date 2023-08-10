import requests
from IPython.display import display, Image
from keys import instagram_client_id, instagram_client_secret, instagram_redirect_uri, instagram_access_token

# get an access token here if the one in keys.py has expired
url = 'https://api.instagram.com/oauth/authorize/?client_id={}&redirect_uri={}&response_type=token&display=touch&scope={}'
scope = 'likes+relationships+public_content'
oauth_token_url = url.format(instagram_client_id, instagram_redirect_uri, scope)
#print('To get token, visit: {}'.format(oauth_token_url))

#get up to 10 media items recently tagged "#milkyway"
tags = 'milkyway'
media_count = 10
url = 'https://api.instagram.com/v1/tags/{}/media/recent?access_token={}&count={}'
url = url.format(tags, instagram_access_token, media_count)
response = requests.get(url)
json = response.json()

for item in json['data']:
    image_url = item['images']['low_resolution']['url'].split('?')[0]
    display(Image(image_url))

#get recent media within some distance (in meters) of a lat-long point
lat = -33.859172
lng = 151.209749
dist = 1000
url = 'https://api.instagram.com/v1/media/search?lat={}&lng={}&access_token={}&distance={}'
url = url.format(lat, lng, instagram_access_token, dist)
response = requests.get(url)
json = response.json()

for item in json['data']:
    image_url = item['images']['low_resolution']['url'].split('?')[0]
    display(Image(image_url))



