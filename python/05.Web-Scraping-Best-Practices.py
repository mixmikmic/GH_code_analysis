# sometimes things work more smoothly if we change the User-Agent in the request headers
headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5)'}

# In the previous example, we were constructing urls manually
url = 'http://www.boxofficemojo.com/movies/?id=soundofmusic.htm&adjust_yr=2017'
requests.get(url)

# we could re-write the same request with params
base_url = 'http://www.boxofficemojo.com/movies/'
params = {
    'id':'soundofmusic',
    'adjust_yr': 2017
}

response = requests.get(url=base_url, headers = headers, params = params)

try:
    r = requests.get(url, timeout=10)
except Exception as e:
    print(e.message)

