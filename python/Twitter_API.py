from twython import Twython

APP_KEY = 'YOUR APP KEY'
APP_SECRET = 'YOUR APP SECRET'

OAUTH_TOKEN = 'YOUR OAUTH TOKEN'
OAUTH_TOKEN_SECRET = 'YOUR OAUTH SECRET'
twitter = Twython(APP_KEY, APP_SECRET,
                  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

twitter.get_application_rate_limit_status()['resources']['search']

APP_KEY = 'YOUR APP KEY'         #SAME AS ABOVE
APP_SECRET = 'YOUR APP SECRET'   #SAME AS ABOVE
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

print ACCESS_TOKEN

APP_KEY = 'YOUR APP KEY'             #SAME AS ABOVE
ACCESS_TOKEN = 'YOUR ACCESS TOKEN'   #COPY AND PASTE FROM OUTPUT FROM ABOVE COMMAND
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

twitter.get_application_rate_limit_status()['resources']['search']

