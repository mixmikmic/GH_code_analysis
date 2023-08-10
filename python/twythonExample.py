from twython import Twython

APP_KEY = 'G9gsmmCEgoL5Ja2IUVetEiE8w'
APP_SECRET = 'LcWh4FrdlrXM1X3PJEv9Z7mxO0hJjzNAvttoENGOxle4Xqbk3A'
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()

print(ACCESS_TOKEN)

APP_KEY = 'G9gsmmCEgoL5Ja2IUVetEiE8w'
ACCESS_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAOiw2AAAAAAAv975fTxwbyP4mG2rcEVuIwNysvs%3DjIXdEhlyq9bTmvPa6TsJmGnzLjgP75cGA3MwptdCt8b7t3Iabp'   #COPY AND PASTE FROM OUTPUT FROM ABOVE COMMAND
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

twitter.get_application_rate_limit_status()['resources']['search']

twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

twitter.search(q='#LocalGovCamp')

twitter.search(q='LocalGovCamp', result_type='popular')





