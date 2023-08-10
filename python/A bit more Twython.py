from twython import Twython
from twython import TwythonError
import json
import hidden
#using TwythonError to get a more helpful error message

# retrieve the keys and secrets
secrets=hidden.oauth()
APP_KEY=secrets['consumer_key']
APP_SECRET=secrets['consumer_secret']
OAUTH_TOKEN=secrets['token_key']
OAUTH_TOKEN_SECRET=secrets['token_secret']

twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
print(twitter)
try:
    user_timeline = twitter.get_user_timeline(screen_name='gdsteam')
except TwythonError as e:
    print(e)

print(user_timeline)

twitter.get_application_rate_limit_status()['resources']['search']

twitter.get_home_timeline()

results = twitter.cursor(twitter.search, q='govuk')
for result in results:
#     print(result['id_str'])
    print(result)

# using the documentation at https://twython.readthedocs.io/en/latest/api.html to try a few things out 
# Tiwtter have (very) recently changed their developer documentation urls so the links through to them
# from Twython no longer work - however, they are still available eg
# https://developer.twitter.com/en/docs/accounts-and-users/follow-search-get-users/api-reference/get-friends-list

twitter.get_friends_list(screen_name='grumpygrandma', count=2)

friendList=twitter.get_friends_list(screen_name='grumpygrandma', count=2)

print(type(friendList))

print(friendList.keys())

print(type(friendList['users']))

for friend in friendList['users']:
#     print(type(friend))
    print('Name is:',friend['name'])
    print('Description is:',friend['description'])

# i have no idea what use this would be... the id is my twitter id
twitter.get_available_trends(id='14394792')

# Full list of Twitter api methoda is at
# https://developer.twitter.com/en/docs/api-reference-index

