journos = set()

for line in open("journos.txt"):
    journos.add(line.strip().lower())

len(journos)

import twarc

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

t = twarc.Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

for tweet in t.filter("trump,clinton,sanders"):
    if 'user' not in tweet:
        continue
    user = tweet['user']['screen_name'].lower()
    if user in journos:
        print(user, tweet['text'])

for line in open("tweets.json"):
    tweet = json.loads(line)
    if 'user' not in tweet:
        continue
    user = tweet['user']['screen_name'].lower()
    if user in journos:
        print(user, tweet['text'])

