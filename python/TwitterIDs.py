import twitter
import json
import tweepy
import sys
import datetime
import detectlanguage

consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""
    

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret)

#ids = api.GetFollowerIDs(screen_name='trtworld', cursor=None, stringify_ids=True, count=5, total_count=5)

f = open('followers1.txt', 'w', encoding='utf-8')

ncur, pcur, ids = api.GetFollowerIDsPaged(screen_name='trtworld', cursor=-1, stringify_ids=True)

for id in ids:
    f.write(id)
    f.write("\n")

    
f.close()

i = 2
while(ncur != 0):
    print(str(i) + ' ' + str(ncur))
    filename = 'followers' + str(i) + '.txt'
    i = i + 1
    f = open(filename, 'w', encoding='utf-8')
    ncur, pcur, ids = api.GetFollowerIDsPaged(screen_name='trtworld', cursor=ncur, stringify_ids=True)
    for id in ids:
        f.write(id)
        f.write("\n")
        
    f.close()



