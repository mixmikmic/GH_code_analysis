import twitter
import json
import tweepy
import sys
import datetime
import detectlanguage
import tweepy
import re
import os

consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret)

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api2 = tweepy.API(auth)

userIDs = []

filename = 'followerIDs/followers5IDs.txt'
f = open(filename, 'r', encoding='utf-8')

l = f.read().split()
userIDs.append(l)

print(len(userIDs[0]))

f.close()

filename = 'followerIDs/followers6IDs.txt'
f = open(filename, 'r', encoding='utf-8')

l = f.read().split()
userIDs.append(l)

print(len(userIDs[1]))

f.close()

filename = 'followerIDs/followers7IDs.txt'
f = open(filename, 'r', encoding='utf-8')

l = f.read().split()
userIDs.append(l)

print(len(userIDs[1]))

f.close()

followers = []
i = 1
for ids in userIDs:
    print(i)
    i = i + 1
    for j in range( int((len(ids) / 100))+1 ):
        if 100 * (j+1) > len(ids):
            if 100 * j == len(ids):
                break
            else:
                users_list = api2.lookup_users(user_ids=ids[(j*100):len(ids)])
        else:
            users_list = api2.lookup_users(user_ids=ids[(j*100):((j+1)*100)])
            
        for user in users_list:
            fs = []
            if hasattr(user, 'status') and user.lang == 'en' and int(user.statuses_count) >= 200 and user.protected == False:
                fs.append(user.id_str)
                fs.append(user.screen_name)
                fs.append(user.statuses_count)
                fs.append(user.lang)
                followers.append(fs)
                
                
print(len(followers))

print(followers[5])

i = 1
for fol in followers:
    i = i + 1
    if i % 50 == 0:
        print(i)
    filename = "tweets/" + fol[1] + ".txt"
    f = open(filename, 'w', encoding='utf-8')

    statuses = api.GetUserTimeline(screen_name = fol[1], count = 200)

    for s in statuses:
        f.write(s.text)
        f.write(" ")

    f.close()

