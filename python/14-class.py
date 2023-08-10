api_key = "XXXX"
api_secret = "XXXX"
access_token = "XXXX"
token_secret = "XXXX"

import twython

twitter = twython.Twython(api_key, api_secret, access_token, token_secret)

twitter.update_status(status="")

# Application mode 
twitter = twython.Twython(api_key, api_secret)
auth = twitter.get_authentication_tokens()
print("Log into Twitter as the user you want to authorize and visit this URL:")
print("\t" + auth['auth_url'])

# We copy the pin number provided when we visit the authorization URL 
pin = "XXXX"

twitter = twython.Twython(api_key, api_secret, auth['oauth_token'], auth['oauth_token_secret'])
tokens = twitter.get_authorized_tokens(pin)

new_access_token = tokens['oauth_token']
new_token_secret = tokens['oauth_token_secret']
print("your access token:", new_access_token)
print("your token secret:", new_token_secret)

# these tokens last forever! we can write a new application to post using this twitter bot. 
# But, if we wanted to post to a different account, we would need to redo this process. 

twitter = twython.Twython(api_key, api_secret, new_access_token, new_token_secret)

twitter.update_status(status="hello!")

import pg8000
lakes = []

conn = pg8000.connect(database="mondial")
cursor = conn.cursor()

cursor.execute("SELECT name, area, depth, elevation, type, river from lake")
for row in cursor.fetchall():
    lake = {'name': row[0],
           'area': int(row[1]),
           'depth': int(row[2]),
           'elevation': int(row[3]),
           'type': row[4], 
           'river': row[5]}
    lakes.append(lake)

len(lakes)

# Saving your data as json so you don't have to rely on your SQL database
import json
json.dump(lakes, open("lakes.json", "w"))

sentences = { 'area': 'The area of {} is {} square kilometers.',
            'depth': 'The depth of {} is {} meters.',
            'elevation': 'The elevation of {} is {} meters.',
            'type': 'The type of {} is {}.',
            'river': '{} empties into a river named {}.',}

import random

lake = random.choice(lakes)
col = random.choice(list(lake.keys()))

print(lake)
print(col)

sentence_template = sentences[col]
output = sentence_template.format(lake['name'], lake[col])
output

# Let's make it a function! 

def random_lake_sentence(lakes, sentences):
    
    lake = random.choice(lakes)
    col = random.choice(list(lake.keys()))

    sentence_template = sentences[col]
    output = sentence_template.format(lake['name'], lake[col])
    return output

for i in range(10):
    print(random_lake_sentence(lakes, sentences))
    
# key error: name!
# we don't have a sentence template for name

def random_lake_sentence(lakes, sentences):
    
    lake = random.choice(lakes)
    
    possible_keys = [k for k in lake.keys() if k != 'name']
    col = random.choice(possible_keys)

    sentence_template = sentences[col]
    output = sentence_template.format(lake['name'], lake[col])
    return output

for i in range(100):
    print(random_lake_sentence(lakes, sentences))

def random_lake_sentence(lakes, sentences):
    
    lake = random.choice(lakes)
    
    possible_keys = [k for k in lake.keys() if k != 'name' and lake[k] is not None]
    col = random.choice(possible_keys)

    sentence_template = sentences[col]
    output = sentence_template.format(lake['name'], lake[col])
    return output

for i in range(100):
    print(random_lake_sentence(lakes, sentences))

flare = ['Wow!', "Cool, huh?", 'Now you know.', 'WHAAAAT', 'Neat-o.']

output = random_lake_sentence(lakes, sentences) + " " + random.choice(flare)
twitter.update_status(status=output)



