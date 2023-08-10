import sys
import string
import simplejson
from twython import Twython
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil import parser
import time

#FOR OAUTH AUTHENTICATION -- NEEDED TO ACCESS THE TWITTER API
t = Twython(app_key='APP_KEY', 
    app_secret='APP_SECRET',
    oauth_token='OAUTH_TOKEN',
    oauth_token_secret='OAUTH_TOKEN_SECRET')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# ego_screenname -- the person you're looking to query for
#ego_screenname = "jugander"

# "Bot scholars" that may be useful to look at
ego_screenname = "gilgul"
#ego_screenname = "suneman"
#ego_screenname = "yy"

ego_id = t.lookup_user(screen_name=ego_screenname)[0]['id_str']

# Get the follower list. 
# Rate limiting caps this at ~80000 followers every 900 seconds. Hacky pauses added. 
follower_list = []
nextcurs = -1
while (nextcurs != 0):
    remaining = t.get_lastfunction_header(header='x-rate-limit-remaining')
    secs = int(t.get_lastfunction_header(header='x-rate-limit-reset')) - int(datetime.now().timestamp())
    if ((int(remaining) == 0) and (secs >= 0)):
        print("Waiting " + str(secs) + " seconds...")
        time.sleep(secs+3)
    
    follower_object = t.get_followers_ids(user_id = ego_id, count = 5000, cursor = nextcurs)
    follower_list += follower_object['ids']
    nextcurs = follower_object['next_cursor']
    print(str(len(follower_list))) 

# Status of API allowances from last call
print("Allowed " + str(t.get_lastfunction_header(header='x-rate-limit-remaining')) +
      " more follow requests over the next " + 
      str(int(t.get_lastfunction_header(header='x-rate-limit-reset')) - int(datetime.now().timestamp())) +
        " seconds.")

len(follower_list)

# Once you have the followers, now need to gather the create_times of those followers

# Optionally downsample
if (len(follower_list) > 20000):
    downsample_rate = 1
else:
    downsample_rate = 1

# Begin
i = downsample_rate
k = 0
ts_list_mk = []
for follower_chunk in chunker(follower_list,100):
    k += 100
    if (i == downsample_rate):
        i = 1
    else:
        i += 1
        continue
    
    if isinstance(t.get_lastfunction_header(header='x-rate-limit-reset'),str):
        remaining = t.get_lastfunction_header(header='x-rate-limit-remaining')
        secs = int(t.get_lastfunction_header(header='x-rate-limit-reset')) - int(datetime.now().timestamp())
        if ((int(remaining) == 0) and (secs >= 0)):
            print("Waiting " + str(secs) + " seconds...")
            time.sleep(secs+2)

    x = t.lookup_user(user_id=follower_chunk)
    ts_mk = [parser.parse(u['created_at']).timestamp() for u in x]
    ts_list_mk += ts_mk
    print(str(len(ts_list_mk)) + " " + str(k)) 

ts_list_dt = [datetime.fromtimestamp(x) for x in ts_list_mk]
ts_list_dt.reverse()

# API status again, from last call
print("Allowed " + str(t.get_lastfunction_header(header='x-rate-limit-remaining')) +
      " more user requests over the next " + 
      str(int(t.get_lastfunction_header(header='x-rate-limit-reset')) - int(datetime.now().timestamp())) +
        " seconds.")

len(ts_list_dt)

# Histogram of create_times, sometimes reveals things:

plt.hist(ts_list_dt,bins=100);
plt.show()

# The "createtime fingerprint"

start=1
stop=len(ts_list_dt)
plt.plot(range(len(ts_list_dt[start:stop])),ts_list_dt[start:stop],'r.',markersize=2);
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.xlabel('follower number')
plt.ylabel('account creation date')
plt.title(ego_screenname)
plt.show()

# Clear Twython object from notebook
t = None

