bot_usernames = [];
nonbot_usernames = [];
with open('unprocessed_dataset/bots_data.csv', encoding='latin-1') as f:
    f.readline()
    for line in f:
        try:
            if (line[0] != ' ' and line.split(',')[2][0] != "\""):
                bot_usernames.append(line.split(',')[2])
        except:
            continue
with open('unprocessed_dataset/nonbots_data.csv', encoding='latin-1') as f:
    for line in f:
        try:
            if (line[0] != ' ' and line.split(',')[2][0] != "\""):
                nonbot_usernames.append(line.split(',')[2])
        except:
            continue

len(bot_usernames), len(nonbot_usernames)

from twitter import *

from api_keys import *

twitter = Twitter(auth = OAuth(access_key, access_secret, consumer_key, consumer_secret))

def add_accounts_info(accounts, output):
    temp = twitter.users.lookup(screen_name=','.join(accounts))
    for account in temp:
        output.append(account)

bots_info = []
count = 100

while (count<len(bot_usernames)):
    add_accounts_info(bot_usernames[count-100:count], bots_info)
    count += 100
add_accounts_info(bot_usernames[count-100:len(bot_usernames)], bots_info)

len(bots_info)

nonbots_info = []
count = 100

while (count<len(nonbot_usernames)):
    add_accounts_info(nonbot_usernames[count-100:count], nonbots_info)
    count += 100
add_accounts_info(nonbot_usernames[count-100:len(nonbot_usernames)], nonbots_info)

len(nonbots_info)

import pickle
with open('processed_dataset/bot_accounts.pkl', 'wb') as f:
    pickle.dump(bots_info, f)
    
with open('processed_dataset/nonbot_accounts.pkl', 'wb') as f:
    pickle.dump(nonbots_info, f)

