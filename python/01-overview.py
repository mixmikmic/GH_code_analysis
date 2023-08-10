import collections
import json
import sys
import os
import requests
import bs4
import re
import matplotlib.pyplot as plt
import pandas as pd

# add penemue to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from utils import twiterate
from utils import Collect

get_ipython().magic('matplotlib inline')

j = json.load(open('../data/journalists.json'))
journalists = [user['id_str'] for user in Collect(lists=j).members]    

with open('../data/output/journalist_id_strs.json', 'w') as f:
    json.dump(journalists, f)

o = json.load(open('../data/organisations.json'))
organisations = [user['id_str'] for user in Collect(lists=o).members]

with open('../data/output/organisation_id_strs.json', 'w') as f:
    json.dump(organisations, f)

def pie(joi, ooi):
    """Generate a pie chart of tweet distribution
    
    :param joi: A python list
                The list should contain 
                appropriate twitter profile 
                data.
    
    :param ooi: A python list
                The list should contain 
                appropriate twitter profile 
                data.
    """
    
    # calculate mean percentage
    joi_size = len(joi)
    ooi_size = len(ooi)
    total = joi_size + ooi_size

    joi_mean = (joi_size / total) * 100
    ooi_mean = (ooi_size / total) * 100

    # data to plot
    sizes = [joi_mean, ooi_mean]
    labels = "Journalists", "Organisations"
    colors = ['lightskyblue', 'lightcoral']
    
    # plot
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        shadow=True, startangle=90)

    plt.axis('equal')
    plt.show()

def bar(joi, ooi, label):
    """Generate a bar chart of the top 10
    occurances within the provided data
    
    :param joi:   A python list
                  The list should contain 
                  appropriate twitter profile 
                  data.
    
    :param ooi:   A python list
                  The list should contain 
                  appropriate twitter profile 
                  data.
                
    :param label: String
                  Label for the y-axis
    """
    
    # create list of users
    id_strs = ["@%s" % screen_name for screen_name in joi]
    id_strs += ["@%s" % screen_name for screen_name in ooi]

    # count occurances
    counter = collections.Counter(id_strs)
    most_common = counter.most_common(10)

    # data to plot
    labels, y = zip(*most_common)
    x = range(len(labels))
    
    # plot
    plt.bar(x, y, alpha=0.5)
    plt.xticks(x, labels, rotation='90')
    plt.ylabel(label)
    plt.show()

def get_original(tweet, search):
    """Identify original tweet
    
    :param tweet:  A tweet object
    :param search: A list of str_ids to search
    
    :return: A twitter screen_name as a string
    """
    
    if (tweet["user"]["id_str"] in search 
        and tweet["in_reply_to_status_id_str"] is None 
        and "retweeted_status" not in tweet):
            return tweet["user"]["screen_name"]

joi_originals = twiterate(lambda tweet : get_original(tweet, journalists))
ooi_originals = twiterate(lambda tweet : get_original(tweet, organisations))

pie(joi_originals, ooi_originals)

bar(joi_originals, ooi_originals, label="Original Tweets")

with open('../data/output/joi_originals.json', 'w') as f:
    json.dump(joi_originals, f)

with open('../data/output/ooi_originals.json', 'w') as f:
    json.dump(ooi_originals, f)

def get_retweet(tweet, search):
    """Identify a retweet
    
    :param tweet:  A tweet object
    :param search: A list of str_ids to search
    
    :return: A twitter screen_name as a string
    """
    
    if (tweet["user"]["id_str"] in search 
        and "retweeted_status" in tweet):
            return tweet["user"]["screen_name"]

joi_retweets = twiterate(lambda tweet : get_retweet(tweet, journalists))
ooi_retweets = twiterate(lambda tweet : get_retweet(tweet, organisations))

pie(joi_retweets, ooi_retweets)

bar(joi_retweets, ooi_retweets, label="Retweets")

with open('../data/output/joi_retweets.json', 'w') as f:
    json.dump(joi_retweets, f)

with open('../data/output/ooi_retweets.json', 'w') as f:
    json.dump(ooi_retweets, f)

def get_reply(tweet, search):
    """Identify a reply
    
    :param tweet: A tweet object
    :param search: A list of str_ids to search
    
    :return: A twitter screen_name as a string
    """
    
    if (tweet["user"]["id_str"] in search 
        and tweet["in_reply_to_status_id_str"] is not None):
            return tweet["user"]["screen_name"]

joi_replies = twiterate(lambda tweet : get_reply(tweet, journalists))
ooi_replies = twiterate(lambda tweet : get_reply(tweet, organisations))

pie(joi_replies, ooi_replies)

bar(joi_replies, ooi_replies, label="Replies")

with open('../data/output/joi_replies.json', 'w') as f:
    json.dump(joi_replies, f)

with open('../data/output/ooi_replies.json', 'w') as f:
    json.dump(ooi_replies, f)

def get_url(tweet):
    """Extract urls from a tweet
    
    :param tweet: A tweet object
    :return: A url as a string
    """
    
    for url in tweet["entities"]["urls"]:
        return url["expanded_url"]

urls = twiterate(get_url)

def get_title(url):
    """Get the contents of title tag from
    the webpage at the link
    
    :param url: String containing a link
    :return: String containing page title
    """
    
    # get title text
    html = requests.get(url)
    page = bs4.BeautifulSoup(html.text, "html.parser")
    title = page.title.string if page.title != None else ""
    # remove markdown grammar
    title = re.sub(r"\r|\n|\||\s+", " ", title)
    # remove leading & trailing whitespace
    title = title.lstrip().rstrip()
    
    return title

lc = collections.Counter(urls)
mcl = [(url, get_title(url), occ) 
       for (url, occ) in lc.most_common(10)]

pd.DataFrame(mcl, 
             range(1, len(mcl) + 1), 
             ['Link', 'Title', 'Occurances'])

pd.DataFrame([len(urls), len(set(urls))], 
             ['Total', 'Unique'], 
             ['Links'])

def get_hashtag(tweet):
    """Extract hashtags from a tweet
    
    :param tweet: A tweet object
    :return: A list of hashtags
    """
    
    for hashtags in tweet["entities"]["hashtags"]:
        return hashtags["text"]
    
hashtags = twiterate(get_hashtag)
hc = collections.Counter(hashtags)
mch = [("#" + key, value) 
       for (key, value) in hc.most_common(10)]

labels, y = zip(*mch)
x = range(len(labels))

plt.bar(x, y, alpha=0.5)
plt.xticks(x, labels, rotation='90')
plt.ylabel("Tweets")
plt.show()



