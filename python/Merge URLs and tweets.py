import gzip
import json
import pandas as pd

from collections import defaultdict, Counter

get_ipython().run_cell_magic('time', '', 'data = []\nmedia_types = defaultdict(int)\nurl_types = defaultdict(int)\nhas_urls = 0\nunique_urls = set()\nwith gzip.open("all_ids.txt.json.gz") as fp:\n    for line in fp:\n        d = json.loads(line.strip())\n        data.append(d)\n        if \'entities\' not in d:\n            continue\n        if \'media\' in d[\'entities\']:\n            m_entities = d[\'entities\'][\'media\']\n            for m in m_entities:\n                m_type = m[\'type\']\n                media_types[m_type] += 1\n        if \'urls\' in d[\'entities\']:\n            m_entities = d[\'entities\'][\'urls\']\n            if len(m_entities) > 0:\n                has_urls += 1\n            for m in m_entities:\n                media_types[\'url\'] += 1\n                m = m[\'expanded_url\']\n                m_type = m.split("/", 3)[2]\n                unique_urls.add((m, m_type))\n                url_types[m_type] += 1\n                \nprint(media_types)\nurl_types = Counter(url_types)\nprint("Of {} tweets, {} contain a total of {} urls with {} unique domains and {} unique urls".format(\n        len(data), has_urls, media_types["url"], len(url_types), len(unique_urls)))')

url_types.most_common(50)

sorted(unique_urls,
                      key=lambda x: url_types[x[1]],
                     reverse=True)[:10]

len(data)

data[0].keys()

data[0][u'source']

data[0][u'is_quote_status']

data[0][u'quoted_status']['text']

data[0]['text']

count_quoted = 0
has_coordinates = 0
count_replies = 0
language_ids = defaultdict(int)
count_user_locs = 0
user_locs = Counter()
count_verified = 0
for d in data:
    count_quoted += d.get('is_quote_status', 0)
    coords = d.get(u'coordinates', None)
    repl_id = d.get(u'in_reply_to_status_id', None)
    has_coordinates += (coords is not None)
    count_replies += (repl_id is not None)
    loc = d['user'].get('location', u'')
    count_verified += d['user']['verified']
    if loc != u'':
        count_user_locs += 1
        user_locs.update([loc])
    language_ids[d['lang']] += 1
    
print count_quoted, has_coordinates, count_replies, count_user_locs, count_verified
print("Of {} tweets, {} have coordinates, while {} have user locations, comprising of {} unique locations".format(
        len(data), has_coordinates, count_user_locs, len(user_locs)
    ))

user_locs.most_common(10)

len(data)

data[0]['user']

df = pd.read_csv("URL_CAT_MAPPINGS.txt", sep="\t")
df.head()

df['URL_EXP_SUCCESS'] = (df.EXPANDED_STATUS < 2)
df.head()

URL_DICT = dict(zip(df[df.URL_CATS != 'UNK'].URL, df[df.URL_CATS != 'UNK'].URL_CATS))
URL_MAPS = dict(zip(df.URL, df.URL_DOMAIN))
URL_EXP_SUCCESS = dict(zip(df.URL, df.URL_EXP_SUCCESS))
len(URL_DICT), df.shape, len(URL_MAPS), len(URL_EXP_SUCCESS)

df.URL.head().values

URL_MAPS['http://bit.ly/1SqTn5d']

found_urls = 0
twitter_urls = 0
total_urls = 0
tid_mapped_urls = []
url_types = defaultdict(int)
for d in data:
    if 'urls' in d['entities']:
            m_entities = d['entities']['urls']
            for m in m_entities:
                total_urls += 1
                m = m['expanded_url']
                m_cats = "UNK"
                if m in URL_DICT:
                    found_urls += 1
                    m_cats = URL_DICT[m]
                elif m.startswith("https://twitter.com") or m.startswith("http://twitter.com"):
                    found_urls += 1
                    twitter_urls += 1
                    m_cats = "socialmedia|twitter"
                else:
                    m_type = "failed_url"
                    if URL_EXP_SUCCESS[m]:
                        m_type = URL_MAPS.get(m, "None.com")
                    """
                    m_type = m.split("/", 3)[2]
                    #m_type = m_type.split("/", 3)[2]
                    if m_type.startswith("www."):
                        m_type = m_type[4:]
                    """
                    url_types[m_type] += 1
                tid_mapped_urls.append((d["id"], m, m_cats))
print "Data: %s, Total: %s, Found: %s, Twitter: %s" % (len(data), total_urls, found_urls, twitter_urls)
url_types = Counter(url_types)
url_types.most_common(10)

url_types.most_common(50)

sum(url_types.values())

tid_mapped_urls[:10]

df_mapped_cats = pd.DataFrame(tid_mapped_urls, columns=["TID", "URL", "CATS"])
df_mapped_cats.head()

df_mapped_cats.to_csv("TID_URL_CATS.txt", sep="\t", index=False)
get_ipython().system(' head TID_URL_CATS.txt')

def extract_meta_features(x):
    u_data = x["user"]
    u_url = u_data['url']
    if u_url is not None:
        u_url = u_data['entities']['url']['urls'][0]['expanded_url']
    return (x["id"],
            x['created_at'],
            x['retweet_count'],
            x['favorite_count'], 
            x['in_reply_to_status_id'] is not None,
            'quoted_status' in x and x['quoted_status'] is not None,
            len(x['entities']['hashtags']),
            len(x['entities']['urls']),
            len(x['entities']['user_mentions']),
            0 if 'media' not in x['entities'] else len(x['entities']['media']), # Has photos
            u_data['id'],
            u_data[u'created_at'],
            u_data[u'listed_count'],
            u_data[u'favourites_count'],
            u_data[u'followers_count'],
            u_data[u'friends_count'],
            u_data[u'statuses_count'],
            u_data[u'verified'],
            u_data[u'location'].replace('\r', ''),
            u_data[u'name'].replace('\r',''),
            u_url
           )
    

extract_meta_features(data[0])

df_meta = pd.DataFrame((extract_meta_features(d) for d in data),
                      columns=["t_id", "t_created", "t_retweets",
                              "t_favorites", "t_is_reply", "t_is_quote",
                              "t_n_hashtags", "t_n_urls", "t_n_mentions",
                              "t_n_media",
                               "u_id", "u_created",
                               "u_n_listed", "u_n_favorites", "u_n_followers",
                               "u_n_friends", "u_n_statuses",
                               "u_is_verified", "u_location", "u_name", "u_url"
                              ])
df_meta.head()

df_meta.dtypes

df_meta[df_meta.u_url.apply(lambda x: x is not None)]["u_url"].head()

df_meta.to_csv("TID_META.txt", sep="\t", index=False, encoding='utf-8')
get_ipython().system(' head TID_META.txt')

df_meta[df_meta.u_url.apply(lambda x: x is not None)]["u_url"].shape

df_meta.shape



