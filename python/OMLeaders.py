from LibOM.Tools import *

ClientWT = WatchTower()
influencer_names = ClientWT.retrieve_influencers()

print len(influencer_names)
print "_"*25
for inf in influencer_names: print inf

Credentials = {}
Credentials['Consumer_Key'] = "your key" 
Credentials['Consumer_Secret'] = "your secret"
Credentials['Access_Token'] = "your token" 
Credentials['Access_Token_Secret'] = "your token secret"

ClientTwitter = Twitter(Credentials)

debates = ClientTwitter.retrieve_tweets(influencer_names[0:5], 50)

for user,data in debates.items():
    print "_"*80
    print "Username: ", user
    print "Number of tweets: ", data['ntweets']
    print "-"*20
    for tweet in data['content'].split("\n"): print tweet

MD = MakerDictionary()

for name, code in MD.categories.items(): print name, code

# The tentative list of expressions under each category:
for category, mappings in MD.table_Mfeatures.items():
    print "_"*60
    print MD.get_category_name(category), " :: ", category
    print sorted(mappings['content'])

SB = ScoreBoard()
for user in debates.keys():
    ntweets = debates[user]['ntweets']
    text = debates[user]['content']
    nmappings, nwords, counts = extract_features(text, MD)
    features = {"ntweets":ntweets, "nwords":nwords, "nmappings":nmappings, "counts":counts}
    SB.add_actor(user, features)
    print user, ntweets, nwords, nmappings
    print counts

for k, v in SB.table.items():
    print "_" * 20
    print k, v["ntweets"], v["nwords"], v["nmappings"]
    for type in v['scores'].keys():
        print type, v['scores'][type]

SB.compute_rankings('0', 'per_word')
SB.compute_rankings('5', 'per_word')
SB.compute_rankings('all', 'per_word')

def print_score_table(rankings):
    for rtype, scores in rankings.items():
        print "_"*20
        print rtype
        for user,score in list(scores): print user, " : ", score

print_score_table(SB.rankings)

SB.get_rankings_one('7', 'per_word')
SB.get_rankings_one('1', 'raw')
SB.get_rankings_one('all', 'per_tweet')
print_score_table(SB.rankings)

username = '3DPrintGirl'
score = SB.get_score_one(username, 'all', 'per_tweet')
print score

cat_codes = MD.categories.values()
sub_scores = SB.get_scores(username, cat_codes, 'per_tweet')
print sub_scores

def get_anew_user_score(username):
    tdata = ClientTwitter.accumulate_auser_tweets(username)
    ntweets = tdata['ntweets']
    if not ntweets: return  
    text = tdata['content']
    nmappings, nwords, counts = extract_features(text, MD)
    features = {"ntweets":ntweets, "nwords":nwords, "nmappings":nmappings, "counts":counts}
    SB.add_actor(username, features)
    score = SB.get_score_one(username, 'all', 'per_tweet')
    categories = MD.categories.values()
    sub_scores = SB.get_scores(username, categories, 'per_tweet')
    return ntweets, sub_scores

for score in SB.get_rankings_one('all', 'per_tweet'): print score

tweetdata = get_anew_user_score('BernieSanders')
print "Number of tweets used: ", tweetdata[0]
print "scores (theme:score)::"
for theme,score in tweetdata[1].items():
    print theme, ":", score

print SB.get_score_one('BernieSanders', 'all', 'per_tweet')

for score in SB.get_rankings_one('all', 'per_tweet'): print score

