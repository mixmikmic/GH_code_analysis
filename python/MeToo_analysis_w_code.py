# Import modules & data
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

# You'll need to set the path to the data that you have pulled from the twitter API
df = pd.read_csv('tweets/metoo_11_09.csv')
df['dt'] = pd.to_datetime(df['dt'])
hour_counts = df.groupby(['dt'])['id'].count()

# Plot time series of total number of tweets
hour_counts.plot(lw=3, figsize=(15,10), rot=60);
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Date', fontsize=20);
# plt.rc('xtick',labelsize=20)
# plt.rc('ytick',labelsize=20)

# Plot restricted (temporally) time series to view periodicity
hour_counts.loc['2017-10-23':'2017-10-25'].plot(lw=3, figsize=(15,10), rot=60);
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Datetime', fontsize=20);

# Group by whether it is a retweet or not
df['retweet'] = df.retweeted_status.isnull() == 0 ## is it a retweet or not?
df['retweet'] = df['retweet'].map({1: 'Yes', 0: 'No'}) # Map to strings that mean something
hour_counts_by_rt = df.groupby(['dt', 'retweet'])['id'].count()

# Plot two time series: retweets and original tweets
hour_counts_by_rt.unstack(level=1).plot(lw=3, rot=60, figsize=(15,10));
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Datetime', fontsize=20);

# Plot time series of original tweets
hour_counts = df[df['retweet'] == 'No'].groupby(['dt'])['id'].count()
hour_counts.plot(lw=3, figsize=(15,10), rot=60);
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Datetime', fontsize=20);

# Plot barplot of retweets and original tweets
df.groupby(['retweet'])['id'].count().plot(figsize=(9,6), kind='bar');
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Retweet', fontsize=20);

# How many tweets were retweeted > 1,000 times? 
yo = df.sort_values(by=['retweet_count'], ascending=False)
print(str(sum(yo.retweet_count > 1000)) + ' tweets resulted from original tweets retweeted > 1,000 times.')
rt_list = yo[yo.retweet_count > 1000].retweeted_status.tolist()
_ = []
for c in rt_list:
    try:
        _.append(int(c[55:73]))
    except:
        pass

print(str(len(set(_))) + ' original tweets accounted for these.')
print('They accounted for ' + str(len(_)/len(df)) + '% of total tweets.')
rt_list = yo[yo.retweet_count > 100].retweeted_status.tolist()
__ = []
for c in rt_list:
    try:
        __.append(int(c[55:73]))
    except:
        pass
# How many tweets were retweeted > 100 times? 
print(str(sum(yo.retweet_count > 100)) + ' tweets resulted from original tweets retweeted > 100 times.')
rt_list = yo[yo.retweet_count > 100].retweeted_status.tolist()
_ = []
for c in rt_list:
    try:
        _.append(int(c[55:73]))
    except:
        pass

print(str(len(set(_))) + ' original tweets accounted for these.')
print('They accounted for ' + str(len(_)/len(df)) + '% of total tweets.')

# Top 5:
from collections import OrderedDict
list(OrderedDict.fromkeys(_))[:5]

class Tweet(object):
    def __init__(self, embed_str=None):
        self.embed_str = embed_str

    def _repr_html_(self):
        return self.embed_str

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Reminder that if a woman didn&#39;t post <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a>, it doesn&#39;t mean she wasn&#39;t sexually assaulted or harassed. Survivors don&#39;t owe you their story.</p>&mdash; Alexis Benveniste (@apbenven) <a href="https://twitter.com/apbenven/status/919902089110872064?ref_src=twsrc%5Etfw">October 16, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">For my wife, for my daughters, for all women...I stand with all of you.  This has gotta change.  <a href="https://twitter.com/hashtag/metoo?src=hash&amp;ref_src=twsrc%5Etfw">#metoo</a> <a href="https://twitter.com/hashtag/nomore?src=hash&amp;ref_src=twsrc%5Etfw">#nomore</a></p>&mdash; Jensen Ackles (@JensenAckles) <a href="https://twitter.com/JensenAckles/status/920149248880009217?ref_src=twsrc%5Etfw">October 17, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">For those carrying their <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> with them silently, you are loved, cherished, and believed. You do not owe your story to anyone.</p>&mdash; Grace Starling (@GraceStarling4) <a href="https://twitter.com/GraceStarling4/status/919756449965838336?ref_src=twsrc%5Etfw">October 16, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr"><a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a>. I was 14, he was 36. I may be Deaf, but silence is the last thing you will ever hear from me. <a href="https://t.co/hLmBJ7PgmK">pic.twitter.com/hLmBJ7PgmK</a></p>&mdash; Marlee Matlin (@MarleeMatlin) <a href="https://twitter.com/MarleeMatlin/status/920453826364235776?ref_src=twsrc%5Etfw">October 18, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="und" dir="ltr"><a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> <a href="https://t.co/VWssdltU3n">https://t.co/VWssdltU3n</a></p>&mdash; Monica Lewinsky (@MonicaLewinsky) <a href="https://twitter.com/MonicaLewinsky/status/919732300862181377?ref_src=twsrc%5Etfw">October 16, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

# Top 5 retweeted in data collected
from collections import Counter
d = Counter(_).most_common(5)
d

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">this is why I had to leave Crystal Castles. here is my story: <a href="https://t.co/bs9aJRwgms">https://t.co/bs9aJRwgms</a> <a href="https://twitter.com/hashtag/metoo?src=hash&amp;ref_src=twsrc%5Etfw">#metoo</a></p>&mdash; ALICE GLASS (@ALICEGLASS) <a href="https://twitter.com/ALICEGLASS/status/922875671021436928?ref_src=twsrc%5Etfw">October 24, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">There is no easy way to tell you. But it’s time. <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> <a href="https://t.co/HnwugEWtJF">https://t.co/HnwugEWtJF</a></p>&mdash; Breanna Stewart (@bre_stewart30) <a href="https://twitter.com/bre_stewart30/status/924954161510150145?ref_src=twsrc%5Etfw">October 30, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="und" dir="ltr">I la diputada s&#39;ha quedat sola denunciant l&#39;assetjament sexual a les dones. Quina pena i quina vergonya. <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a><br>© Patrick Hertzog/AFP <a href="https://t.co/AtCxumRu5Z">pic.twitter.com/AtCxumRu5Z</a></p>&mdash; Eva Piquer (@EvaPiquer) <a href="https://twitter.com/EvaPiquer/status/923300624665419776?ref_src=twsrc%5Etfw">October 25, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="fr" dir="ltr">Formidable dessin de PrincessH dans <a href="https://twitter.com/LaCroix?ref_src=twsrc%5Etfw">@LaCroix</a>.<a href="https://twitter.com/hashtag/harcelement?src=hash&amp;ref_src=twsrc%5Etfw">#harcelement</a> <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> <a href="https://t.co/JFm1do6WlO">pic.twitter.com/JFm1do6WlO</a></p>&mdash; Frédéric Pommier (@fred_pom) <a href="https://twitter.com/fred_pom/status/925998002967375872?ref_src=twsrc%5Etfw">November 2, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="ko" dir="ltr">세계 최고 수준 성평등을 이뤘다고 자랑하던 스웨덴에선<br>지금 여성들의 성폭력 고발이 봇물처럼 쏟아지고 있다<br><br>스웨덴 ‘미투’<a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> 운동을 보며<a href="https://t.co/yNx73XQXMz">https://t.co/yNx73XQXMz</a> <a href="https://t.co/ZeLtOLCaIi">pic.twitter.com/ZeLtOLCaIi</a></p>&mdash; 여성신문 (@wnewskr) <a href="https://twitter.com/wnewskr/status/924464132493000704?ref_src=twsrc%5Etfw">October 29, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)




# Map Language to human-readable language name
df['Language'] = df['lang'].map({'en': 'English', 'und': 'Unidentified', 'fr': 'French', 'nl': 'Dutch',
                                'de': 'German', 'sv': 'Swedish', 'ja': 'Japanese', 'es': 'Spanish', 'ko': 'Korean',
                                'it': 'Italian'})


# Plot barplot of number of tweets by language
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Language', fontsize=20);
df.groupby(['Language'])['id'].count().sort_values(ascending=False).head(n=10).plot(figsize=(9,6), kind='bar');

# Plot barplot w/ log y
df.groupby(['Language'])['id'].count().sort_values(ascending=False).head(n=10).plot(figsize=(9,6), 
                                                                                    kind='bar', logy=True);
plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Language', fontsize=20);

# how many tweets were English language?
df['en'] = df['lang'] == 'en'
df.en.sum()/len(df)

# Plot time series of top 5 languages
langs = list(df.groupby(['Language'])['id'].count().sort_values(ascending=False).head(n=5).index)
hour_counts_by_lang = df[df['Language'].isin(langs)].groupby(['dt', 'Language'])['id'].count()

hour_counts_by_lang.unstack(level=1).plot(lw=3, rot=60, figsize=(15,10));

plt.ylabel('Number of tweets', fontsize=20);
plt.xlabel('Date', fontsize=20);

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="und" dir="ltr">I la diputada s&#39;ha quedat sola denunciant l&#39;assetjament sexual a les dones. Quina pena i quina vergonya. <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a><br>© Patrick Hertzog/AFP <a href="https://t.co/AtCxumRu5Z">pic.twitter.com/AtCxumRu5Z</a></p>&mdash; Eva Piquer (@EvaPiquer) <a href="https://twitter.com/EvaPiquer/status/923300624665419776?ref_src=twsrc%5Etfw">October 25, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="fr" dir="ltr">A la manif, on trouve les 4 phrases à dire à une femme victime.<a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> <a href="https://twitter.com/hashtag/Balancetonporc?src=hash&amp;ref_src=twsrc%5Etfw">#Balancetonporc</a> <a href="https://t.co/3lmauq3U4S">pic.twitter.com/3lmauq3U4S</a></p>&mdash; Caroline De Haas (@carolinedehaas) <a href="https://twitter.com/carolinedehaas/status/924654851149189120?ref_src=twsrc%5Etfw">October 29, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="fr" dir="ltr">Morgane était à la manifestation <a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> à Paris. Pour elle, cette mobilisation ne doit être qu’un début. <a href="https://t.co/m0SWleuwJN">pic.twitter.com/m0SWleuwJN</a></p>&mdash; Brut FR (@brutofficiel) <a href="https://twitter.com/brutofficiel/status/924961633654530048?ref_src=twsrc%5Etfw">October 30, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

s = ("""
<blockquote class="twitter-tweet" data-lang="en"><p lang="fr" dir="ltr">Celle ci, je l’aime bcp bcp!<a href="https://twitter.com/hashtag/MeToo?src=hash&amp;ref_src=twsrc%5Etfw">#MeToo</a> <a href="https://t.co/b5bBA4VC8D">pic.twitter.com/b5bBA4VC8D</a></p>&mdash; caroline le diore (@DioreLd) <a href="https://twitter.com/DioreLd/status/924667126744576000?ref_src=twsrc%5Etfw">October 29, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
""")
Tweet(s)

