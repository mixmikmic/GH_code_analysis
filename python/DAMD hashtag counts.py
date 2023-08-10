import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

damd = pd.read_csv("", index_col="tweet_id")
damd['hashtags'] = damd['hashtags'].astype(str)
damd.head(3)

hashtags = {}
for (tweet, hashtagsinthistweet) in damd['hashtags'].map(lambda l: l.split(';')).items():
    for hashtag in hashtagsinthistweet:
        hashtag = hashtag.lower()
        if hasthag not in hashtags.keys():
            hashtags[hashtag] = [t]
        else:
            hashtags[hashtag].append(t)

hashtags.pop('damd'); # semicolon at the end of the line suspends output

counts = pd.Series([len(hashtags[h]) for h in hashtags.keys()], name="count", index=hashtags.keys())
counts.describe()

counts.hist()

ax = counts.loc[counts > 5].sort_values().plot.barh(grid=True, figsize=(5, 15), title="Hashtag occurrences, ignored case (count > 5)")
for (patch, hashtag) in zip(ax.patches, counts.loc[counts > 5].sort_values()):
    ax.annotate(hashtag, (patch.get_width() + 5, patch.get_y()))

