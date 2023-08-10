
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from youtube_data import youtube_search

test = youtube_search("Imagine Dragons")
test.keys()

test['commentCount'][:5]

df = pd.DataFrame(data=test)
df.head()

df1 = df[['title','viewCount','channelTitle','commentCount','likeCount','dislikeCount','tags','favoriteCount','videoId','channelId','categoryId']]
df1.columns = ['Title','viewCount','channelTitle','commentCount','likeCount','dislikeCount','tags','favoriteCount','videoId','channelId','categoryId']
df1.head()

import numpy as np
numeric_dtype = ['viewCount','commentCount','likeCount','dislikeCount','favoriteCount']
for i in numeric_dtype:
    df1[i] = df[i].astype(int)

ImagineDragons = df1[df1['channelTitle']=='ImagineDragonsVEVO']
ImagineDragons.head()

ImagineDragons = ImagineDragons.sort_values(ascending=False,by='viewCount')
plt.bar(range(ImagineDragons.shape[0]),ImagineDragons['viewCount'])
plt.xticks(range(ImagineDragons.shape[0]),ImagineDragons['Title'],rotation=90)
plt.ylabel('viewCount in 100 millions')

plt.show()

