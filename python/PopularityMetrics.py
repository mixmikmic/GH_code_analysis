import pickle

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_style("whitegrid")

import sqlite3
import pandas as pd

import datetime
cutoff1 = datetime.datetime(2015, 6, 1)

import re

conn = sqlite3.connect('crossvalidated.db')

# return all the records for questions posts from posts table
ques_query = "SELECT * FROM posts WHERE PostTypeId==1"

qpost_df = pd.read_sql_query(ques_query, conn)

qpost_df.shape

# return all the records for answer posts from posts table
ans_query = "SELECT * FROM posts WHERE PostTypeId==2"

apost_df = pd.read_sql_query(ans_query, conn)

post_df = pd.concat([qpost_df,apost_df])

apost_df.head()

qpost_df.CreationDate = pd.to_datetime(qpost_df.CreationDate)
apost_df.CreationDate = pd.to_datetime(apost_df.CreationDate)

def getAnsCount(row):
    records = apost_df[(apost_df['OwnerUserId']== row['OwnerUserId']) & (apost_df['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Accepted by Originator Votes
tt = qpost_df.apply(getAnsCount,axis=1)

sns.distplot(tt, rug=False)

#qpost_df = qpost_df.merge(tt, left_index=True, right_index=True)

# return all the records for questions posts from posts table
posthist_query = "SELECT PostHistoryTypeId,CreationDate,UserId FROM posthistory"
posthist_df = pd.read_sql_query(posthist_query, conn)
posthist_df.CreationDate = pd.to_datetime(posthist_df.CreationDate)

postsedited = posthist_df[posthist_df['PostHistoryTypeId'].isin([4,5,6])]

def getAppEditsCount(row):    
    records = postsedited[(postsedited['UserId']== row['OwnerUserId']) & (postsedited['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Approved Edited Suggestions Count
AESC = qpost_df.apply(getAppEditsCount,axis=1)

sns.distplot(AESC, rug=False)

AESC.value_counts().head()

# return all the records from badges table
badges_query = "SELECT Name,Date,UserId FROM badges"
badges_df = pd.read_sql_query(badges_query, conn)
badges_df.Date = pd.to_datetime(badges_df.Date)

badges_df.shape

#target = ['Great Answer', 'Good Answer', 'Nice Answer','Favorite Question', 'Stellar Question', 'Good Question','Nice Question']
#badges_df = badges_df[badges_df['Name'].isin(target)]  
#relatively small set, count all the badges received by the user

def getBadgeCount(row):    
    records = badges_df[(badges_df['UserId']== row['OwnerUserId']) & (badges_df['Date']< row['CreationDate'])]
    return records.shape[0]

#Total Badge Count
TBC = qpost_df.apply(getBadgeCount,axis=1)

TBC.value_counts().head()

# return all the records from votes table
votes_query = "SELECT VoteTypeId,CreationDate,PostId,UserId FROM votes"
votes_df = pd.read_sql_query(votes_query, conn)
votes_df.CreationDate = pd.to_datetime(votes_df.CreationDate)

votes_df = votes_df[votes_df['VoteTypeId'].isin([6,10])]

def getTCDVCount(row):    
    postid_target = post_df[(post_df['OwnerUserId']== row['OwnerUserId'])&(post_df['CreationDate']< row['CreationDate'])].Id.tolist()
    records = votes_df[(votes_df['PostId'].isin(postid_target)) & (votes_df['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Total Close and Deletion Votes¶
TCDV = qpost_df.apply(getTCDVCount,axis=1)

TCDV.value_counts()

# return all the records from votes table
votes_query = "SELECT VoteTypeId,CreationDate,PostId,UserId FROM votes"
votes_df = pd.read_sql_query(votes_query, conn)
votes_df.CreationDate = pd.to_datetime(votes_df.CreationDate)

votes_df = votes_df[votes_df['VoteTypeId']==2]

def getUpVoteCount(row):    
    #get the past post ids from the user
    postid_target = post_df[(post_df['OwnerUserId']== row['OwnerUserId'])&(post_df['CreationDate']< row['CreationDate'])].Id.tolist()
    records = votes_df[(votes_df['PostId'].isin(postid_target)) & (votes_df['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Total Up Votes by the time of new post
TUVC = qpost_df.apply(getTCDVCount,axis=1)

TUVC.value_counts().head()

# return all the records from votes table
votes_query = "SELECT VoteTypeId,CreationDate,PostId,UserId FROM votes"
votes_df = pd.read_sql_query(votes_query, conn)
votes_df.CreationDate = pd.to_datetime(votes_df.CreationDate)
votes_df = votes_df[votes_df['VoteTypeId']==3]

def getDownVoteCount(row):    
    #get the past post ids from the user
    postid_target = post_df[(post_df['OwnerUserId']== row['OwnerUserId'])&(post_df['CreationDate']< row['CreationDate'])].Id.tolist()
    records = votes_df[(votes_df['PostId'].isin(postid_target)) & (votes_df['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Total Up Votes by the time of new post
TDVC = qpost_df.apply(getDownVoteCount,axis=1)

TDVC.value_counts().head()

votes_query = "SELECT VoteTypeId,CreationDate,PostId,UserId FROM votes WHERE VoteTypeId==5"
votes_df = pd.read_sql_query(votes_query, conn)
votes_df.CreationDate = pd.to_datetime(votes_df.CreationDate)

def getFavVoteCount(row):    
    #get the past post ids from the user
    postid_target = post_df[(post_df['OwnerUserId']== row['OwnerUserId'])&(post_df['CreationDate']< row['CreationDate'])].Id.tolist()
    records = votes_df[(votes_df['PostId'].isin(postid_target)) & (votes_df['CreationDate']< row['CreationDate'])]
    return records.shape[0]

#Total Up Votes by the time of new post
TFVC = qpost_df.apply(getFavVoteCount,axis=1)

TFVC.value_counts().head()

qpost_df['AccpAnsCnt'] = tt
qpost_df['AppEditCnt'] = AESC
qpost_df['BadgeCnt'] = TBC
qpost_df['CloDelVCnt'] = TCDV
qpost_df['UpVoteCnt'] = TUVC
qpost_df['DownVoteCnt'] = TDVC
qpost_df['FavVoteCnt'] = TFVC

target_cols = ['Id','CreationDate','OwnerUserId','AccpAnsCnt','AppEditCnt','BadgeCnt','CloDelVCnt','UpVoteCnt',               'DownVoteCnt','FavVoteCnt','Score']

popularity_metric_df = qpost_df[target_cols]

popularity_metric_df.head(10)

import pickle
with open('popularity_metric.pickle', 'wb') as handle:
    pickle.dump(popularity_metric_df, handle)

