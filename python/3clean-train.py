import numpy as np
import pandas as pd
df=pd.read_csv('data/train.csv',sep="\t")
df = df.drop('Unnamed: 0', 1)
df.head()

#print("size of data with empty phrase rows=",len(df.index))

#define a map to change empty phrases to NaN
def change_nan(x):
    if isinstance(x,str):
        words=x.split()
        if len(words)>0:
            return x
    return np.nan

df['Phrase'] = df.Phrase.map(change_nan)

#remove NaN rows of df
df=df.dropna()

#reset indexing
df=df.reset_index(drop=True)

#print("size of data after empty phrase rows deleted=",len(df.index))

df.head()

same=[]
for x in df.Phrase:
    same=df.Phrase[df.Phrase==x]
    if len(same.index)>1:
        break
df.loc[same.index[0]]

df.loc[same.index[1]]

#print("data type of sentiment scores= ",df.Sentiment.dtype)

#convert sentiment scores to float to write averaged ones
df.Sentiment=df.Sentiment.astype(float)
df.dtypes
#print("data type of sentiment scores after conversion= ",df.Sentiment.dtype)

#write size of data before merge
#print("data size before merge= ",len(df.index))

#"phrase" holds the current phrase to compare through whole set and find the indexes of same other phrase rows 
#which will written in "same"
#then they will changed to NaN for further clean process
#summation holds the summation of same phrase sentiment scores that will avaraged and written to the first phrase among all same phrases
same=[]
summation=0
phrase=[]

for index,row in df.iterrows():
    if row.Phrase!=np.nan :
        same=df.Phrase[df.Phrase==row.Phrase]
        if len(same.index)>1:
            summation=0
            phrase=df.loc[same.index[0],'Phrase']
            for x in range(1,len(same.index)):
                df.loc[same.index[x],'Phrase']=np.nan
                summation+=df.loc[same.index[x],'Sentiment']
            summation+=df.loc[same.index[0],'Sentiment']
            df.loc[same.index[0],'Sentiment']=summation/len(same.index)

#clean all NaN rows
df=df.dropna()

#write size of data after merge
#print("data size after merge= ",len(df.index))

df=df.dropna()
df=df.reset_index(drop=True)
df.head(15)

df.to_csv("data/train-clean.csv", sep='\t')
df.describe()

print("done succesfully")



