import pandas as pd
from IPython.core.debugger import Tracer; debug = Tracer()
import numpy as np
df=pd.read_csv('data/train-clean.csv',sep="\t")
df = df.drop('Unnamed: 0', 1)
#print("total data size: ",len(df))

#code finds the longest phrases of same sentenceid, assign other phrases as NaN then delete related rows
#"same" holds the indexes of phrases with same sentence id
#"ind_ex" holds the longest phrase index up to that point of code
#"lenght" holds the length of longest phrase at index "ind_ex" up to that point of code
#"new_lenght holds the coming phrase length that will be compared to "length" if longer "ind_ex" and "length" will be refreshed
same=[]
ind_ex_=0
length=0
new_length=0

for indx,row in df.iterrows():
    if row.PhraseId!=np.nan :
        same=df.SentenceId[df.SentenceId==row.SentenceId]
        if len(same.index)>1:
            ind_ex=same.index[0]
            length=len(df.loc[ind_ex,'Phrase'])
            for x in range(1,len(same.index)):
                new_length=len(df.loc[same.index[x],'Phrase'])
                if new_length>length:
                    df.loc[ind_ex,"PhraseId"]=np.nan
                    ind_ex=same.index[x]
                    length=new_length
                else:
                    df.loc[same.index[x],"PhraseId"]=np.nan
                    
df=df.dropna()

#print("size of sentence data: ",len(df))

df.head()

df=df.reset_index(drop=True)

df.to_csv("data/train-w2v.csv", sep='\t')

print("done succesfully")

