import numpy as np
import gensim, logging
import pandas as pd
model= gensim.models.Word2Vec.load("w2vmodel")
words=pd.read_csv('data/words.csv',sep="\t")
train=pd.read_csv('data/train-clean.csv',sep="\t")
test=pd.read_csv('data/test.csv',sep="\t")

test['calculation'] = pd.Series(np.NaN , index=test.index)
test['error'] = pd.Series(np.NaN , index=test.index)
test.head()

from IPython.core.debugger import Tracer; debug = Tracer()
from scipy.spatial import distance
  

phrase=[]
tmp1_phrase=[]
tmp2_phrase=[]
attribution_count=0
attribution_sum=0


for indx,row in test.iterrows():
    #check if phrase is in string type
    if isinstance(row.Phrase,str):
        #check if there is something
        if len(row.Phrase):
            #chech if that phrase is in the train set alreadt
            find=train.Phrase[train.Phrase==row.Phrase]
            if len(find.index):
                #if already in train set assign the score directly
                test.loc[indx,"calculation"]=train.loc[find.index[0],"Sentiment"]
            else:
                attribution_count=0
                attribution_sum=0
                phrase=row.Phrase.split()
                length=len(phrase)
                #if phrase contains single word
                if length>1:
                    tmp=1
                    # with tmp+1<length words could be checked seperately
                    while tmp<length:
                        for x in range(0,tmp+1):
                            #devide phrase into 2 part one is the major word group that will be searched as it is
                            #other group words will be scored seperately
                            tmp1_phrase=phrase[x:(length-tmp+x)]
                            tmp2_phrase=list(set(phrase)-set(tmp1_phrase))
                            find=train.Phrase[train.Phrase==" ".join(tmp1_phrase)]
                            a=len(find.index)
                            #check if word group is found if found break
                            if a:
                                break
                        #if found calculate score
                        if a:
                            attribution_count+=1
                            attribution_sum+=train.loc[find.index[0],"Sentiment"]
                            for y in tmp2_phrase:
                                find=train.Phrase[train.Phrase==y]
                                if len(find.index):
                                    attribution_count+=1
                                    attribution_sum+=train.loc[find.index[0],"Sentiment"]
                                else:
                                    try:
                                        vec=model[y]
                                        tmpvec=model[words.loc[0,"Phrase"]]
                                        tmpindx=0
                                        dist=distance.euclidean(vec,tmpvec)
                                        for indx1,row in words.iterrows():
                                            tmpdist=distance.euclidean(vec,model[row.Phrase])
                                            if tmpdist<dist:
                                                dist=tmpdist
                                                tmpvec=model[row.Phrase]
                                                tmpindx=indx1
                                        attribution_count+=1
                                        attribution_sum+=words.loc[tmpindx,"Sentiment"]
                                    except:
                                        pass
                            break
                        tmp+=1
                #if single word phrase calculate score
                else:
                    try:
                        vec=model[row.Phrase]
                        tmpvec=model[words.loc[0,"Phrase"]]
                        tmpindx=0
                        dist=distance.euclidean(vec,tmpvec)
                        for indx1,row in words.iterrows():
                            tmpdist=distance.euclidean(vec,model[row.Phrase])
                            if tmpdist<dist:
                                dist=tmpdist
                                tmpvec=model[row.Phrase]
                                tmpindx=indx1
                        attribution_count+=1
                        attribution_sum+=words.loc[tmpindx,"Sentiment"]
                    except:
                        pass
                if attribution_count!=0:
                    test.loc[indx,"calculation"]=attribution_sum/attribution_count

test.head()

for indx,row in test.iterrows():
    test.loc[indx,"error"]= abs(row.Sentiment-row.calculation)
test.head()

test = test.drop('Unnamed: 0', 1)
test.to_csv("data/test-calculated.csv", sep='\t')

