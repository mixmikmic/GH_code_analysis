import pandas as pd
Medataset=pd.read_csv('Medataset1.csv') 

Medataset.head(5)

Medataset.drop(labels='Unnamed: 0',axis=1,inplace=True)
ColumnsName1=list(Medataset.columns)
PersonalCol1=ColumnsName1[:3]
DiplomCol1=ColumnsName1[3:7]
SchoolCol1=ColumnsName1[7:10]

ColumnsName1[10:]

Medataset

import numpy as np



Medataset['PERC2']=Medataset['PERC2'].fillna(0)

Medataset['PERC2'].head(2)

Element=dict(
     ACADYEAR="%s" % '::'.join([x for x in 'espoirmurha']),
     MENT1="%s" % '::'.join([x for x in 'espoirmurha']),
     MENT2="%s" % '::'.join([x for x in 'espoirmurha']),
     PROM=list([x for x in 'espoirmurha'])
    )

Element

def f(x):
    year=list(x['ACADYEAR'])
    ment=list(x['MENT2'])
    return pd.Series(dict(zip(year,ment)))
Medataset.groupby('ID').apply(f)

def f(x):
    year=list(x['ACADYEAR'])
    ment=list(x['MENT2'])
    mydict = {key:value for key, value in zip(year,ment)}
    return pd.Series(mydict)
Medataset.groupby('ID').apply(f).head(10)

def countMention(alist,mention):
    #this function will return the ratio of the mention 
    PossibleMention=['ADMIS AU MEMOIRE',
 'ADMIS AU STAGE',
 'ADMIS AU STAGE ET AU MEMOIRE',
 'ADMIS AU STAGE ET AU TFC',
 'ADMIS AU TFC',
 'AJOURNE',
 'AR',
 'ASSIMILE AUX AJOURNES',
 'ASSIMILE AUX NON ADMISSIBLES DS LA MEME FILIERE',
 'DISTINCTION',
 'GRANDE DISTINCTION',
 'NON ADMISSIBLE DS LA MEME FILIERE',
 'SATISFACTION'] #or we neeed to get allmention from the dataset
    if isinstance(alist,list) and mention in PossibleMention:
        return float((alist.count(mention))/len(alist))
    else:
        raise TypeError('Invalid type')

def f(x):
    return pd.Series(dict(
     ACADYEAR=list(x['ACADYEAR']),
     MENT1=list(x['MENT1']),
     MENT2=list(x['MENT2']),
     PERC1=list(x['PERC1']),
     PERC2=list(x['PERC2']),
     FAC=reduce(lambda x:x, [x['FAC'].iloc[0]]),
     OPT=reduce(lambda x:x, [x['OPT'].iloc[0]]),
     PROM=list(x['PROM']),
     NumberYear=len(list(x['ACADYEAR'])),
     RatioAJourne=countMention(list(x['MENT1']),'AJOURNE')
    ))
Medataset.groupby('ID').apply(f)

