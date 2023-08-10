import pandas as pd
from pandas import Series,DataFrame

titanic_df=pd.read_csv('train.csv')

titanic_df.head()

titanic_df.info()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.factorplot('Sex', data=titanic_df, kind='count')

sns.factorplot('Sex',data=titanic_df, hue='Pclass', kind='count')

sns.factorplot('Pclass',data=titanic_df, hue='Sex', kind='count')

def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['Person']=titanic_df[['Age','Sex']].apply(male_female_child, axis=1)

titanic_df.head(10)

sns.factorplot('Pclass',data=titanic_df, hue='Person', kind='count')

titanic_df['Age'].hist(bins=80)

titanic_df['Age'].mean()

titanic_df['Person'].value_counts()

titanic_df['Age'].value_counts()

fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='Person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()

titanic_df.head()

deck = titanic_df['Cabin'].dropna()

deck.head()

levels = []
for level in deck:
    levels.append(level[0])
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data=cabin_df,palette='winter_d',kind='count', order=['A','B','C','D','E','F','T'])

cabin_df = cabin_df[cabin_df != 'T']
sns.factorplot('Cabin', data=cabin_df,palette='summer',kind='count', order=['A','B','C','D','E','F','T'])

sns.factorplot('Embarked', data=titanic_df, hue='Pclass',order=['C','Q','S'], kind='count')

# Who was alone and who was with family?
titanic_df.head()

titanic_df['Alone'] = titanic_df.SibSp+titanic_df.Parch
titanic_df['Alone']

titanic_df['Alone'].loc[titanic_df['Alone']>0]='With Family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'

titanic_df.head()

sns.factorplot('Alone',data=titanic_df,palette='Blues', kind='count')

titanic_df['Survivor']=titanic_df.Survived.map({0:'no', 1:'yes'})
sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')

#men had low survival rate, probably due to the idea of 'women and children first'
sns.factorplot('Pclass', 'Survived', hue='Person',data=titanic_df)

#how does age affect survival rate?
sns.lmplot('Age','Survived', data=titanic_df)
#less chance to survive if older

sns.lmplot('Age','Survived', hue='Pclass',data=titanic_df, palette='winter')

generations=[10,20,40,60,80]
sns.lmplot('Age','Survived', hue='Pclass',data=titanic_df, palette='winter', x_bins=generations)

#better chance to survive if you are an older female
sns.lmplot('Age','Survived', hue='Sex',data=titanic_df, palette='winter', x_bins=generations)

#did the deck have an impact on chance of survival?
titanic_df['Cabin'].head()

levels=[]
for level in titanic_df['Cabin']:
    if not pd.isnull(level):
        levels.append(level[0])
    else:
        levels.append('None')
titanic_df['Deck']=levels
titanic_df.head()

sns.factorplot('Deck', 'Survived',data=titanic_df, order=['A','B','C','D','E','F','G','T','None'])

#much less chance of survival if not in a cabin
sns.factorplot('Deck', hue='Survivor',data=titanic_df, kind='count',order=['A','B','C','D','E','F','G','T','None'])

#good chance of survival in cabins B, D, and E
sns.factorplot('Deck', hue='Survivor',data=titanic_df, kind='count',order=['A','B','C','D','E','F','G'])

#did passengers have a better chance at survival if they were alone?
sns.factorplot('Alone',data=titanic_df,palette='Blues', kind='count')

#people who were not with family actually had a less chance of survival
sns.factorplot('Alone', hue='Survivor',data=titanic_df,palette='Blues', kind='count')

isAlone=titanic_df[titanic_df['Alone']=='Alone']
isAlone['Survivor'].value_counts()

#chance of survival if alone
163./(163.+374.)

withFam=titanic_df[titanic_df['Alone']=='With Family']
withFam['Survivor'].value_counts()

#chance of survival with family
179./(175+179)



