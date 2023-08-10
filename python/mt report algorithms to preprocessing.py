import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df = pd.read_pickle('mtpickle.p')

df = df.drop('Step Start Time', 1)
df = df.drop('First Transaction Time', 1)
df = df.drop('Correct Transaction Time', 1)
df = df.drop('Step End Time', 1)
df = df.drop('Step Duration (sec)', 1)
df = df.drop('Correct Step Duration (sec)', 1)
df = df.drop('Error Step Duration (sec)', 1)
df = df.drop('Incorrects', 1)
df = df.drop('Hints', 1)
df = df.drop('Corrects', 1)

df = df.drop('Opportunity(Default)', 1)
df = df.drop('KC(Default)', 1)
df = df.drop('Problem Hierarchy', 1)
df = df.drop('Problem View', 1)
df = df.drop('Step Name', 1)
df = df.drop('Anon Student Id', 1)
df = df.drop('Row', 1)
df = df.drop('Problem Name', 1)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]
df = df.drop('Correct First Attempt',1).join(df['Correct First Attempt']) #make CFA the last column

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))

features = df.columns[:-1]
target = df.columns[-1]

df

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

clf = DummyClassifier()
clf.fit(train[features], train[target])
y_pred = clf.predict(test[features])

accuracy_score(y_pred, test[target])

from sklearn.metrics import f1_score
f1_score(test[target], y_pred, average=None)

pd.set_option('max_colwidth', -1)
df = pd.read_csv('algebra_2005_2006_train.txt', sep='\t')
df.head(5)[['KC(Default)', 'Opportunity(Default)']]

df['KC(Default)'] = df['KC(Default)'].str.split('~~') #split string into list
df['Opportunity(Default)'] = df['Opportunity(Default)'].str.split('~~') #split string into list
df.head(5)[['KC(Default)', 'Opportunity(Default)']]

df = df.drop('KC(Default)', 1).join(df['KC(Default)'].str.join('|').str.get_dummies()) #combined dummies for KCs

df_dict = pd.read_pickle('kclistpickle.p')
df_dict.sort_values(by=['Num_KCs'], ascending=False).head(5)

d = dict(zip(df_dict['KC'],df_dict['Label'])) #make dictionary
df = df.rename(columns = d) #rename columns
df = df.drop('Correct First Attempt',1).join(df['Correct First Attempt']) #make CFA the last column
df.columns = df.columns.str.lower() #make column names lowercase
df.head(5)

head = df[['Row', 'KC(Default)', 'Opportunity(Default)']]
d = []
for index, row in head.iterrows():
    if np.all(pd.isnull(row['KC(Default)'])):
        d.append({})
    else:
        keys = row['KC(Default)']
        values = row['Opportunity(Default)']
        dictionary = dict(zip(keys, values))
        d.append(dictionary)
s = pd.Series(d)

s = s.to_frame()
head = head.join(s)
head.rename(columns = {0:'nested'}, inplace = True)
head = head.drop('KC(Default)', 1)
head = head.drop('Opportunity(Default)', 1)

head.head(5)

def unpack(df, column):
    ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(0)], axis=1)
    ret = ret.drop(column, 1)
    return ret

head = head.applymap(lambda x: {} if pd.isnull(x) else x)
df_unpacked = unpack(head, 'nested')

df_unpacked.tail(5)

df_kc = pd.read_pickle('mtunpacked.p')
df_sum = pd.read_pickle('mtpickle.p')

df_kc['Correct First Attempt'] = df_sum['Correct First Attempt']
df_kc['Sum_OCs'] = (df_kc[df_kc.columns[1:113]].astype(int)).sum(axis=1)
df_kc['Num_KCs'] = (df_sum[df_sum.columns[18:130]] == 1).sum(axis=1)

df_kc.tail(5)



