import pandas as pd

import seaborn as sb
get_ipython().magic('matplotlib inline')

df = pd.read_csv('../datasets/beer.tsv', sep='\t') 
df['WR'] = df.WR.fillna(0)
df['ABV'] = df.ABV.fillna(0)
df['Type'] = df.Type.fillna("na")

df.Brewery.unique()

df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
model = LogisticRegression()

X = df[['ABV', 'WR']]
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train,y_train)
scores = model.score(X_test,y_test)

print(scores)

l = df.iloc[1][['ABV', 'WR']]
print (l)
print ("Predicted Type: ", model.predict(l))

## As we see the real value of this enrty is Imperial IPA. How can we improve the model? 
df.iloc[1]

df[df['Brewery'] == 'Alpine Beer Company']

df.groupby(["Brewery", "Type"])[['Type']].count() 







df['type_Stout'] = df['Type'].map(lambda k: 1 if 'Stout' in k else 0)
df['type_IPA'] = df['Type'].map(lambda k: 1 if 'IPA' in k else 0)
df['type_Ale'] = df['Type'].map(lambda k: 1 if 'Ale' in k else 0)

