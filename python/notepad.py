import pandas as pd, numpy as np

ls

test_set = pd.read_csv('test.csv')

train_set = pd.read_csv('train.csv')

train_set.head()

test_set.head()

full = train_set.append(test_set, ignore_index=True)

full.head(10)

full.describe()

full.info()

#Age, Cabin, Embarked, Fare has lost value, and Survived is the result that we have to predict


full['Age'] = full['Age'].fillna(full['Age'].mean())
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full['Embarked'] = full['Embarked'].fillna('S')
full['Cabin'] = full['Cabin'].fillna('U')


full.info()


sex_mapDict = {'male':1, 'female':0}
sex = full['Sex'].map(sex_mapDict)
sex.head()

ls

#embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(full['Embarked'], prefix = 'Embarked')
embarkedDf.head()

pclassDf = pd.get_dummies(full['Pclass'], prefix = 'Pclass')
pclassDf.head()

full['Name'].head()

def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3

titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
titleDf.head()

set(titleDf['Title'].values) # to check the kinds of titles in the dataset

title_mapDict = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Don': 'Royalty',
    'Dona': 'Royalty',
    'Dr': 'Officer',
    'Jonkheer':'Officer', #Jonkheer (female equivalent: Jonkvrouw) is a Dutch honorific of nobility
    'Lady': 'Royalty',
    'Major': 'Officer',
    'Master': 'Master',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Ms': 'Mrs',
    'Rev': 'Officer',
    'Sir': 'Royalty',
    'the Countess': 'Royalty'
}
# to reduce the dimension from 10+ to 6

titleDf['Title'] = titleDf['Title'].map(title_mapDict)
titleDf = pd.get_dummies(titleDf['Title'])
titleDf.head()

full = pd.concat([full, titleDf], axis = 1)
full.drop('Name', axis = 1, inplace=True)
full.head()

full['Cabin'] = full['Cabin'].map(lambda x: x[0])
cabinDf = pd.get_dummies(full['Cabin'], prefix = 'Cabin')
cabinDf.head()

familyDf = pd.DataFrame()
familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s== 1 else 0 )

familyDf[ 'Family_Small' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

familyDf[ 'Family_Large' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

familyDf.head()

full = pd.concat([full,familyDf],axis=1)

full.head()

corrDf = full.corr()
corrDf

# to select 10 features with large correlation with the target(here is 'survived')
label_list = (abs(corrDf['Survived']).sort_values(ascending =False)).head(11)
type(label_list.index.values)
label_list

full_X = full[label_list.index.values]

full_X.head()

full_X.columns

source_X = full_X.dropna().drop(columns = 'Survived')
source_Y = full_X.dropna()['Survived']
source_Y.head()

from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(source_X, source_Y, train_size= .8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
model.score(test_X, test_y)

pred_X = full_X.loc[(full_X['Survived'] != 0) & (full_X['Survived'] != 1)].drop(columns='Survived')
print(pred_X.head())
pred_Y = model.predict(pred_X)
pred_Y.astype(int)

passenger_id = full['PassengerId'][891:]
predDf = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': pred_Y.astype(int)
})

predDf.head()
predDf.to_csv('titanic_survival.csv', index = False)

ls



