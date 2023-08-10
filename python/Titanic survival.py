import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

tt = pd.read_csv('titanic_train.csv')

tt.head()

sns.heatmap(tt.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# yellow means null values like in age and cabin
# maybe drop cabine due to many missing data
# yticklabels=False gives the graph
# cbar=False gives the side bar on the very right
# cmap='viridis' - gives color

tt.isnull().sum()

# fill in the null values in age and drop cabin

tt.groupby('Pclass')['Age'].mean()

def dataClean(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age

tt['Age'] = tt[['Age','Pclass']].apply(dataClean,axis=1)

tt.drop('Cabin',axis=1,inplace=True)

tt.dropna(inplace=True)

sns.heatmap(tt.isnull(),yticklabels=False,cbar=False,cmap='viridis')

tt.isnull().sum() # no more null values

# play with the data 

# explore the factors that correclate with survial
tt.info()

sns.countplot(x='Survived',data=tt,)
# palette='RdBu_r'=color

sns.set_style('whitegrid')
g=sns.FacetGrid(tt,col='Sex',hue='Sex')
g.map(sns.countplot,'Survived')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=tt,palette='rainbow')


sns.stripplot(x='Age',y='Survived',data=tt, jitter=True)
# poor older people

sns.stripplot(x='Survived',y='Fare',data=tt,palette='rainbow')
# damn those 1%


sns.stripplot(x='Survived',y='Cabin',data=tt,palette='rainbow')

sns.countplot(x='Survived',hue='SibSp',data=tt,palette='rainbow')

sns.countplot(x='Survived',hue='Parch',data=tt,palette='rainbow')

sns.distplot(tt['Age'].dropna(),kde=False,color='darkred',bins=30)
#,kde=False - to remove a line to see the bell

sns.distplot(tt['SibSp'].dropna(),kde=False,color='darkred',bins=30)

sns.countplot(x='SibSp',data=tt)
# most people dont have any sisblings and children on board

g=sns.FacetGrid(tt,col='Pclass')
g.map(sns.countplot,'SibSp')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='SibSp',data=tt,palette='RdBu_r')
# single person died the most and survive also

sns.set_style('whitegrid')
sns.countplot(x='SibSp',hue='Sex',data=tt,palette='RdBu_r')

tt['Fare'].hist(color='green',bins=40,figsize=(10,4))

plt.figure(figsize=(10, 7))
sns.boxplot(x='Sex',y='Age',data=tt,palette='winter')
# shows the age range of each class

tt.head()

# dont need passenger id, name, ticket
tt.drop({'PassengerId','Name','Ticket'},axis=1, inplace=True)

tt.head()

# for categrical features(Sex, Embarked), need to change into something that is accepted by sklearn

# 2 options, use dummies variables or labelencoder and onehot encoder

pd.get_dummies(tt['Sex']).head()
# but this is an issue of mulcolumity
# to avoid this, drop a column to keep female or male

pd.get_dummies(tt['Embarked']).head()

# macheine learning algorithm is not able to take in male and female strings,
# need 0 or 1 for each
# embarked also
sex = pd.get_dummies(tt['Sex'],drop_first=True)
embark = pd.get_dummies(tt['Embarked'],drop_first=True)

tt.drop(['Sex','Embarked'],axis=1,inplace=True)

tt = pd.concat([tt,sex,embark],axis=1)
# add those new data columns to the database

tt.head()

X=tt.drop('Survived',axis=1)
y=tt['Survived']

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaled_X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression

lor = LogisticRegression()
lor.fit(X_train,y_train)

pred_lor = lor.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print (confusion_matrix(y_test,pred_lor))
print(classification_report(y_test,pred_lor))

tt.head()

X.describe()

# infomation of person A : 2.0,32,2,2,299,1,1,1
new_pred = lor.predict(scaler.transform(np.array([[2.0,32,2,2,299,1,1,1]])))

print('Will person A survive?')
print('...')

if new_pred == 1:
    print('Yes')
else:
    print('Totally dead')

# what a pity. He or she might have change the world. Lmao

