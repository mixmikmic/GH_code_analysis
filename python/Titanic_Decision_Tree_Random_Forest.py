import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

titanic_df = pd.read_csv('train.csv')
titanic_df.head()

titanic_df.info()

# Q1 Who were the passengers on Titanic??
titanic_df['Sex'].value_counts()

titanic_df.groupby('Sex')['Sex'].count()

sns.factorplot('Sex',data=titanic_df,kind="count")
# shows that there are more male passengers as compared to female passangers

titanic_df.groupby('Pclass').count()

# Distribution of males and females based on pclass and place of embarkment
titanic_df.groupby(['Pclass','Sex','Embarked']).size()

print titanic_df.groupby(['Pclass','Sex']).size()
print 
sex_age_df = pd.crosstab(index=[titanic_df['Pclass']], columns=[titanic_df['Sex']])
sex_age_df

sns.factorplot('Pclass',data=titanic_df,kind='count',hue='Sex')

def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis = 1)

titanic_df.head(10)

titanic_df.groupby(['Pclass','Person']).size()

sns.factorplot('Pclass',data=titanic_df,kind='count',hue='Person')

# Ages of people 
titanic_df['Age'].hist(bins = 70)
# Mean is around 30

titanic_df['Age'].mean()

titanic_df['Person'].value_counts()

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

# Distribution of female ages
titanic_df[titanic_df['Sex'] == 'female']['Age'].hist(bins = 70)

# Distribution of male ages
titanic_df[titanic_df['Sex'] == 'male']['Age'].hist(bins = 70)

#How age is distributed w.r.t. to class
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()

# Q2 What deck the passengers on and how does it relate to their class?
titanic_df.head()

# Percentage of Null Values in all columns =
p_null= (len(titanic_df) - titanic_df.count())*100.0/len(titanic_df)
p_null

# Percentage of Null Values in cabin column =
p_null_cabin= (len(titanic_df['Cabin']) - titanic_df['Cabin'].count())*100.0/len(titanic_df['Cabin'])
p_null_cabin

sns.factorplot('Pclass', data=titanic_df, kind='count', hue='Person', order=[1,2,3], 
               hue_order=['child','female','male'], aspect=2)

# Do the same as above, but split the passengers into either survived or not
sns.factorplot('Pclass', data=titanic_df, kind='count', hue='Person', col='Survived', order=[1,2,3], 
               hue_order=['child','female','male'], aspect=1.25, size=5)

sns.factorplot('Embarked', data=titanic_df, kind='count', hue='Pclass')

titanic_df.groupby(['Embarked','Pclass']).size()

titanic_df.Embarked.value_counts()

embarked_vs_pclass = pd.crosstab(index = [titanic_df['Embarked']], columns=[titanic_df['Pclass']],margins=True)
embarked_vs_pclass

def alone_with_family(passenger):
    parch,sibsp = passenger
    if (parch == 0) & (sibsp == 0):
        return 'alone'
    else:
        return 'with_family'

titanic_df['alone_or_with_family'] = titanic_df[['Parch','SibSp']].apply(alone_with_family,axis = 1)
titanic_df.head()

titanic_df['alone_or_with_family'].value_counts()

fg=sns.factorplot('alone_or_with_family', data=titanic_df, kind='count', hue='Pclass', col='Person')

pd.crosstab(index = [titanic_df['alone_or_with_family'],titanic_df['Person']], columns=[titanic_df['Pclass']],margins=True)

def titanic_preprocessing(train, test):
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    combine = [train_df, test_df]

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_df['Title'], train_df['Sex'])

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    guess_ages = np.zeros((2,3))
    guess_ages

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) &                                       (dataset['Pclass'] == j+1)]['Age'].dropna()

                age_guess = guess_df.median()

                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    combine = [train_df, test_df]

    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    
    train_df = pd.get_dummies(train_df)
    test_df  = pd.get_dummies(test_df)

    test_df.head(10)
    return (train_df, test_df)

train = 'train.csv'
test = 'test.csv'
titanic_train,titanic_test = titanic_preprocessing(train, test)

print titanic_test.shape, titanic_train.shape

titanic_train.head()

titanic_test.head()

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)

Y_train = titanic_train['Survived']

Y_train.head()

X_train = titanic_train.iloc[:,1:]

X_train.columns

X_train.shape

clf = clf.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
Y_pred = clf.predict(titanic_test.iloc[:,1:])
df1 = pd.DataFrame(np.array([titanic_test.iloc[:,0],Y_pred]).T,columns=['PassengerId','Survived'])

df1.head()

df1.to_csv('prediction.csv',index=False)

df1.columns

def plot_decision_tree(clf,feature_name,target_name):
    from IPython.display import Image 
    from StringIO import StringIO
    import pydotplus
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_name,  
                         class_names=target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())

plot_decision_tree(clf,X_train.columns,df1.columns[1])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

Y = titanic_train['Survived']
X = titanic_train.iloc[:,1:]

def grid_search_output(parameters,X,Y,clf):
    num_test = 0.20
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = num_test,random_state = 23)
    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)    
    return grid_obj.cv_results_

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 6)
def plot_my_data(X,Ytest,Ytrain):
    if type(X.values()[0][0]) is str:        
        x = range(len(X.values()[0]))
        my_xticks = X.values()[0]
        plt.xticks(x, my_xticks)
        plt.plot(x,Ytest,'r-',x,Ytrain)
    else:
        plt.plot(X.values()[0],Ytest,'r-',X.values()[0],Ytrain)
    plt.xlabel(X.keys()[0])
    plt.ylabel('Mean Scores')
    plt.show()
    return plt

parameters1 = {'n_estimators': [2,4,6,8,16,32]}
parameters2 = {'max_features': ['log2', 'sqrt','auto']}
parameters3 = {'criterion': ['entropy', 'gini']}
parameters4 = {'max_depth': [2, 3, 5, 10]}
parameters5 = {'min_samples_split': [2, 3, 5]}
parameters6 = {'min_samples_leaf': [1,5,8]}

clf = RandomForestClassifier(oob_score=True)
t1 = grid_search_output(parameters1,X,Y,clf)
plot_my_data(parameters1,t1['mean_test_score'],t1['mean_train_score'])

t2 = grid_search_output(parameters2,X,Y,clf)
plot_my_data(parameters2,t2['mean_test_score'],t2['mean_train_score'])

t3 = grid_search_output(parameters3,X,Y,clf)
plot_my_data(parameters3,t3['mean_test_score'],t3['mean_train_score'])

t4 = grid_search_output(parameters4,X,Y,clf)
plot_my_data(parameters4,t4['mean_test_score'],t4['mean_train_score'])

t5 = grid_search_output(parameters5,X,Y,clf)
plot_my_data(parameters5,t5['mean_test_score'],t5['mean_train_score'])

t6 = grid_search_output(parameters6,X,Y,clf)
plot_my_data(parameters6,t6['mean_test_score'],t6['mean_train_score'])

clf1 = RandomForestClassifier(n_estimators=15,criterion='gini',min_samples_leaf=5, max_depth=5,max_features='sqrt')

clf1 = clf1.fit(X_train,Y_train)
Y_pred_random_forest = clf1.predict(titanic_test.iloc[:,1:])
df_random = DataFrame(np.array([titanic_test.iloc[:,0],Y_pred_random_forest]).T,columns=['PassengerId','Survived'])
df_random.head()

df_random.to_csv('prediction_random_forest.csv',index=False)

