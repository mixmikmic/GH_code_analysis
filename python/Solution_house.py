import pandas as pd
import numpy as np
import datetime

train=pd.read_csv('final_train_data.csv')

train.describe()

get_ipython().run_line_magic('matplotlib', 'inline')

#train.hist(figsize=(15,15))

# converting object type columns into numerical values
#train['space'] = train['space'].convert_objects(convert_numeric=True)
#train['bedrooms'] = train['bedrooms'].convert_objects(convert_numeric=True)
#train['bathrooms'] = train['bathrooms'].convert_objects(convert_numeric=True)
#train['bathrooms'] = train['bathrooms'].convert_objects(convert_numeric=True)
#train['Knight'] = train['Knight'].convert_objects(convert_numeric=True)
#train['Dock'] = train['Dock'].convert_objects(convert_numeric=True)
#train['Capital'] = train['Capital'].convert_objects(convert_numeric=True)
#train['Royal Market'] = train['Royal Market'].convert_objects(convert_numeric=True)
#train['Guarding_tower'] = train['Guarding_tower'].convert_objects(convert_numeric=True)
#train['River'] = train['River'].convert_objects(convert_numeric=True)

train['bathrooms']= train['bathrooms'].fillna(np.mean(train['bathrooms']))
train['bedrooms']= train['bedrooms'].fillna(np.mean(train['bedrooms']))
train['Capital']= train['Capital'].fillna(np.mean(train['Capital']))
train['farm']= train['farm'].fillna(0)
train['Royal Market']= train['Royal Market'].fillna(np.mean(train['Royal Market']))
train['Knight']= train['Knight'].fillna(np.mean(train['Knight']))
train['Guarding_tower']= train['Guarding_tower'].fillna(np.mean(train['Guarding_tower']))
train['Dock']= train['Dock'].fillna(np.mean(train['Dock']))
train['River']= train['River'].fillna(np.mean(train['River']))
train['renovation']= train['renovation'].fillna(0)
train['visit']= train['visit'].fillna(0.0)
train['curse'] =train['curse'].fillna(0)
train['holy_tree'] =train['holy_tree'].fillna(0)
train['space'] =train['space'].fillna(0)
train['bless'] =train['bless'].fillna(np.mean(train['bless']))
train['space'] =train['space'].fillna(0)
train['dining']=train['dining'].fillna(0)



train.corr()

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sns.set(style="white", color_codes=True)

plt.figure(figsize=(15,15)) 
sns.heatmap(train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()

train.columns[train.isnull().any()]

train.head(5)

train['Location'] = train['Location'].fillna("cms")

location_map= {" King's Landing":0," The Mountains":1," Servant's Premises":2," Cursed Land":3,"cms" :4 }

train['Location'] = train['Location'].apply(lambda x: location_map[x])

train['Location'].dtype

uniq = train.Location.unique()
uniq

miss= train.isnull().sum()/len(train)
miss= miss[miss>0]
miss
# no missing values 

# converting date-time string into Datetime object to extract year ,month .....
train['month']=(train['priced'].to_string().lstrip()).split('/')[0];
train['day']=(train['priced'].to_string()).split('/')[1];
train['year']=((train['priced'].to_string()).split('/')[2]).split(' ')[0];

train['month_b']=(train['built'].to_string().lstrip()).split('/')[0];
train['day_b']=(train['built'].to_string()).split('/')[1];
train['year_b']=((train['built'].to_string()).split('/')[2]).split(' ')[0];

train['month'] = (train['month'].to_string().lstrip()).split(' ')[1]
train['month_b'] = (train['month_b'].to_string().lstrip()).split(' ')[1]

train.head(2)

sns.distplot(train['Capital'])
train['Capital'].skew()

sns.distplot(train['Dock'])
train['Dock'].skew()

sns.distplot(train['River'])
train['River'].skew()

sns.distplot(train['Capital'])
train['Capital'].skew()

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sns.set(style="white", color_codes=True)

from sklearn.linear_model import LinearRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn import metrics                          #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier          #for using Decision Tree Algoithm
from sklearn.ensemble import RandomForestClassifier      # A combine model of many decision trees
from sklearn.cross_validation import cross_val_score , KFold
#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

train.columns

#train.info()

train.head(1)

train['year']= train['year'].astype(str).astype(int)
#train['month']= train['month'].astype(str).astype(int)
train['day']= train['day'].astype(str).astype(int)
train['year_b']= train['year_b'].astype(str).astype(int)
#train['month_b']= train['month_b'].astype(str).astype(int)
train['day_b']= train['day_b'].astype(str).astype(int)

# age of house = year_priced -year_built

train['old'] = train['year']-train['year_b']
train['old'].dtype
train['old']= train['old'].fillna(5)

train['price']

# correlation between old and price
c = train[['old','price']]
c.corr()



#train['bless'] = np.log(train['bless'])

train['River'].skew()

# inverse transform
#train['Capital'] = 1/train['Capital']

'''train['Dock'] = np.log(train['Dock'])
train['River'] =np.log(train['River'])
train['Royal Market'] = np.log(train['Royal Market'])
train['Guarding_tower'] = np.log(train['Guarding_tower'])'''

#train['bless'] = np.log(train['bless'])

X = train[['dining','River','bathrooms','old','farm','bedrooms','bless','Dock','curse','Location','Capital','Royal Market','Guarding_tower','holy_tree','renovation','year','day','day_b','year_b']]
Y = train[['price']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.32,random_state=225)

model_1 = RandomForestRegressor(n_estimators=300, max_depth=13, min_samples_split=11, min_samples_leaf=7, min_weight_fraction_leaf=0.0, max_features=11, max_leaf_nodes=None, min_impurity_split=0.0,random_state=7)
model_1.fit(X_train,Y_train)
predict_1 =model_1.predict(X_test)
print " R^2 Accuracy of RFR is {0}".format(metrics.r2_score(Y_test,predict_1))

model_2 = LinearRegression()
model_2.fit(X_train,Y_train)
predict_2 =model_2.predict(X_test)
print " R^2 Accuracy of LR is {0}".format(metrics.r2_score(Y_test,predict_2))

test = pd.read_csv('final_test_data.csv')

test['bathrooms']= test['bathrooms'].fillna(np.mean(test['bathrooms']))
test['bedrooms']= test['bedrooms'].fillna(np.mean(test['bedrooms']))
test['Capital']= test['Capital'].fillna(np.mean(test['Capital']))
test['farm']= test['farm'].fillna(0)
test['Royal Market']= test['Royal Market'].fillna(np.mean(test['Royal Market']))
test['Knight']= test['Knight'].fillna(np.mean(test['Knight']))
test['Guarding_tower']= test['Guarding_tower'].fillna(np.mean(test['Guarding_tower']))
test['Dock']= test['Dock'].fillna(np.mean(test['Dock']))
test['River']= test['River'].fillna(np.mean(test['River']))
test['renovation']= test['renovation'].fillna(0)
test['visit']= test['visit'].fillna(0.0)
test['curse'] =test['curse'].fillna(0)
test['holy_tree'] =test['holy_tree'].fillna(0)
test['space'] =test['space'].fillna(0)
test['bless'] =test['bless'].fillna(np.mean(test['bless']))
test['space'] =test['space'].fillna(0)
test['dining']=test['dining'].fillna(0)

test['Location'] = test['Location'].fillna("cms")

test['Location'] = test['Location'].apply(lambda x: location_map[x])

# converting date-time string into Datetime object to extract year ,month .....
test['month']=(test['priced'].to_string().lstrip()).split('/')[0];
test['day']=(test['priced'].to_string()).split('/')[1];
test['year']=((test['priced'].to_string()).split('/')[2]).split(' ')[0];

test['month_b']=(test['built'].to_string().lstrip()).split('/')[0];
test['day_b']=(test['built'].to_string()).split('/')[1];
test['year_b']=((test['built'].to_string()).split('/')[2]).split(' ')[0];

test['month'] = (test['month'].to_string().lstrip()).split(' ')[1]
test['month_b'] = (test['month_b'].to_string().lstrip()).split(' ')[1]

test['year']= test['year'].astype(str).astype(int)
#train['month']= train['month'].astype(str).astype(int)
test['day']= test['day'].astype(str).astype(int)
test['year_b']= test['year_b'].astype(str).astype(int)
#train['month_b']= train['month_b'].astype(str).astype(int)
test['day_b']= test['day_b'].astype(str).astype(int)

test['old'] = test['year']-test['year_b']

# feature transformation
#test['bless'] = np.log(test['bless'])
#test['Capital'] = 1/ train['Capital']

# log transforming the columns for improving symmetry of distribution.
'''test['Dock'] = np.log(test['Dock'])
test['River'] =np.log(test['River'])
test['Royal Market'] = np.log(test['Royal Market'])
test['Guarding_tower'] = np.log(test['Guarding_tower'])'''

#selecting the same features as used in training model.
W = test[['dining','River','bathrooms','old','farm','bedrooms','bless','Dock','curse','Location','Capital','Royal Market','Guarding_tower','holy_tree','renovation','year','day','day_b','year_b']]

# predicting on test dataset
predict_1t= model_1.predict(W) 

predict_2t = model_2.predict(W) # LinearRegression >>>final submission

idx =train['House ID']

# generating csv file for submission.
columns =['Golden Grains']
submit = pd.DataFrame(data= predict_1t,columns=columns)
submit['House ID']=idx
submit = submit[['House ID','Golden Grains']]
submit.to_csv("RFRSOLUTION.csv",index=False) #  RFR() solution

# generating csv file for submission.
columns =['Golden Grains']
submit = pd.DataFrame(data= predict_2t,columns=columns)
submit['House ID']=idx
submit = submit[['House ID','Golden Grains']]
submit.to_csv("lRSOLUTION.csv",index=False) #  lR() solution>>> final submission

