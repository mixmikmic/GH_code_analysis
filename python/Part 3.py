get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from  datetime import timedelta as td
import pandas as pd
import numpy as np
import json
from patsy import dmatrices
import statsmodels.discrete.discrete_model as sm
from sklearn.preprocessing import robust_scale
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

json_request=open('uber_data_challenge.json')
json_data = json.load(json_request)
data=pd.DataFrame(json_data)
json_request.close()

data.head(10)

plt.figure()
histo=data[data.avg_rating_of_driver.isnull()][['avg_rating_by_driver','trips_in_first_30_days']].hist()
plt.suptitle('Users with missing information about their average ratings')

data.loc[data.avg_rating_of_driver.isnull(),"avg_rating_of_driver"]=0.0

data[data.avg_rating_by_driver.isnull()][['avg_rating_of_driver','trips_in_first_30_days']].hist()

data.loc[data.avg_rating_by_driver.isnull(),"avg_rating_by_driver"]=0.0

plt.figure()
plt.subplot(2,2 , 1)
bp1=data.boxplot(column=['avg_dist','trips_in_first_30_days'])
plt.subplot(2, 2, 2)
bp2=data.boxplot(column=['weekday_pct','surge_pct'])
plt.subplot(2, 1, 2)
bp3=data.boxplot(column=['avg_rating_by_driver','avg_rating_of_driver'])

data[['signup_date','last_trip_date']]=data[['signup_date','last_trip_date']].apply(pd.to_datetime)

latest_date=max(data.last_trip_date)
print(latest_date)

data['retained']=(max(data.last_trip_date)-data.last_trip_date)
data['retained']=data['retained']<=td(days=30)
data['inactive']=~data['retained']

data.retained.mean()

city=data.groupby('city')

ax=city.mean().drop(['retained','inactive','uber_black_user','weekday_pct'],axis=1).plot(kind='bar')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center right', bbox_to_anchor=(1.65,0.5))

city.sum()[['inactive','retained']].plot(kind='bar',stacked=True)

phone=data.groupby('phone')

ax2=phone.mean().drop(['retained','inactive','uber_black_user','weekday_pct'],axis=1).plot(kind='bar')
handles2, labels2 = ax2.get_legend_handles_labels()
lgd2 = ax2.legend(handles2, labels2, loc='center right', bbox_to_anchor=(1.65,0.5))

phone.sum()[['inactive','retained']].plot(kind='bar',stacked=True)

data.corr()

y,X=dmatrices('retained ~uber_black_user+trips_in_first_30_days+surge_pct+avg_surge+city+phone+avg_dist+weekday_pct+avg_rating_by_driver+avg_rating_of_driver',data,return_type="dataframe")
y = y['retained[True]']

LR=sm.Logit(y,X)
results=LR.fit()
results.summary()

X_norm = robust_scale(X)
X_train,X_test,y_train,y_test = train_test_split(X_norm,y
                                                 ,test_size = 0.3
                                                 ,random_state=6)

parameters = {
    'random_state' : [0],
    'alpha': [0.001, 0.0001, 0.00001, 0.000001],
    'l1_ratio' : [0,0.1,0.25,0.5,0.75,0.9,1],
}
grid_search = GridSearchCV(SGDClassifier(penalty='elasticnet',loss='log'),
                           parameters,cv=10)
grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_estimator_.get_params()
print(best_parameters)
print(grid_search.best_score_)

elastic = SGDClassifier(**best_parameters)
elastic.fit(X_train,y_train)
y_pred = grid_search.predict(X_test)

pd.DataFrame(elastic.coef_,columns=X.columns)

print(confusion_matrix(y_pred,y_test))
print (classification_report(y_pred,y_test ,digits=3))
print(np.mean(abs(y_pred-y_test)))



