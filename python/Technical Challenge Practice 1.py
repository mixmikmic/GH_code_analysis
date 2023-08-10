import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import datetime as dt
get_ipython().magic('matplotlib inline')

#import data
logins = pd.read_json('logins.json')
logins.head()

print(logins.info())
print(logins.describe())

login_15 = logins.resample('15T', on='login_time').count()
login_15.columns = ['login_count']
login_15.head()

#login_15.reset_index(inplace=True)
#login_15['login_date'] = login_15.login_time.dt.date
#login_15[login_15.login_count == 0].login_count.groupby(login_15.login_date).count().sort_values(ascending=False).head()

daily_logins = login_15.login_count.groupby(login_15.login_date).sum()
fig, ax = plt.subplots(figsize=(15,5))
plt.bar(daily_logins.index, height=daily_logins.values)
plt.title('Daily Logins')
plt.ylabel('# Logins')
plt.xlabel('Date')

fig, ax = plt.subplots(figsize=(15,5))
plt.plot(login_15.index, login_15.login_count)
plt.xlabel('login time (15 min)')
plt.ylabel('login count')
print(login_15.sort_values('login_count', ascending=False).head(10))



login_15.reset_index(inplace=True)
login_15['login_just_time'] = login_15.login_time.dt.time
login_15['login_day'] = login_15.login_time.dt.day
login_15['login_month'] = login_15.login_time.dt.month
login_15.head()

login_april = login_15[login_15.login_month == 4]
login_april.reset_index(inplace=True, drop=True)

fig1, ax1 = plt.subplots(figsize=(15,5))
for i in login_april.login_day.unique():
    plt.plot(login_april[login_april.login_day==i].login_just_time, login_april[login_april.login_day==i].login_count)

plt.xlabel('login time (15 min)')
plt.ylabel('login count')
plt.xlim(0)

login_jan = login_15[login_15.login_month == 1]
login_jan.reset_index(inplace=True, drop=True)
login_feb = login_15[login_15.login_month == 2]
login_feb.reset_index(inplace=True, drop=True)
login_mar = login_15[login_15.login_month == 3]
login_mar.reset_index(inplace=True, drop=True)

fig1, ax1 = plt.subplots(figsize=(15,5))
for i in login_jan.login_day.unique():
    plt.plot(login_jan[login_jan.login_day==i].login_just_time, login_jan[login_jan.login_day==i].login_count, alpha=.5)
for i in login_april.login_day.unique():
    plt.plot(login_april[login_april.login_day==i].login_just_time, login_april[login_april.login_day==i].login_count, alpha=.5)
for i in login_feb.login_day.unique():
    plt.plot(login_feb[login_feb.login_day==i].login_just_time, login_feb[login_feb.login_day==i].login_count, alpha=.5)
for i in login_mar.login_day.unique():
    plt.plot(login_mar[login_mar.login_day==i].login_just_time, login_mar[login_mar.login_day==i].login_count, alpha=.5)

plt.title('Login Counts Jan 1st - Apr 13th')
plt.xlabel('login time (15 min)')
plt.ylabel('login count')
plt.xlim(0)

fig1, ax1 = plt.subplots(figsize=(15,5))
for i in login_jan.login_day.unique():
    plt.plot(login_jan[login_jan.login_day==i].login_just_time, login_jan[login_jan.login_day==i].login_count, alpha=.5)
plt.title('Jan')
plt.ylabel('login count')
fig2, ax2 = plt.subplots(figsize=(15,5))
for i in login_feb.login_day.unique():
    plt.plot(login_feb[login_feb.login_day==i].login_just_time, login_feb[login_feb.login_day==i].login_count, alpha=.5)
plt.title('Feb')
plt.ylabel('login count')
fig3, ax3 = plt.subplots(figsize=(15,5))
for i in login_mar.login_day.unique():
    plt.plot(login_mar[login_mar.login_day==i].login_just_time, login_mar[login_mar.login_day==i].login_count, alpha=.5)
plt.title('Mar')
plt.ylabel('login count')
fig4, ax4 = plt.subplots(figsize=(15,5))
for i in login_april.login_day.unique():
    plt.plot(login_april[login_april.login_day==i].login_just_time, login_april[login_april.login_day==i].login_count)
plt.title('Apr')
plt.xlabel('login time (15 min)')
plt.ylabel('login count')
plt.xlim(0)

login_3h = logins.resample('3H', on='login_time').count()
login_3h.columns = ['login_count']
login_3h.reset_index(inplace=True)
login_3h['login_just_time'] = login_3h.login_time.dt.time
login_3h['time_string'] = login_3h.login_just_time.astype(str)
login_3h['login_day'] = login_3h.login_time.dt.day
login_3h['login_month'] = login_3h.login_time.dt.month
login_3h.head()
#login_3h.info()

early_logins = login_3h[login_3h.time_string == '03:00:00']
jan_early_logins = early_logins[early_logins.login_month==1]
jan_early_logins.head()
sns.barplot(x='login_day', y='login_count', data=jan_early_logins)
plt.title('January Login Counts 3am-6am')
plt.ylabel('login counts')
plt.xlabel('day of month')
#weekends are where those early morning login spikes are coming form

















import json 
data = json.load(open('ultimate_data_challenge.json'))
riders = pd.DataFrame(data)
riders.info()

riders.last_trip_date = pd.to_datetime(riders.last_trip_date, format="%Y-%m-%d")
riders.signup_date = pd.to_datetime(riders.signup_date, format="%Y-%m-%d")
riders.signup_date = riders.signup_date.dt.day
riders.info()
#probably want to set avg_rating missing values to either 0 or the mean. 

lastday = riders.last_trip_date.max() - pd.Timedelta('30 days')
print(lastday)
print(lastday + pd.Timedelta('30 days'))
#we can infer from this that the data was pulled on 7/1/2014 and any rider who has used the service since 6/1/2014 is retained

riders['retained'] = (riders.last_trip_date > lastday)

riders.head()

print('retention rate:', len(riders[riders.retained == True])/len(riders.retained)*100, '%')



#riders.phone.fillna(0, inplace=True)
#riders.phone.replace(0,'No Phone', inplace=True)
sns.countplot(x="phone", hue="retained", data=riders)
plt.title('Phone Type and Retainment')

sns.countplot(x="city", hue="retained", data=riders)
plt.title('City and Retainment')

sns.countplot(x="retained", hue="ultimate_black_user", data=riders)

riders.avg_rating_of_driver.fillna(0, inplace=True)
riders.avg_rating_by_driver.fillna(0, inplace=True)
sns.violinplot(x='retained', y='avg_rating_of_driver', data=riders)
plt.title('Average Rating of Driver and Retainment')

sns.violinplot(x='retained', y='avg_rating_by_driver', data=riders)
plt.title('Average Rating by Driver and Retainment')

riders['rating_of_by_driver'] = (riders.avg_rating_of_driver * riders.avg_rating_by_driver) ** 3
plt.hist(riders[riders.rating_of_by_driver.isnull() == 0].rating_of_by_driver)

riders.avg_rating_by_driver.min()

sns.violinplot(x='retained', y='rating_of_by_driver', data=riders)

#plt.hist(riders.avg_dist)
riders['avg_dist_log'] = np.log((1+ riders.avg_dist))
plt.hist(riders.avg_dist_log)

#riders_under50 = riders[riders.avg_dist < 30]
sns.violinplot(x='retained', y='avg_dist', data=riders)
plt.title('Average Distance (<30 mi) and Retainment')

sns.violinplot(x='retained', y='weekday_pct', data=riders)

riders_30_days = riders[riders.trips_in_first_30_days < 20]

sns.violinplot(x='retained', y='trips_in_first_30_days', data=riders_30_days)
plt.title('Trips in first 30 Days and Retainment')

sns.violinplot(x='retained', y='signup_date', data=riders)
plt.title('Signup Date and Retainment')

riders.head()

y = riders.retained
#X = riders.drop(['retained', 'last_trip_date', 'avg_rating_by_driver', 'avg_rating_of_driver'], 1)
X = riders.drop(['retained', 'last_trip_date', 'avg_dist'], 1)

#riders.rating_of_by_driver.replace('NaN', 0, inplace=True)
X = X.drop('phone', 1)
X_phone = pd.get_dummies(riders.phone, drop_first=False)
X = pd.get_dummies(X, drop_first=True)
X['android'] = X_phone.Android
X['iphone'] = X_phone.iPhone
X.avg_rating_by_driver = X.avg_rating_by_driver ** 3
X.avg_rating_by_driver.fillna(X.avg_rating_by_driver.mean(), inplace=True)
X.avg_rating_of_driver.fillna(X.avg_rating_of_driver.mean(), inplace=True)
#X = X.drop(['retained', 'last_trip_date', 'avg_rating_by_driver', 'avg_rating_of_driver', 'rating_of_driver_cubed'], 1)
X.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

Cs = [0.001, 0.003, 0.1, 0.03, 1, 3, 10, 30, 100]

param_grid = {'C':Cs}
logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

logreg_cv.fit(X.values, y.values)

print(logreg_cv.best_params_) 
print(logreg_cv.best_score_)

# Split the data into a training and test set.
Xlr, Xtestlr, ylr, ytestlr = train_test_split(X.values, y.values, test_size=.3, random_state=42)

clf = LogisticRegression(C=0.03)
# Fit the model on the trainng data.
clf.fit(Xlr, ylr)

ylf_pred=clf.predict(Xtestlr)
# Print the accuracy from the testing data.
print(accuracy_score(ylf_pred, ytestlr))
confusion_matrix(ytestlr, ylf_pred)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Xrf, Xtestrf, yrf, ytestrf = train_test_split(X.values, y.values, test_size=.3, random_state=42)
estimators = range(10, 300, 10)
accuracy = []
for i in estimators:
    rf = RandomForestClassifier(n_estimators = i)
    rf.fit(Xrf, yrf)
    accuracy.append(accuracy_score(rf.predict(Xtestrf), ytestrf))
    #print(accuracy_score(rf.predict(Xtestrf), ytestrf))
    #print(rf.feature_importances_) 

plt.plot(estimators, accuracy)
plt.title('avg_rating_by_driver cubed, log of avg_dist')

from sklearn.metrics import confusion_matrix
Xrf, Xtestrf, yrf, ytestrf = train_test_split(X.values, y.values, test_size=.3, random_state=42)

rf = RandomForestClassifier(n_estimators = 150)
rf.fit(Xrf, yrf)
y_pred = rf.predict(Xtestrf)
print(X.columns)
print(accuracy_score(y_pred, ytestrf))
print(rf.feature_importances_) 
confusion_matrix(ytestrf, y_pred)

#true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}

X.head()# .772



