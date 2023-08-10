import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import preprocessing
color = sns.color_palette()
get_ipython().magic('matplotlib inline')

df_train = pd.read_json("train.json")
df_test = pd.read_json("test.json")

int_level = df_train['interest_level'].value_counts()
plt.pie(int_level,labels= ['low','medium','high'], autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

drace_df = df_train.loc[:,['interest_level', 'created', 'manager_id', 'bathrooms']]

drace_df.head()

drace_df.describe()

drace_df.manager_id.value_counts().head()

drace_df.bathrooms.value_counts()

drace_df.created = pd.to_datetime(drace_df.created)

drace_df['date_created'] = drace_df.created.dt.date
drace_df['weekday_created'] = drace_df.created.dt.dayofweek
drace_df['hour_created'] = drace_df.created.dt.hour

drace_df.head()

drace_df['interest_level'] = pd.Categorical(drace_df['interest_level'], categories= ['low', 'medium', 'high'], ordered=True)

weekday_listings = drace_df['weekday_created'].value_counts()
sns.barplot(weekday_listings.index, weekday_listings.values, alpha=0.8)
#not sure this showed much.  A little as expected.

daily_listings_intrMode = drace_df.groupby(['date_created'])['interest_level'].value_counts()

daily_listings_intrMode = pd.DataFrame(daily_listings_intrMode)
daily_listings_intrMode = daily_listings_intrMode.unstack(level=1)

daily_listings_intrMode.head()

daily_listings_intrMode.plot(kind='bar', figsize = (20,12))

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_train['manager_id'].values))
df_train['manager_id'] = lbl.transform(list(df_train['manager_id'].values))

lbl = preprocessing.LabelEncoder()
lbl.fit(list(drace_df['manager_id'].values))
drace_df['manager_id'] = lbl.transform(list(drace_df['manager_id'].values))

df_train["num_photos"] = df_train["photos"].apply(len)
df_train["num_features"] = df_train["features"].apply(len)
df_train["num_description_words"] = df_train["description"].apply(lambda x: len(x.split(" ")))

features_to_use = ["latitude", "longitude", "price",
                   "num_photos", "num_features", "num_description_words"
                   ]
features_to_use.append('manager_id')

X = df_train[features_to_use]
y = df_train["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# compute fractions and count for each manager
temp = pd.concat([X_train.manager_id,pd.get_dummies(y_train)], axis = 1).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = X_train.groupby('manager_id').count().iloc[:,1]

# remember the manager_ids look different because we encoded them in the previous step 
print(temp.tail(10))

# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

# get ixes for unranked managers...
unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
print(temp.tail(10))

drace_df = drace_df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = drace_df['high_frac'].isnull()
drace_df.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
drace_df.head()

# add the features computed on the training dataset to the validation dataset if we want to use the entire dataframe, 
#I didn't use it
X_val = X_val.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = X_val['high_frac'].isnull()
X_val.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
X_val.head()

feats_used = ['bathrooms', 'hour_created', 'manager_id']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train[feats_used], y_train)
y_pred = rf.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred)

pd.Series(index = feats_used, data = rf.feature_importances_).sort_values().plot(kind = 'bar')

drace_df.manager_skill.describe()

feats_used = ['bathrooms', 'hour_created', 'manager_skill']
x = drace_df[feats_used]
y = drace_df["interest_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train[feats_used], y_train)
y_pred = rf.predict_proba(x_test[feats_used])
log_loss(y_test, y_pred)

pd.Series(index = feats_used, data = rf.feature_importances_).sort_values().plot(kind = 'bar')



