# We begin by importing numpy and pandas, as usual.
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We'll be evaluating our models as well.
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import roc_curve

train_df = pd.read_csv('./data/flint_train.csv')
test_df = pd.read_csv('./data/flint_test.csv')
train_df.head()

dummy_columns = ['Owner_Type', 'Residential_Building_Style', 'USPS_Vacancy', 
                 'Building_Storeys', 'Rental', 'Use_Type', 'Prop_Class', 'Zoning', 'Future_Landuse', 'DRAFT_Zone',
                 'Housing_Condition_2012', 'Housing_Condition_2014','Hydrant_Type', 'Ward', 'PRECINCT', 'CENTRACT', 
                 'Commercial_Condition_2013','CENBLOCK', 'SL_Type', 'SL_Type2', 'SL_Lead', 'Homestead']

drop_columns = ['sample_id', 'google_add', 'parcel_id', 'Date_Submitted']

combined_df = train_df.append(test_df)

combined_df = combined_df.drop(drop_columns, axis=1)
combined_df = pd.get_dummies(combined_df, columns = dummy_columns)

train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

# The combining of the dataframes created an empty column for lead in test_df.  We drop it here.
test_df = test_df.drop('Lead_(ppb)', axis=1)

#train_df = train_df.drop(drop_columns, axis=1)
#train_df = pd.get_dummies(train_df, columns=dummy_columns)

from sklearn.cross_validation import train_test_split

Ydata_r = train_df['Lead_(ppb)']
Ydata_c = train_df['Lead_(ppb)'] > 15
Xdata = train_df.drop('Lead_(ppb)', axis=1)

# We'll be starting with a regression problem, so split on Ydata_r

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata_c)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(Xtrain, Ytrain)
yhat1 = dt.predict_proba(Xtest)

fig = plt.figure()
fig.set_size_inches(8,8)

fpr, tpr, _ = roc_curve(Ytest, yhat1[:,1])
plt.plot(fpr, tpr, label= 'Decision Tree (area = %0.5f)' % roc(Ytest, yhat1[:,1]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Service Line Classifiers')
plt.legend(loc="lower right")

plt.show()

0.3*0.65*0.6*0.5, 0.7*0.35*0.4*0.45

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1024, n_jobs=-1)
rf.fit(Xtrain, Ytrain)
yhat2 = rf.predict_proba(Xtest)

fig = plt.figure()
fig.set_size_inches(8,8)

fpr, tpr, _ = roc_curve(Ytest, yhat2[:,1])
plt.plot(fpr, tpr, label= 'Random Forest (area = %0.5f)' % roc(Ytest, yhat2[:,1]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Service Line Classifiers')
plt.legend(loc="lower right")

plt.show()

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=4096, n_jobs=-1)
et.fit(Xtrain, Ytrain)
yhat3 = et.predict_proba(Xtest)

fig = plt.figure()
fig.set_size_inches(8,8)

fpr, tpr, _ = roc_curve(Ytest, yhat3[:,1])
plt.plot(fpr, tpr, label= 'Random Forest (area = %0.5f)' % roc(Ytest, yhat3[:,1]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Service Line Classifiers')
plt.legend(loc="lower right")

plt.show()

# Now let's make a submission together.
# The logistic regression model was probably best, so we'll use that.

rf = RandomForestClassifier(n_estimators=1024, n_jobs=-1)
rf.fit(Xdata, Ydata_c)
#yhat2 = rf.predict_proba(Xtest)

yhat = rf.predict_proba(test_df)

# The predict_proba method produces an N x 2 vector that has the probability of
# label 0 in the first columna and the probability of label 1 in the second column.
# The submission asks for the probability of 1, we we pull that out.

pred = yhat[:,1]

# Finally, the submission asks for the sample ids of the predictions.  We'll get
# those now and then create the submission data frame.

sample_ids = pd.read_csv('./data/flint_test.csv', usecols=['sample_id'])

submission_df = pd.DataFrame({'sample_id':sample_ids['sample_id'], 'Lead_gt_15':pred})
submission_df.to_csv('./data/submission.csv', index=False)



