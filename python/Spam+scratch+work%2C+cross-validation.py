import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Grab and process the raw data.
data_path = ("https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/"
             "master/sms_spam_collection/SMSSpamCollection"
            )
sms_raw = pd.read_csv(data_path, delimiter= '\t', header=None)
sms_raw.columns = ['spam', 'message']

keywords = ['click', 'offer', 'winner', 'buy', 'free', 'cash', 'urgent', 'XXX', 'win','cash','contract','mobile','CASH','WINNER!!','URGENT!','urgent','Urgent']

for key in keywords:
    # Note that we add spaces around the key so that we're getting the word,
    # not just pattern matching.
    sms_raw[str(key)] = sms_raw.message.str.contains(
        str(key) + ' ',
        case=False
    )

sms_spam = sms_raw.query('spam == True')
sms_spam.head(50)

sms_raw['allcaps'] = sms_raw.message.str.isupper()

sms_raw.head()

#Before we go further, let's turn the spam column into a boolean so we can easily do some statistics to prepare for modeling.
sms_raw['spam'] = (sms_raw['spam'] == 'spam')
# Note that if you run this cell a second time everything will become false.
# So... Don't.

sms_raw.head()

data = sms_raw[keywords + ['allcaps']]
target = sms_raw['spam']

data.head()

# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))

print(1-467/5572)

from sklearn.metrics import confusion_matrix
confusion_matrix(target, y_pred)

                # SPAM
#Predicter     True         False
#True           HIT        False Neg
#False         False Pos      HIT

# translation - there are 4748 + 77 = ... HAM messages, I've only mis-identified 77 of those -- FALSE POSITIVE, type 1 error
# there are 390 + 357 = ... SPAM messages, I've mis-identified over half of those (390) - miss, type 2 error, false negative

#390 of my 467 errors are failing to identify spam.
#Sensitivity is percentage of positives correctly identified
print(357/(390+357))

# Specificity__is the percentage of negatives correctly identified, 4748/4825 or,
print(4748/4825)
# Note that I did worse here than the example but much better on "sensitivity" which makes sense

# add the y_pred to my dataframe
sms_raw['y_pred'] = y_pred

sms_raw.columns

sms_compare= sms_raw[['spam','y_pred']]

spamct = sms_compare.query('spam == True & y_pred == True')
print('Number of actual spam, categorized as spam')
print(spamct.count())
spamctmiss = sms_compare.query('spam == True & y_pred == False')
print('Number of actual spam, NOT categorized as spam -- False Negative')
print(spamctmiss.count())
#Sensitivity is percentage of positives correctly identified
# So sensitivity for me is:
print(spamct.count()/(spamct.count()+spamctmiss.count()))


hamct = sms_compare.query('spam == False & y_pred == False')
print('Number of actual ham, categorized as ham')
print(hamct.count())
hamctmiss = sms_compare.query('spam == False & y_pred == True')
print('Number of actual ham, categorized as spam -- False Positive')
print(hamctmiss.count())
#Sensitivity is percentage of positives correctly identified
# So sensitivity for me is:
print(hamct.count()/(hamct.count()+hamctmiss.count()))

sms_raw.head(1)

#dataframe = sms_raw

msk = np.random.rand(len(sms_raw)) < 0.8
train = sms_raw[msk]
test = sms_raw[~msk]

print(len(train))
print(len(test))

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points : {}".format(
    data2.shape[0],
    (test['spam'] != y_pred).sum()
))

print(1-108/1145)

#Make 5 folds
val1, val2, val3, val4, val5 = np.split(sms_raw.sample(frac=1), [int(.2*len(sms_raw)), int(.4*len(sms_raw)),int(.6*len(sms_raw)),int(.8*len(sms_raw))])

test = val1
rest = [val2,val3,val4,val5]
train = pd.concat(rest)

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points in {} : {}".format(
    data2.shape[0],'val1',
    (test['spam'] != y_pred).sum()
))

test = val2
rest = [val1,val3,val4,val5]
train = pd.concat(rest)

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points in {} : {}".format(
    data2.shape[0],'val2',
    (test['spam'] != y_pred).sum()
))

test = val3
rest = [val1,val2,val4,val5]
train = pd.concat(rest)

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points in {} : {}".format(
    data2.shape[0],'val3',
    (test['spam'] != y_pred).sum()
))

test = val4
rest = [val1,val2,val3,val5]
train = pd.concat(rest)

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points in {} : {}".format(
    data2.shape[0],'val4',
    (test['spam'] != y_pred).sum()
))

test = val5
rest = [val1,val2,val3,val4]
train = pd.concat(rest)

data = train[keywords]
data2 = test[keywords]
target = train['spam']
# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data2)

# Display our results.
print("Number of mislabeled points out of a total {} test points in {} : {}".format(
    data2.shape[0],'val5',
    (test['spam'] != y_pred).sum()
))

#So my results for my 5 folds were: 95,90, 103,  97,  86
err = [95,90, 103,  97,  86]
avg_error = np.mean(err)
print(avg_error)
sd_error = np.std(err)
print(sd_error)



