get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.plot(np.arange(5))

df = pd.read_csv("./data/train.csv")

df.head(10)

df.describe()

df.isnull().sum()

df['Property_Area'].value_counts()

df['ApplicantIncome'].hist(bins=50)

df.boxplot(column='ApplicantIncome')

df.boxplot(column='ApplicantIncome', by='Education')

df['LoanAmount'].hist(bins=50)

df.boxplot(column='LoanAmount')

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status', index='Credit_History', aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean())
print('Frequency Table for Credit History:')
print(temp1)
print('\nProbability of Getting a Loan for Each Credit History Class')
print(temp2)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)

df['Self_Employed'].value_counts()

df['Self_Employed'].fillna('No', inplace=True)

table=df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
#Define a function to return value of this pivot_table
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]
#Fill in missing values using a categorically relevant median
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

df.isnull().sum()

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

df['Gender'].value_counts()

df['Married'].value_counts()

df['Married'].fillna('Yes', inplace=True)

df['Dependents'].value_counts()

df['Dependents'].fillna(0, inplace=True) # Clear majority of applicants have 0 dependents, so we'll fill using that

df['Credit_History'].value_counts()

df[(df['Credit_History'].isnull())].describe()

df['LoanOverTotal'] = df['LoanAmount'] / df['TotalIncome']

df[(df['Credit_History'].isnull())].sort_values(by='Loan_Status')

def fillCredit(x):
    if (x['LoanOverTotal'] > .025) | (x['TotalIncome'] > 5000):
        return 1
    else:
        return 0

df['Credit_History'].fillna(df[df['Credit_History'].isnull()].apply(fillCredit, axis=1), inplace=True)

df.isnull().sum()

print(df['Loan_Amount_Term'].mean())
print(df['Loan_Amount_Term'].median())

df['Loan_Amount_Term'].hist(bins=10)

df['Loan_Amount_Term'].fillna(360, inplace=True)

df['Gender'].fillna('Male', inplace=True)

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for item in var_mod:
    df[item] = le.fit_transform(df[item])

df.loc[df['Dependents'] == "3+", 'Dependents'] = 3

df['Dependents'].value_counts()

df.loc[df['Dependents'] == '0', 'Dependents'] = 0

df.head(10)

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

def classification_model(model, data, predictors, outcome):
    #Fit the model
    model.fit(data[predictors], data[outcome])
    #make predictions on the training set
    predictions = model.predict(data[predictors])
    #print accuracy (This should be a really useful tutorial, since it seems to have everything neatly organized)
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error =[]
    for train, test in kf.split(data):
        #filter training data
        train_predictors = (data[predictors].ix[train,:])
        # The target we're using to train the algorithm.
        train_target = data[outcome].ix[train]
        #Training the algorithm using the predictors and target
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run
        error.append(model.score(data[predictors].ix[test,:], data[outcome].ix[test]))
    print("Cross-validation score: %s" % "{0:.3%}".format(np.mean(error)))
    #Fit the model again so it can be referred to outside the function
    model.fit(data[predictors], data[outcome])    

outcome_var = "Loan_Status"
model = LogisticRegression()
predictor_var = ["Credit_History", "Education"]
classification_model(model, df, predictor_var, outcome_var)

predictor_var = ["Credit_History", "Education", "Married", "Property_Area"]
classification_model(model, df, predictor_var, outcome_var)

predictor_var = ["Credit_History", "Education", "Married", "TotalIncome"]
classification_model(model, df, predictor_var, outcome_var)

model = DecisionTreeClassifier()
predictor_var = ["Credit_History","Gender", "Education", "Married"]
classification_model(model, df, predictor_var, outcome_var)

predictor_var = ["Credit_History","Loan_Amount_Term", "LoanAmount_log"]
classification_model(model, df, predictor_var, outcome_var)

model = RandomForestClassifier(n_estimators=100)
predictor_var = ["Gender","Married","Dependents","Education","Self_Employed", "Credit_History","Loan_Amount_Term", "LoanAmount_log", "Property_Area", "TotalIncome_log"]
classification_model(model, df, predictor_var, outcome_var)

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

model = RandomForestClassifier(n_estimators=50, min_samples_split=10, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Property_Area','Dependents']
classification_model(model, df,predictor_var,outcome_var)

test_df = pd.read_csv("./data/test.csv")  #read in the test set. 
test_df.head(10)

test_df.isnull().sum()

df['Loan_Amount_Term'].hist(bins=10)

test_df['Loan_Amount_Term'].fillna(360, inplace = True)

df['Self_Employed'].value_counts()

test_df['Self_Employed'].fillna('No', inplace=True)

test_df['LoanAmount'].fillna(test_df[test_df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

test_df['LoanAmount_log'] = np.log(test_df['LoanAmount'])

test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']
test_df['TotalIncome_log'] = np.log(test_df['TotalIncome'])
test_df['TotalIncome_log'].hist(bins=20)

test_df.isnull().sum()

test_df['Gender'].fillna('Male', inplace=True)

test_var_mod = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for item in test_var_mod:
     test_df[item] = le.fit_transform(test_df[item])

test_df['LoanOverTotal'] = test_df['LoanAmount'] / test_df['TotalIncome']

test_df['Credit_History'].fillna(test_df[test_df['Credit_History'].isnull()].apply(fillCredit, axis=1), inplace=True)

test_df.isnull().sum()

test_df['Dependents'].value_counts()

test_df['Dependents'].fillna(0, inplace=True)

test_df.loc[test_df['Dependents'] == "3+", 'Dependents'] = 3

test_df.loc[test_df['Dependents'] == "0", 'Dependents'] = 0

test_df.loc[test_df['Dependents'] == "1", 'Dependents'] = 1
test_df.loc[test_df['Dependents'] == "2", 'Dependents'] = 2

test_df.isnull().sum()

test_df.head(10)

model = RandomForestClassifier(n_estimators=50, min_samples_split=10, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Property_Area','Dependents']
model.fit(df[predictor_var],df['Loan_Status'])

predictions = model.predict(test_df[predictor_var])

predictions[:10]

submission = pd.DataFrame({
        "Loan_ID": test_df["Loan_ID"],
        "Loan_Status": predictions
    })

submission.to_csv('CN_Loan_Submission.csv', index=False)

predictions.dtype

submission.loc[submission['Loan_Status'] == 1, 'Loan_Status'] = "Y"
submission.loc[submission['Loan_Status'] == 0, 'Loan_Status'] = 'N'

submission.head()



