import pandas as pd

#Read in data, print shape
loans_2007 = pd.read_csv("loans_2007.csv")
print("DF SHAPE pre cleanse: ",loans_2007.shape, '\n')

#Drop columns with more than half of values missing
loans_2007.drop_duplicates()
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
print("DF SHAPE post cleanse: ",loans_2007.shape, '\n')

#Review column descriptions to understand feature
file = pd.ExcelFile('col_descrip.xlsx')
col_descrip = file.parse('Sheet1')
col_descrip.iloc[0:5,]

first=[]
second=[]
third=[]
dtype=[]
for col in loans_2007.columns:
    first.append(loans_2007[col][0])
    second.append(loans_2007[col][1])
    third.append(loans_2007[col][2])
    dtype.append(type(loans_2007[col][0]))

first_rows = pd.DataFrame({'Columns':loans_2007.columns,'type':dtype,'first':first,'second':second,'third':third})
col_descrip = pd.merge(col_descrip,first_rows, on='Columns')

#print column names and descriptions for review
col_descrip

#Remove columns not useful for our analysis
drop_cols = ["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d",
             "zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
             "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"]
             

print("DF SHAPE pre cleanse: ",loans_2007.shape, '\n')
loans_2007 = loans_2007.drop(drop_cols, axis=1)
print("DF SHAPE post cleanse: ",loans_2007.shape, '\n')

#Print unique values in target column (loan_status), and proportion % of each
target = loans_2007['loan_status'].value_counts(dropna=False)

percent_total = []
for i in range(0,10):
    percent_total.append(target[i]/loans_2007.shape[0])

pd.DataFrame({'Result of Loan':target.index,'Number of Obs':target.values, 'Percent Total':percent_total})

#Select only rows in target column with 'Full Paid' or 'Charged Off' values, replace with 1 and 0
#select only rows with 'Fully Paid' or 'Charged Off' in loan_status column
print("DF SHAPE pre cleanse: ",loans_2007.shape, '\n')
loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | (loans_2007['loan_status'] == "Charged Off")]

#map replace binary data (1,0)
status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}

loans_2007 = loans_2007.replace(status_replace)

#print sample of new loan_status column
print("DF SHAPE post cleanse: ",loans_2007.shape, '\n')
print(loans_2007['loan_status'].head())

#Drop columns that only have only one unique value
print("DF SHAPE pre cleanse: ",loans_2007.shape, '\n')
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans = loans_2007.drop(drop_columns, axis=1)
print("Dropped columns: ", drop_columns, '\n')
print("DF SHAPE post cleanse: ",loans.shape, '\n')

#Handling missing values
# 1. remove columns entirely where more than 1% of the rows for that column contain a null value.
# 2. remove the remaining rows containing null values.

#create missing value count
null_counts = loans.isnull().sum()
print(null_counts)

#remove 'pub_rec_bankruptcies'
print("DF SHAPE post cleanse: ",loans.shape, '\n')
loans = loans.drop("pub_rec_bankruptcies", axis=1)
loans = loans.dropna(axis=0)
print("DF SHAPE post cleanse: ",loans.shape, '\n')

#Converting values to numeric
# 1. identify columns with non-numeric types
# 2. delete columns that can't or shouldn't be converted to either numeric or dummy vars.
# 3. transorm remaining columns to either numeric or categorical dummy variables

#how many object columns are there?
print(loans.dtypes.value_counts())

#isolate object type columns, and select sample rows
object_columns_df = loans.select_dtypes(include=["object"])
print(object_columns_df.iloc[1])

#create emp_length mapping dict
mapping_dict = {"emp_length": 
        {"10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0}}
loans = loans.replace(mapping_dict)

#drop date columns (requires too much feature engineering for now...) and addr_state (it would require 49 dummy variabbles)
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)

#convert int_rate and revol_util column to numeric
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")

#define categorical variable columns
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]

#create dummy variables
dummy_df = pd.get_dummies(loans[cat_columns])

#concat dummy_df and loans. Drop original categorical columns. print loans df columns
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)
print(loans.columns, '\n')

print("DF SHAPE post cleanse: ",loans.shape)

loans.shape

#Create Error Metric filters function
import pandas as pd
import numpy

#create fpr,tpr formula
def fpr_tpr(preds,target):
    # False positives.
    fp_filter = (preds == 1) & (target == 0)
    fp = len(preds[fp_filter])

    # True positives.
    tp_filter = (preds == 1) & (target == 1)
    tp = len(preds[tp_filter])

    # False negatives.
    fn_filter = (preds == 0) & (target == 1)
    fn = len(preds[fn_filter])

    # True negatives
    tn_filter = (preds == 0) & (target == 0)
    tn = len(preds[tn_filter])

    # Rates
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr,fpr

import pandas as pd

#Read in cleaned data, print shape
loans2 = pd.read_csv("cleaned_loans_2007.csv")
loans2.shape

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt

#initialize
lr = LogisticRegression()
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]

#train model
kf = KFold(features.shape[0],n_folds=3,random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

#Use tpr/fpr function
tpr1, fpr1 = fpr_tpr(predictions,loans["loan_status"])

#plot the ROC
probs = cross_val_predict(lr, features, target, cv=kf, method='predict_proba')
falp,trup,thresholds = metrics.roc_curve(target,probs[:,1])
y=list(range(1,65))
rand=[x*(1/64) for x in y]
plt.plot(rand,rand)
plt.plot(falp,trup)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Reciever Operator Curve')

print('True Positive Rate: ',tpr1)
print('False Positive Rate: ',fpr1)

plt.show()

from sklearn.metrics import roc_auc_score
auc_score = metrics.roc_auc_score(target,probs[:,1])
print("AUC Score: ", auc_score)

fpr_tpr(predictions,loans["loan_status"])

features

# Import EarlyStopping
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model, SET THE CALLBACKS ARG AS EARLY STOPPING AND SET EPOCHS TO 30
model.fit(predictors, target, epochs=30, validation_split=0.30, callbacks=[early_stopping_monitor])

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression(class_weight="balanced")
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

#Use tpr/fpr function
tpr2, fpr2 = fpr_tpr(predictions,loans["loan_status"])

# plot the ROC
probs = cross_val_predict(lr, features, target, cv=kf, method='predict_proba')
fal_p,tru_p,thresholds = metrics.roc_curve(target,probs[:,1])
y=list(range(1,65))
rand=[x*(1/64) for x in y]
plt.plot(rand,rand)
plt.plot(fal_p,tru_p)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Reciever Operator Curve')

print('True Positive Rate: ',tpr2)
print('False Positive Rate: ',fpr2)

plt.show()

from sklearn.metrics import roc_auc_score
auc_score = metrics.roc_auc_score(target,probs[:,1])
print("AUC Score: ", auc_score)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict

#set the penalty to 10x!
custom_penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight=custom_penalty)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

#Use tpr/fpr function
tpr3, fpr3 = fpr_tpr(predictions,loans["loan_status"])

# plot the ROC
probs = cross_val_predict(lr, features, target, cv=kf, method='predict_proba')
fal_p,tru_p,thresholds = metrics.roc_curve(target,probs[:,1])
y=list(range(1,65))
rand=[x*(1/64) for x in y]
plt.plot(rand,rand)
plt.plot(fal_p,tru_p)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Reciever Operator Curve')

print('True Positive Rate: ',tpr3)
print('False Positive Rate: ',fpr3)

plt.show()

from sklearn.metrics import roc_auc_score
auc_score = metrics.roc_auc_score(target,probs[:,1])
print("AUC Score: ", auc_score)

pd.DataFrame({"Model 1":[tpr1,fpr1],"Model 2":[tpr2,fpr2],"Model 3":[tpr3,fpr3]},index=['True Pos. Rate','False Pos. Rate'])

